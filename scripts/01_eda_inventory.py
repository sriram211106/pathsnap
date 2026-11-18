"""Step 1a - Dataset inventory and data-quality checks (train + student_test).

Run from the repo root:
    python scripts/01_eda_inventory.py --data-root data/raw --out outputs/eda

Writes:
    outputs/eda/trip_inventory.csv   one row per trip (files, counts, gaps, network, route checks)
    outputs/eda/points.parquet       one row per GPS point (dt, implied speed, distance to reference route)
    outputs/eda/summary.txt          partition-level summary printed to screen as well
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely import STRtree, LineString, points as shp_points
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.io import PARTITIONS, list_trips, load_trip  # noqa: E402
from pathsnap.geo import haversine_m, make_projector, project  # noqa: E402


def track_stats(tr: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    s = {}
    s["n_points"] = len(tr)
    s["n_nan_rows"] = int(tr[["lon", "lat", "t"]].isna().any(axis=1).sum())
    bad = (tr["lon"].abs() > 180) | (tr["lat"].abs() > 90) | ((tr["lon"] == 0) & (tr["lat"] == 0))
    s["n_invalid_coords"] = int(bad.sum())
    s["n_duplicate_rows"] = int(tr.duplicated().sum())
    s["n_duplicate_t"] = int(tr["t"].duplicated().sum())

    t = tr["t"].to_numpy()
    dt = np.diff(t)
    s["n_non_increasing_t"] = int((dt <= 0).sum())
    s["share_dt_eq_1"] = float(np.isclose(dt, 1.0).mean()) if len(dt) else np.nan
    s["dt_max"] = float(dt.max()) if len(dt) else np.nan
    s["n_gaps_gt_2s"] = int((dt > 2).sum())
    s["n_gaps_gt_10s"] = int((dt > 10).sum())
    s["n_gaps_gt_30s"] = int((dt > 30).sum())
    s["duration_s"] = float(t[-1] - t[0]) if len(t) else np.nan

    lon, lat = tr["lon"].to_numpy(), tr["lat"].to_numpy()
    step_m = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step_m / dt, np.nan)
    s["track_length_km"] = float(np.nansum(step_m) / 1000)
    s["speed_p50_ms"] = float(np.nanpercentile(speed, 50)) if len(speed) else np.nan
    s["speed_p99_ms"] = float(np.nanpercentile(speed, 99)) if len(speed) else np.nan
    s["speed_max_ms"] = float(np.nanmax(speed)) if len(speed) else np.nan
    s["lon_min"], s["lon_max"] = float(np.nanmin(lon)), float(np.nanmax(lon))
    s["lat_min"], s["lat_max"] = float(np.nanmin(lat)), float(np.nanmax(lat))
    s["lon_center"], s["lat_center"] = float(np.nanmean(lon)), float(np.nanmean(lat))

    # per-point frame (dt/speed refer to the step that ARRIVES at this point; first point = NaN)
    pts = pd.DataFrame({
        "point_idx": np.arange(len(tr)),
        "dt": np.r_[np.nan, dt],
        "step_m": np.r_[np.nan, step_m],
        "speed_ms": np.r_[np.nan, speed],
    })
    return s, pts


def network_stats(nodes: pd.DataFrame, arcs: pd.DataFrame) -> dict:
    s = {}
    n_nodes, n_arcs = len(nodes), len(arcs)
    s["n_nodes"], s["n_arcs"] = n_nodes, n_arcs
    s["n_nan_nodes"] = int(nodes.isna().any(axis=1).sum())
    s["n_nan_arcs"] = int(arcs.isna().any(axis=1).sum())

    a = arcs.dropna().astype(np.int64)
    u, v = a["u"].to_numpy(), a["v"].to_numpy()
    valid = (u >= 0) & (u < n_nodes) & (v >= 0) & (v < n_nodes)
    s["n_arcs_bad_node_ref"] = int((~valid).sum())
    u, v = u[valid], v[valid]
    s["n_self_loops"] = int((u == v).sum())
    key = u * n_nodes + v
    s["n_duplicate_arcs"] = int(len(key) - len(np.unique(key)))
    rev = v * n_nodes + u
    s["share_arcs_with_reverse"] = float(np.isin(rev, key).mean()) if len(key) else np.nan

    used = np.zeros(n_nodes, dtype=bool)
    used[u] = True
    used[v] = True
    s["n_isolated_nodes"] = int((~used).sum())

    g = coo_matrix((np.ones(len(u)), (u, v)), shape=(n_nodes, n_nodes)).tocsr()
    n_wcc, lab_w = connected_components(g, directed=True, connection="weak")
    n_scc, lab_s = connected_components(g, directed=True, connection="strong")
    # ignore isolated nodes when sizing components
    s["n_weak_components"] = int(n_wcc)
    s["largest_wcc_share_of_used_nodes"] = float(np.bincount(lab_w[used]).max() / used.sum()) if used.any() else np.nan
    s["n_strong_components"] = int(n_scc)
    s["largest_scc_share_of_used_nodes"] = float(np.bincount(lab_s[used]).max() / used.sum()) if used.any() else np.nan
    return s


def route_stats(route, nodes, arcs) -> dict:
    s = {"has_route": route is not None}
    if route is None or len(route) == 0:
        s["route_len_arcs"] = 0
        return s
    r = route[~np.isnan(route)].astype(np.int64)
    s["route_len_arcs"] = int(len(r))
    s["route_n_nan"] = int(np.isnan(route).sum())
    s["route_n_unique_arcs"] = int(len(np.unique(r)))
    s["route_n_bad_arc_ref"] = int(((r < 0) | (r >= len(arcs))).sum())
    r = r[(r >= 0) & (r < len(arcs))]
    s["route_n_consecutive_repeats"] = int((r[1:] == r[:-1]).sum())
    au, av = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    # continuity: arc i must end where arc i+1 starts
    s["route_n_continuity_breaks"] = int((av[r[:-1]] != au[r[1:]]).sum())
    nl, nt = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    seg = haversine_m(nl[au[r].astype(int)], nt[au[r].astype(int)], nl[av[r].astype(int)], nt[av[r].astype(int)])
    s["route_length_km"] = float(seg.sum() / 1000)
    return s


def gps_to_route_distance(tr, route, nodes, arcs) -> np.ndarray:
    """Distance (m) from every GPS point to the nearest arc of the reference route.
    This is NOT a label - it measures GPS noise / route plausibility and guides the
    candidate-search radius later. Uses a per-trip UTM projection."""
    if route is None or len(route) == 0:
        return np.full(len(tr), np.nan)
    r = route[~np.isnan(route)].astype(np.int64)
    r = r[(r >= 0) & (r < len(arcs))]
    r = np.unique(r)
    proj = make_projector(tr["lon"].mean(), tr["lat"].mean())
    nx, ny = project(proj, nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    u = arcs["u"].to_numpy()[r].astype(int)
    v = arcs["v"].to_numpy()[r].astype(int)
    lines = [LineString([(nx[a], ny[a]), (nx[b], ny[b])]) for a, b in zip(u, v)]
    tree = STRtree(lines)
    px, py = project(proj, tr["lon"].to_numpy(), tr["lat"].to_numpy())
    ok = np.isfinite(px) & np.isfinite(py)
    out = np.full(len(tr), np.nan)
    idx, dist = tree.query_nearest(shp_points(px[ok], py[ok]), return_distance=True, all_matches=False)
    # idx[0] = position in the query array, dist = distance to nearest line
    tmp = np.full(ok.sum(), np.nan)
    tmp[idx[0]] = dist
    out[ok] = tmp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out", default="outputs/eda")
    args = ap.parse_args()
    root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, point_frames = [], []
    for part in PARTITIONS:
        pdir = root / part
        if not pdir.exists():
            print(f"[warn] missing partition folder: {pdir}")
            continue
        for tid in tqdm(list_trips(pdir), desc=part):
            trip = load_trip(pdir / tid)
            row = {"partition": part, "trip_id": tid}
            for ext, p in trip["files"].items():
                row[f"has_{ext}"] = p.exists()
                row[f"size_kb_{ext}"] = round(p.stat().st_size / 1024, 1) if p.exists() else np.nan
            if trip["track"] is None or trip["nodes"] is None or trip["arcs"] is None:
                rows.append(row)
                continue
            ts, pts = track_stats(trip["track"])
            row.update(ts)
            row.update(network_stats(trip["nodes"], trip["arcs"]))
            row.update(route_stats(trip["route"], trip["nodes"], trip["arcs"]))
            d = gps_to_route_distance(trip["track"], trip["route"], trip["nodes"], trip["arcs"])
            pts["dist_to_route_m"] = d
            row["gps_route_dist_p50_m"] = float(np.nanpercentile(d, 50)) if np.isfinite(d).any() else np.nan
            row["gps_route_dist_p95_m"] = float(np.nanpercentile(d, 95)) if np.isfinite(d).any() else np.nan
            row["gps_route_dist_max_m"] = float(np.nanmax(d)) if np.isfinite(d).any() else np.nan
            row["points_per_route_arc"] = row["n_points"] / max(row.get("route_len_arcs", 0), 1)
            rows.append(row)
            pts.insert(0, "trip_id", tid)
            pts.insert(0, "partition", part)
            point_frames.append(pts)

    inv = pd.DataFrame(rows)
    inv.to_csv(out / "trip_inventory.csv", index=False)
    if point_frames:
        pd.concat(point_frames, ignore_index=True).to_parquet(out / "points.parquet", index=False)

    lines = []
    for part, g in inv.groupby("partition"):
        lines.append(f"\n===== {part}: {len(g)} trips =====")
        cols = ["n_points", "duration_s", "track_length_km", "share_dt_eq_1", "dt_max", "n_gaps_gt_10s",
                "speed_p99_ms", "n_nodes", "n_arcs", "n_isolated_nodes", "n_weak_components",
                "share_arcs_with_reverse", "route_len_arcs", "route_n_continuity_breaks",
                "gps_route_dist_p50_m", "gps_route_dist_p95_m", "points_per_route_arc"]
        cols = [c for c in cols if c in g]
        lines.append(g[cols].describe().T[["min", "50%", "mean", "max"]].round(3).to_string())
        flag_cols = ["n_nan_rows", "n_invalid_coords", "n_duplicate_rows", "n_non_increasing_t",
                     "n_arcs_bad_node_ref", "route_n_bad_arc_ref", "route_n_continuity_breaks"]
        for c in flag_cols:
            if c in g:
                bad = g.loc[g[c] > 0, "trip_id"].tolist()
                lines.append(f"{c:>28}: {len(bad)} trips {bad[:15]}")
        missing = {e: g.loc[~g[f'has_{e}'], 'trip_id'].tolist() for e in ("track", "nodes", "arcs", "route")}
        lines.append(f"missing files: {missing}")
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text)
    print(f"\nWrote {out/'trip_inventory.csv'}, {out/'points.parquet'}, {out/'summary.txt'}")


if __name__ == "__main__":
    main()
