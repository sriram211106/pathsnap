"""Step 1c - Interactive Folium map of one trip: GPS points, local road network, reference route.

Run:
    python scripts/03_eda_folium_map.py --trip-dir data/raw/train/00000001
    python scripts/03_eda_folium_map.py --trip-dir data/raw/student_test/00000000 --buffer-m 200

Output: outputs/eda/maps/<partition>_<trip_id>.html  (open in a browser)

Folium/Leaflet wants (lat, lon); our files store (lon, lat). GeoJSON, however, uses
(lon, lat) - so the network/route layers go in as GeoJSON and the points are swapped by hand.
"""
import argparse
import sys
from pathlib import Path

import folium
import numpy as np
from shapely import LineString, STRtree
from shapely import points as shp_points

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.io import load_trip  # noqa: E402
from pathsnap.geo import make_projector, project  # noqa: E402

LEGEND = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
            padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font-size: 13px;">
<b>Trip {tid} ({part})</b><br>
<span style="color:#9e9e9e;">&#9644;</span> Road network (clipped)<br>
<span style="color:#1b9e3a;">&#9644;</span> Reference route<br>
<span style="color:#1f4fd1;">&#9679;</span> GPS observation (&le; {far} m from route)<br>
<span style="color:#ff8c00;">&#9679;</span> GPS observation &gt; {far} m from route<br>
<span style="color:#d62728;">&#9679;</span> GPS point after a gap &gt; {gap}s<br>
</div>
"""


def arcs_geojson(arc_ids, nodes, arcs, props=None):
    lon, lat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    u, v = arcs["u"].to_numpy().astype(int), arcs["v"].to_numpy().astype(int)
    coords = [[[lon[u[a]], lat[u[a]]], [lon[v[a]], lat[v[a]]]] for a in arc_ids]
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": props or {}, "geometry": {"type": "MultiLineString", "coordinates": coords}}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip-dir", required=True)
    ap.add_argument("--out", default="outputs/eda/maps")
    ap.add_argument("--buffer-m", type=float, default=300.0, help="network kept within this distance of the GPS bbox")
    ap.add_argument("--max-network-arcs", type=int, default=30000)
    ap.add_argument("--gap-s", type=float, default=5.0)
    ap.add_argument("--max-gps-markers", type=int, default=5000)
    ap.add_argument("--far-m", type=float, default=50.0, help="GPS points farther than this from the route are orange")
    args = ap.parse_args()

    trip_dir = Path(args.trip_dir)
    part = trip_dir.parent.name
    trip = load_trip(trip_dir)
    tid, tr, nodes, arcs, route = trip["trip_id"], trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    tr = tr.dropna()

    # clip the (huge) network to a CORRIDOR around the GPS path, in metres.
    # (a bounding box is far too big for long diagonal trips)
    proj = make_projector(tr["lon"].mean(), tr["lat"].mean())
    px, py = project(proj, tr["lon"], tr["lat"])
    nx, ny = project(proj, nodes["lon"], nodes["lat"])
    b = args.buffer_m
    # cheap pre-filter: both endpoints inside the bbox (+buffer)
    inside = (nx >= px.min() - b) & (nx <= px.max() + b) & (ny >= py.min() - b) & (ny <= py.max() + b)
    u, v = arcs["u"].to_numpy().astype(int), arcs["v"].to_numpy().astype(int)
    pre = np.flatnonzero(inside[u] | inside[v])
    # exact filter: arc segment lies within b metres of at least one GPS point
    # (point-wise 'dwithin' query; buffering a 19k-vertex path is far too slow)
    segs = [LineString([(nx[u[a]], ny[u[a]]), (nx[v[a]], ny[v[a]])]) for a in pre]
    tree = STRtree(segs)
    step = max(1, len(px) // 5000)  # GPS points ~1 s apart; every k-th point is enough for clipping
    hits = tree.query(shp_points(px[::step], py[::step]), predicate="dwithin", distance=b)
    keep = np.sort(pre[np.unique(hits[1])])
    print(f"[info] {len(keep)} network arcs within {b:.0f} m of the GPS path")
    # two-way streets are two arcs; draw each street once
    pair = np.minimum(u[keep], v[keep]) * len(nodes) + np.maximum(u[keep], v[keep])
    _, first = np.unique(pair, return_index=True)
    keep = keep[np.sort(first)]
    if len(keep) > args.max_network_arcs:
        print(f"[info] {len(keep)} network arcs in area; drawing first {args.max_network_arcs}")
        keep = keep[: args.max_network_arcs]

    # Esri tiles: no API key, and (unlike tile.openstreetmap.org) they load from a local file:// HTML
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Esri street map").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Esri satellite").add_to(m)
    folium.GeoJson(arcs_geojson(keep, nodes, arcs), name=f"Road network ({len(keep)} streets)",
                   style_function=lambda _: {"color": "#9e9e9e", "weight": 1, "opacity": 0.7}).add_to(m)

    # distance of each GPS point to the nearest reference-route arc (m) -> colour far points
    dist = np.full(len(tr), np.nan)
    r = None
    if route is not None and len(route):
        r = route[~np.isnan(route)].astype(int)
        ru = np.unique(r)
        rsegs = [LineString([(nx[u[a]], ny[u[a]]), (nx[v[a]], ny[v[a]])]) for a in ru]
        idx, dd = STRtree(rsegs).query_nearest(shp_points(px, py), return_distance=True, all_matches=False)
        dist[idx[0]] = dd
        n_far = int((dist > args.far_m).sum())
        print(f"[info] {n_far} of {len(tr)} GPS points are > {args.far_m:.0f} m from the reference route")

    dt = np.r_[0, np.diff(tr["t"].to_numpy())]
    # very long trips: draw every k-th point (gap points always drawn) so the HTML stays usable
    k = max(1, len(tr) // args.max_gps_markers)
    show = (np.arange(len(tr)) % k == 0) | (dt > args.gap_s)
    label = f"GPS observations ({len(tr)}" + (f", every {k}th shown)" if k > 1 else ")")
    gps = folium.FeatureGroup(name=label)
    for i, (lo, la, t, d, rd) in enumerate(zip(tr["lon"], tr["lat"], tr["t"], dt, dist)):
        if not show[i]:
            continue
        gap = d > args.gap_s
        far = rd > args.far_m
        color = "#d62728" if gap else ("#ff8c00" if far else "#1f4fd1")
        folium.CircleMarker([la, lo], radius=4 if (gap or far) else 2.5, color=color,
                            fill=True, fill_opacity=0.9, weight=0,
                            tooltip=f"idx {i} | t={t:.0f}s | dt={d:.0f}s | {rd:.0f} m from route").add_to(gps)
    gps.add_to(m)

    # route drawn AFTER the points so it stays visible on top of them
    if r is not None:
        folium.GeoJson(arcs_geojson(r, nodes, arcs), name=f"Reference route ({len(r)} arcs)",
                       style_function=lambda _: {"color": "#1b9e3a", "weight": 4, "opacity": 0.9}).add_to(m)

    folium.Marker([tr["lat"].iloc[0], tr["lon"].iloc[0]], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([tr["lat"].iloc[-1], tr["lon"].iloc[-1]], tooltip="End", icon=folium.Icon(color="red")).add_to(m)

    m.fit_bounds([[tr["lat"].min(), tr["lon"].min()], [tr["lat"].max(), tr["lon"].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(tid=tid, part=part, gap=args.gap_s, far=int(args.far_m))))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{part}_{tid}.html"
    m.save(str(path))
    print("saved", path)


if __name__ == "__main__":
    main()
