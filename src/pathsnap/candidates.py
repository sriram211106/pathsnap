"""Step 4 - Candidate generation (used IDENTICALLY by training scripts and by the API).

For each GPS point, list the nearby road arcs that could be the true one.
Uses only GPS positions and the projected network - never labels.

Ties: the two directions of a two-way street are at exactly the same distance.
Ranks use method="min", so both directions always survive a top-K cut together.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from shapely import STRtree, distance as shp_distance, linestrings, points as shp_points


@dataclass(frozen=True)
class CandidateConfig:
    radius_m: float = 30.0   # placeholder - set from scripts/09_candidate_sweep.py results
    max_k: int = 10          # placeholder - set from scripts/09_candidate_sweep.py results

    def to_dict(self):
        return asdict(self)


def build_arc_index(net: pd.DataFrame):
    """Spatial index over all arcs of one trip (geometry in metres)."""
    coords = np.stack([net[["x1", "y1"]].to_numpy(), net[["x2", "y2"]].to_numpy()], axis=1)  # (n, 2, 2)
    segs = linestrings(coords)
    return STRtree(segs), segs


def nearby_arcs(px, py, net: pd.DataFrame, radius_m: float, index=None) -> pd.DataFrame:
    """All arcs within radius_m of each point.

    Returns rows (pt, arc_id, dist_m, rank): pt = row position of the GPS point,
    rank = number of arcs strictly closer to that point (ties share a rank).
    """
    tree, segs = index if index is not None else build_arc_index(net)
    pts = shp_points(np.asarray(px), np.asarray(py))
    pi, aj = tree.query(pts, predicate="dwithin", distance=radius_m)
    df = pd.DataFrame({
        "pt": pi.astype(np.int64),
        "arc_id": net["arc_id"].to_numpy()[aj],
        # rounded to 1 micrometre so A->B and B->A (same street) tie exactly despite float noise
        "dist_m": np.round(shp_distance(pts[pi], segs[aj]), 6),
    })
    df = df.sort_values(["pt", "dist_m", "arc_id"], kind="stable").reset_index(drop=True)
    df["rank"] = (df.groupby("pt")["dist_m"].rank(method="min") - 1).astype(np.int64)
    return df


def generate_candidates(px, py, net: pd.DataFrame, cfg: CandidateConfig, index=None) -> pd.DataFrame:
    """Final candidate set: arcs within cfg.radius_m, at most the cfg.max_k closest (ties kept).

    Points with nothing in range get their single nearest arc(s) as a flagged fallback,
    so every point has at least one candidate (the HMM needs a state for every point).
    """
    index = index if index is not None else build_arc_index(net)
    df = nearby_arcs(px, py, net, cfg.radius_m, index)
    df = df[df["rank"] < cfg.max_k].copy()
    df["is_fallback"] = False

    n = len(px)
    missing = np.setdiff1d(np.arange(n), df["pt"].unique())
    if len(missing):
        tree, segs = index
        pts = shp_points(np.asarray(px)[missing], np.asarray(py)[missing])
        (qi, aj), dist = tree.query_nearest(pts, all_matches=True, return_distance=True)
        fb = pd.DataFrame({
            "pt": missing[qi].astype(np.int64),
            "arc_id": net["arc_id"].to_numpy()[aj],
            "dist_m": dist,
            "rank": 0,
            "is_fallback": True,
        })
        df = pd.concat([df, fb], ignore_index=True)
    return df.sort_values(["pt", "rank", "arc_id"], kind="stable").reset_index(drop=True)
