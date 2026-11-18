"""Step 5 - Features for every (GPS point, candidate arc) pair.

Used IDENTICALLY by training scripts and by the API. Uses only GPS points,
the projected network and the candidate lists - NEVER labels.
Context features never reach across a segment boundary (long pause).
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .labels import angle_diff_deg
from .preprocess import bearing_deg

FEATURES = [
    # 1. geometry
    "dist_m", "rank", "dist_minus_min", "dist_ratio", "along_offset_m", "proj_frac",
    "arc_length_m", "has_reverse",
    # 2. direction
    "heading_diff", "cos_heading_diff", "heading_missing", "speed_smooth_ms", "heading_agree_weighted",
    # 3. context
    "conn_prev", "conn_next", "persist", "win_mean_dist",
    # 4. point quality
    "progress_ratio", "n_candidates", "min_dist", "dt", "is_segment_start", "is_fallback",
]


@dataclass(frozen=True)
class FeatureConfig:
    heading_span_m: float = 10.0     # travel distance used on EACH side of a point for heading
    heading_min_disp_m: float = 5.0  # below this displacement, heading is unknown
    progress_window: int = 10        # points on each side for the progress ratio (~10 s)
    persist_window: int = 5          # points on each side for persistence / window distance

    def to_dict(self):
        return asdict(self)


def _segment_bounds(seg: np.ndarray):
    n = len(seg)
    idx = np.arange(n)
    start_mask = np.r_[True, seg[1:] != seg[:-1]]
    end_mask = np.r_[seg[1:] != seg[:-1], True]
    seg_start = np.maximum.accumulate(np.where(start_mask, idx, 0))
    seg_end = np.minimum.accumulate(np.where(end_mask, idx, n - 1)[::-1])[::-1]
    return seg_start, seg_end, start_mask


def point_context(pts: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """Per-point heading (distance-based) and progress ratio, computed within segments."""
    x, y = pts["x"].to_numpy(), pts["y"].to_numpy()
    seg = pts["segment_id"].to_numpy()
    n = len(pts)
    seg_start, seg_end, start_mask = _segment_bounds(seg)

    step = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
    step[start_mask] = 0.0                       # no travel across a segment boundary
    s = np.cumsum(step)                          # cumulative path length (non-decreasing)

    # heading: from the point ~span metres back to the point ~span metres ahead (same segment)
    jb = np.searchsorted(s, s - cfg.heading_span_m, side="right") - 1
    jf = np.searchsorted(s, s + cfg.heading_span_m, side="left")
    jb = np.clip(jb, seg_start, seg_end)
    jf = np.clip(jf, seg_start, seg_end)
    dx, dy = x[jf] - x[jb], y[jf] - y[jb]
    disp = np.hypot(dx, dy)
    heading = np.where(disp >= cfg.heading_min_disp_m, bearing_deg(dx, dy), np.nan)

    # progress ratio: straight-line distance / path length over +-W points (same segment)
    idx = np.arange(n)
    lo = np.maximum(idx - cfg.progress_window, seg_start)
    hi = np.minimum(idx + cfg.progress_window, seg_end)
    net = np.hypot(x[hi] - x[lo], y[hi] - y[lo])
    path = s[hi] - s[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        prog = np.where(path > 1.0, net / path, 1.0)  # no movement info -> neutral 1.0
    return pd.DataFrame({"heading_dist_deg": heading, "progress_ratio": np.clip(prog, 0, 1)})


def build_features(pts: pd.DataFrame, net: pd.DataFrame, cand: pd.DataFrame,
                   cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """One row per (point, candidate) with all FEATURES. pts rows must be in time order (pt = row)."""
    ctx = point_context(pts, cfg)
    n = len(pts)
    c = cand[["pt", "arc_id", "dist_m", "rank", "is_fallback"]].copy().reset_index(drop=True)
    pt = c["pt"].to_numpy()
    a = c["arc_id"].to_numpy()

    au, av = net["u"].to_numpy(), net["v"].to_numpy()
    u, v = au[a], av[a]
    x1, y1 = net["x1"].to_numpy()[a], net["y1"].to_numpy()[a]
    x2, y2 = net["x2"].to_numpy()[a], net["y2"].to_numpy()[a]
    length = net["length_m"].to_numpy()[a]
    px, py = pts["x"].to_numpy()[pt], pts["y"].to_numpy()[pt]

    # --- 1. geometry
    dx, dy = x2 - x1, y2 - y1
    len2 = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t_raw = np.where(len2 > 0, ((px - x1) * dx + (py - y1) * dy) / len2, 0.5)
    c["proj_frac"] = np.clip(t_raw, 0, 1)
    c["along_offset_m"] = (np.maximum(0, -t_raw) + np.maximum(0, t_raw - 1)) * length
    c["arc_length_m"] = length
    g = c.groupby("pt")["dist_m"]
    c["min_dist"] = g.transform("min")
    c["dist_minus_min"] = c["dist_m"] - c["min_dist"]
    c["dist_ratio"] = c["dist_m"] / (c["min_dist"] + 1.0)
    c["n_candidates"] = g.transform("size")
    n_nodes = int(max(au.max(), av.max())) + 1
    arc_keys = au.astype(np.int64) * n_nodes + av
    c["has_reverse"] = np.isin(v.astype(np.int64) * n_nodes + u, arc_keys)

    # --- 2. direction
    diff = angle_diff_deg(ctx["heading_dist_deg"].to_numpy()[pt], net["bearing_deg"].to_numpy()[a])
    c["heading_missing"] = np.isnan(diff)
    c["heading_diff"] = np.where(np.isnan(diff), 90.0, diff)        # unknown -> neutral 90 deg
    c["cos_heading_diff"] = np.cos(np.radians(c["heading_diff"]))
    sp = np.nan_to_num(pts["speed_smooth_ms"].to_numpy()[pt], nan=0.0)
    c["speed_smooth_ms"] = sp
    c["heading_agree_weighted"] = c["cos_heading_diff"] * np.minimum(sp, 15.0) / 15.0

    # --- 3. context (within the same segment only)
    seg = pts["segment_id"].to_numpy()
    M = len(net)
    cand_key = pt.astype(np.int64) * M + a
    key_v = pt.astype(np.int64) * n_nodes + v     # "candidate of point p ends at node v"
    key_u = pt.astype(np.int64) * n_nodes + u     # "candidate of point p starts at node u"

    def neighbour(o):
        q = pt + o
        ok = (q >= 0) & (q < n)
        ok[ok] &= seg[q[ok]] == seg[pt[ok]]
        return q.astype(np.int64), ok

    q, ok = neighbour(-1)
    same_prev = ok & np.isin(q * M + a, cand_key)
    follows_prev = ok & np.isin(q * n_nodes + u, key_v)   # some prev candidate ends where this starts
    q, ok = neighbour(+1)
    same_next = ok & np.isin(q * M + a, cand_key)
    leads_next = ok & np.isin(q * n_nodes + v, key_u)     # some next candidate starts where this ends
    c["conn_prev"] = same_prev | follows_prev
    c["conn_next"] = same_next | leads_next

    dist_lookup = pd.Series(c["dist_m"].to_numpy(), index=cand_key)
    present_cnt = np.zeros(len(c))
    valid_cnt = np.zeros(len(c))
    dsum = c["dist_m"].to_numpy().copy()
    dcnt = np.ones(len(c))
    W = cfg.persist_window
    for o in [k for k in range(-W, W + 1) if k != 0]:
        q, ok = neighbour(o)
        k = q * M + a
        present = ok & np.isin(k, cand_key)
        present_cnt += present
        valid_cnt += ok
        if present.any():
            dsum[present] += dist_lookup.reindex(k[present]).to_numpy()
            dcnt[present] += 1
    c["persist"] = np.where(valid_cnt > 0, present_cnt / np.maximum(valid_cnt, 1), 0.0)
    c["win_mean_dist"] = dsum / dcnt

    # --- 4. point quality
    c["progress_ratio"] = ctx["progress_ratio"].to_numpy()[pt]
    c["dt"] = np.nan_to_num(pts["dt"].to_numpy()[pt], nan=0.0)
    c["is_segment_start"] = pts["is_segment_start"].to_numpy()[pt]

    for col in ["has_reverse", "heading_missing", "conn_prev", "conn_next", "is_segment_start", "is_fallback"]:
        c[col] = c[col].astype(np.int8)
    return c[["pt", "arc_id"] + FEATURES]
