"""Step 3 - One reference label per GPS point, by order-preserving alignment to the route.

The .route file lists the arcs driven, in order, but does not say which GPS point
belongs to which arc. We assign each GPS point to one route POSITION such that
positions never go backwards in time, minimising the total point-to-arc distance
(dynamic programming, similar to edit distance / DTW).

Labels are GROUND TRUTH: used for training targets and for scoring only.
They are never used as model features. On demo day the API uses this same
function server-side to score predictions on evaluator trips.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelConfig:
    max_label_dist_m: float = 50.0   # farther than this from its aligned arc -> unlabelled (-1)
    cost_cap_m: float = 50.0         # per-point cost cap so off-road points cannot drag the alignment

    def to_dict(self):
        return asdict(self)


def point_to_segments_dist(px, py, x1, y1, x2, y2):
    """Distance (m) from one point to many segments (vectorised over segments)."""
    dx, dy = x2 - x1, y2 - y1
    len2 = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(len2 > 0, ((px - x1) * dx + (py - y1) * dy) / len2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def align_points_to_route(px, py, net: pd.DataFrame, route, cfg: LabelConfig = LabelConfig()):
    """Monotone alignment of GPS points (in time order) to route positions.

    Returns a DataFrame with, per point:
      route_pos       aligned position in the route list (0..R-1), -1 if no route
      label_arc_id    arc id at that position, -1 if unlabelled
      label_dist_m    distance from the point to that arc
      naive_arc_id    nearest route arc ignoring order (baseline, for comparison only)
      naive_dist_m
    """
    n = len(px)
    out = pd.DataFrame({"route_pos": -1, "label_arc_id": -1, "label_dist_m": np.nan,
                        "naive_arc_id": -1, "naive_dist_m": np.nan}, index=range(n))
    if route is None or n == 0:
        return out
    r = np.asarray(route)
    r = r[~np.isnan(r)].astype(np.int64) if r.dtype.kind == "f" else r.astype(np.int64)
    r = r[(r >= 0) & (r < len(net))]
    R = len(r)
    if R == 0:
        return out

    x1 = net["x1"].to_numpy()[r]
    y1 = net["y1"].to_numpy()[r]
    x2 = net["x2"].to_numpy()[r]
    y2 = net["y2"].to_numpy()[r]

    back = np.zeros((n, R), dtype=np.int16 if R < 32_000 else np.int32)  # best previous position
    naive_pos = np.empty(n, dtype=np.int64)
    naive_d = np.empty(n)
    ar = np.arange(R)
    prev = None
    for i in range(n):
        d = point_to_segments_dist(px[i], py[i], x1, y1, x2, y2)
        naive_pos[i] = int(np.argmin(d))
        naive_d[i] = d[naive_pos[i]]
        c = np.minimum(d, cfg.cost_cap_m)
        if prev is None:
            cur = c
        else:
            # best total cost so far ending at ANY position <= j (positions never go backwards)
            pm = np.minimum.accumulate(prev)
            shifted = np.r_[np.inf, pm[:-1]]
            pidx = np.maximum.accumulate(np.where(prev < shifted, ar, 0))  # earliest argmin of prefix
            cur = c + pm
            back[i] = pidx
        prev = cur

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]

    # distance from each point to ITS aligned arc (element-wise)
    lab_d = point_to_segments_dist(np.asarray(px), np.asarray(py), x1[pos], y1[pos], x2[pos], y2[pos])
    lab = np.where(lab_d <= cfg.max_label_dist_m, r[pos], -1)

    out["route_pos"] = pos
    out["label_arc_id"] = lab
    out["label_dist_m"] = lab_d
    out["naive_arc_id"] = r[naive_pos]
    out["naive_dist_m"] = naive_d
    return out


def angle_diff_deg(a, b):
    """Smallest absolute difference between two compass bearings (0..180)."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
