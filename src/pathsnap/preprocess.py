"""Step 2 - Preprocessing (used IDENTICALLY by training scripts and by the API).

Uses ONLY the GPS track and the road network - never the reference route -
because evaluator trips are processed by the same code with no route available
to the model.

Nothing here is fitted to data (no scalers etc.), so it cannot leak information.
Thresholds live in PreprocessConfig and are justified from training-set EDA.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .geo import make_projector, project, utm_epsg


@dataclass(frozen=True)
class PreprocessConfig:
    gap_split_s: float = 30.0          # pause longer than this -> new segment
    max_speed_ms: float = 70.0         # ~250 km/h; faster = physically impossible -> GPS error
    stationary_speed_ms: float = 1.0   # smoothed speed below this -> vehicle considered stopped
    stationary_window: int = 5         # points in the rolling-median speed window
    heading_min_disp_m: float = 2.0    # below this displacement, heading is noise -> NaN

    def to_dict(self):
        return asdict(self)


def bearing_deg(dx, dy):
    """Compass bearing of a displacement: 0 = north (+y), 90 = east (+x)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def trip_projector(track: pd.DataFrame):
    """One UTM projection per trip, chosen from the median GPS position (robust to outliers)."""
    lon_ref, lat_ref = float(track["lon"].median()), float(track["lat"].median())
    return make_projector(lon_ref, lat_ref), utm_epsg(lon_ref, lat_ref)


def preprocess_track(track: pd.DataFrame, proj, cfg: PreprocessConfig = PreprocessConfig()) -> pd.DataFrame:
    """Raw .track rows -> one cleaned row per GPS point (no rows are dropped except NaN rows)."""
    tr = track.dropna(subset=["lon", "lat", "t"]).copy()
    tr["point_idx"] = tr.index.to_numpy()            # original row number in the .track file
    tr = tr.sort_values("t", kind="stable").reset_index(drop=True)

    x, y = project(proj, tr["lon"].to_numpy(), tr["lat"].to_numpy())
    t = tr["t"].to_numpy(dtype=float)
    n = len(tr)
    tr["x"], tr["y"] = x, y

    # time gap, step distance and speed INTO each point
    dt = np.r_[np.nan, np.diff(t)]
    step = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.nan)

    # single-point GPS spike: impossible speed in AND out, but skipping the point looks normal
    vmax = cfg.max_speed_ms
    v_out = np.r_[speed[1:], np.nan]
    skip = np.full(n, np.nan)
    if n >= 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            skip[1:-1] = np.hypot(x[2:] - x[:-2], y[2:] - y[:-2]) / (t[2:] - t[:-2])
    is_jump = (speed > vmax) & (v_out > vmax) & (skip <= vmax)

    # segment boundaries: long pause, or an impossible move that is NOT explained by a spike
    after_jump = np.r_[False, is_jump[:-1]]
    teleport = (speed > vmax) & ~is_jump & ~after_jump
    new_seg = (dt > cfg.gap_split_s) | teleport
    if n:
        new_seg[0] = True
    tr["segment_id"] = np.cumsum(new_seg) - 1
    tr["is_segment_start"] = new_seg

    # values measured across a segment boundary are meaningless -> NaN
    speed = np.where(new_seg, np.nan, speed)
    tr["dt"], tr["step_m"], tr["speed_ms"], tr["is_jump"] = dt, step, speed, is_jump

    # smoothed speed (rolling median within each segment) -> stationary flag
    tr["speed_smooth_ms"] = (
        tr.groupby("segment_id")["speed_ms"]
        .transform(lambda s: s.rolling(cfg.stationary_window, center=True, min_periods=1).median())
    )
    tr["is_stationary"] = tr["speed_smooth_ms"] < cfg.stationary_speed_ms

    # heading: central difference (previous -> next point) within the segment, jump points ignored
    xs = tr["x"].where(~tr["is_jump"])
    ys = tr["y"].where(~tr["is_jump"])
    g = tr["segment_id"]
    x_prev = xs.groupby(g).shift(1).fillna(xs)
    y_prev = ys.groupby(g).shift(1).fillna(ys)
    x_next = xs.groupby(g).shift(-1).fillna(xs)
    y_next = ys.groupby(g).shift(-1).fillna(ys)
    dx, dy = (x_next - x_prev).to_numpy(), (y_next - y_prev).to_numpy()
    disp = np.hypot(dx, dy)
    tr["heading_deg"] = np.where(disp >= cfg.heading_min_disp_m, bearing_deg(dx, dy), np.nan)

    cols = ["point_idx", "t", "lon", "lat", "x", "y", "dt", "step_m", "speed_ms", "speed_smooth_ms",
            "heading_deg", "segment_id", "is_segment_start", "is_jump", "is_stationary"]
    return tr[cols]


def project_network(nodes: pd.DataFrame, arcs: pd.DataFrame, proj) -> pd.DataFrame:
    """One row per directed arc: endpoints in metres, length and bearing (u -> v)."""
    nx, ny = project(proj, nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    u = arcs["u"].to_numpy().astype(np.int64)
    v = arcs["v"].to_numpy().astype(np.int64)
    out = pd.DataFrame({
        "arc_id": np.arange(len(arcs), dtype=np.int64),
        "u": u, "v": v,
        "x1": nx[u], "y1": ny[u], "x2": nx[v], "y2": ny[v],
    })
    out["length_m"] = np.hypot(out["x2"] - out["x1"], out["y2"] - out["y1"])
    out["bearing_deg"] = bearing_deg(out["x2"] - out["x1"], out["y2"] - out["y1"])
    return out
