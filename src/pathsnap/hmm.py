"""Step 9 - HMM + Viterbi route reconstruction (used IDENTICALLY by tuning scripts and the API).

Hidden states : the candidate arcs of each GPS point.
Emission      : the classifier's probability, normalised over the point's candidates,
                raised to the power alpha (emission weight)  -> alpha * log p.
Transition    : route distance along the directed road network between the two candidates
                vs straight-line distance between the two GPS points (Newson & Krumm 2009):
                log T = -|d_route - d_gps| / beta.
Disconnected  : no route within the search cutoff -> worse than any connected option
                (not impossible, so the chain never breaks).
Gaps          : segments (Step 2 splits at pauses > 30 s) are decoded independently.
"""
import heapq
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HMMConfig:
    beta_m: float = 10.0            # transition strictness (metres)
    emission_weight: float = 1.0    # alpha: trust in the classifier
    backtrack_tol_m: float = 10.0   # allowed backwards movement along the same arc (GPS noise)
    cutoff_factor: float = 3.0      # route search radius = factor * straight-line distance ...
    min_cutoff_m: float = 200.0     # ... but at least this
    max_cutoff_m: float = 3000.0    # ... and at most this
    disconnected_margin: float = 5.0  # disconnected = worst connected log-prob minus this

    def to_dict(self):
        return asdict(self)


class RoadGraph:
    """Directed road graph with cached, distance-limited Dijkstra."""

    def __init__(self, net: pd.DataFrame):
        u = net["u"].to_numpy()
        order = np.argsort(u, kind="stable")
        n_nodes = int(max(u.max(), net["v"].max())) + 1
        self.indptr = np.searchsorted(u[order], np.arange(n_nodes + 1))
        self.nbr = net["v"].to_numpy()[order]
        self.w = net["length_m"].to_numpy()[order]
        self.cache = {}

    def dists_from(self, src: int, cutoff: float) -> dict:
        hit = self.cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, x = heapq.heappop(heap)
            if d > dist.get(x, np.inf):
                continue
            for k in range(self.indptr[x], self.indptr[x + 1]):
                y, nd = self.nbr[k], d + self.w[k]
                if nd <= cutoff and nd < dist.get(y, np.inf):
                    dist[y] = nd
                    heapq.heappush(heap, (nd, y))
        self.cache[src] = (cutoff, dist)
        return dist


def prepare_trip(pts: pd.DataFrame, net: pd.DataFrame, cand: pd.DataFrame,
                 cfg: HMMConfig = HMMConfig(), graph: RoadGraph = None) -> dict:
    """Everything Viterbi needs, independent of alpha/beta (so tuning can reuse it).

    pts  : processed points in time order (row position = pt), needs x, y, segment_id
    cand : rows (pt, arc_id, score) for every point (every point must have >= 1 candidate)
    """
    graph = graph or RoadGraph(net)
    n = len(pts)
    c = cand[["pt", "arc_id", "score"]].sort_values(["pt", "score"], ascending=[True, False], kind="stable")
    ptc = c["pt"].to_numpy()
    if len(np.unique(ptc)) != n:
        raise ValueError("every point needs at least one candidate")
    a_all = c["arc_id"].to_numpy()
    s_all = np.clip(c["score"].to_numpy().astype(float), 1e-9, None)

    x, y = pts["x"].to_numpy(), pts["y"].to_numpy()
    U, V = net["u"].to_numpy(), net["v"].to_numpy()
    L = net["length_m"].to_numpy()
    x1, y1, x2, y2 = (net[k].to_numpy() for k in ("x1", "y1", "x2", "y2"))
    dx, dy = x2[a_all] - x1[a_all], y2[a_all] - y1[a_all]
    len2 = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t_all = np.where(len2 > 0, ((x[ptc] - x1[a_all]) * dx + (y[ptc] - y1[a_all]) * dy) / len2, 0.5)
    t_all = np.clip(t_all, 0, 1)

    bounds = np.r_[0, np.flatnonzero(np.diff(ptc)) + 1, len(ptc)]
    arcs, ts, logp = [], [], []
    for k in range(n):
        sl = slice(bounds[k], bounds[k + 1])
        s = s_all[sl]
        arcs.append(a_all[sl])
        ts.append(t_all[sl])
        logp.append(np.log(s / s.sum()))

    seg = pts["segment_id"].to_numpy()
    deltas, cutoffs = [None] * n, np.zeros(n)
    for i in range(1, n):
        if seg[i] != seg[i - 1]:
            continue
        d_gps = float(np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]))
        cutoff = float(np.clip(cfg.cutoff_factor * d_gps, cfg.min_cutoff_m, cfg.max_cutoff_m))
        A, TA, B, TB = arcs[i - 1], ts[i - 1], arcs[i], ts[i]
        D = np.full((len(A), len(B)), np.inf, dtype=np.float32)
        for ia, a in enumerate(A):
            rest = (1.0 - TA[ia]) * L[a]
            dist = graph.dists_from(int(V[a]), cutoff)
            for ib, b in enumerate(B):
                if b == a:
                    along = (TB[ib] - TA[ia]) * L[a]
                    route = max(along, 0.0) if along >= -cfg.backtrack_tol_m else np.inf
                else:
                    sp = dist.get(int(U[b]))
                    route = rest + sp + TB[ib] * L[b] if sp is not None else np.inf
                if np.isfinite(route):
                    D[ia, ib] = abs(route - d_gps)
        deltas[i], cutoffs[i] = D, cutoff
    return {"n": n, "arcs": arcs, "logp": logp, "deltas": deltas, "cutoffs": cutoffs, "segment_id": seg}


def decode(prep: dict, beta_m: float, emission_weight: float, disconnected_margin: float = 5.0):
    """Viterbi over each segment. Returns (arc per point, used_disconnected flag per point)."""
    n = prep["n"]
    out = np.empty(n, dtype=np.int64)
    disc = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        j = i + 1
        while j < n and prep["deltas"][j] is not None:
            j += 1
        # segment = points i .. j-1
        score = emission_weight * prep["logp"][i]
        back = []
        for k in range(i + 1, j):
            D = prep["deltas"][k]
            T = -D / beta_m
            worst = -(prep["cutoffs"][k] / beta_m) - disconnected_margin
            T = np.where(np.isfinite(T), T, worst)
            S = score[:, None] + T
            arg = S.argmax(axis=0)
            score = S[arg, np.arange(S.shape[1])] + emission_weight * prep["logp"][k]
            back.append(arg)
        best = int(score.argmax())
        idx = [best]
        for k in range(j - 1, i, -1):
            prev = int(back[k - i - 1][idx[-1]])
            if not np.isfinite(prep["deltas"][k][prev, idx[-1]]):
                disc[k] = True
            idx.append(prev)
        idx.reverse()
        for k, c in zip(range(i, j), idx):
            out[k] = prep["arcs"][k][c]
        i = j
    return out, disc


def classifier_only(prep: dict) -> np.ndarray:
    """Top-1 per point without any sequence information (candidates are sorted by score)."""
    return np.array([a[0] for a in prep["arcs"]], dtype=np.int64)


def route_continuity(arcs: np.ndarray, segment_id: np.ndarray, net: pd.DataFrame) -> tuple[int, int]:
    """(connected pairs, total pairs) for consecutive points within a segment."""
    U, V = net["u"].to_numpy(), net["v"].to_numpy()
    same = segment_id[1:] == segment_id[:-1]
    a0, a1 = arcs[:-1][same], arcs[1:][same]
    ok = (a0 == a1) | (V[a0] == U[a1])
    return int(ok.sum()), int(len(ok))
