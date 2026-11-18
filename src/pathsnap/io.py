"""Loaders for the raw PathSnap files.

File formats (headerless, whitespace-separated):
  .track : lon  lat  t            (t = seconds, nominally 1 Hz)
  .nodes : lon  lat               (row index = node_id, 0-based)
  .arcs  : from_node  to_node     (row index = arc_id, 0-based; arcs are DIRECTED)
  .route : arc_id                 (chronological list of arcs the vehicle drove)

NOTE: node_id / arc_id are local to each trip folder. Arc 57 in trip A has
nothing to do with arc 57 in trip B.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PARTITIONS = ("train", "student_test")


def _read_table(path: Path, names: list[str]) -> pd.DataFrame:
    """Read a headerless whitespace file. Non-numeric cells become NaN (so we can count them)."""
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=names, on_bad_lines="warn")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=names, dtype=float)
    for c in names:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def list_trips(partition_dir: Path) -> list[str]:
    """All trip IDs (8-digit folder names) inside a partition folder, sorted."""
    partition_dir = Path(partition_dir)
    return sorted(p.name for p in partition_dir.iterdir() if p.is_dir() and p.name.isdigit())


def trip_files(trip_dir: Path) -> dict[str, Path]:
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    return {ext: trip_dir / f"{tid}.{ext}" for ext in ("track", "nodes", "arcs", "route")}


def load_track(path: Path) -> pd.DataFrame:
    return _read_table(path, ["lon", "lat", "t"])


def load_nodes(path: Path) -> pd.DataFrame:
    df = _read_table(path, ["lon", "lat"])
    df.index.name = "node_id"
    return df


def load_arcs(path: Path) -> pd.DataFrame:
    df = _read_table(path, ["u", "v"])
    df.index.name = "arc_id"
    return df


def load_route(path: Path) -> np.ndarray | None:
    """Returns the ordered arc_id array, or None if the trip has no route file."""
    path = Path(path)
    if not path.exists():
        return None
    df = _read_table(path, ["arc_id"])
    return df["arc_id"].to_numpy()


def load_trip(trip_dir: Path) -> dict:
    """Load all four files of one trip. Missing files come back as None."""
    files = trip_files(trip_dir)
    return {
        "trip_id": Path(trip_dir).name,
        "files": files,
        "track": load_track(files["track"]) if files["track"].exists() else None,
        "nodes": load_nodes(files["nodes"]) if files["nodes"].exists() else None,
        "arcs": load_arcs(files["arcs"]) if files["arcs"].exists() else None,
        "route": load_route(files["route"]),
    }
