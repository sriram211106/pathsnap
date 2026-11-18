"""Geographic helpers.

Why: lon/lat are angles. 0.001 deg of longitude is ~64 m in Germany (lat 50)
but ~111 m at the equator, so Euclidean distance in degrees is wrong.
We project each trip into its local UTM zone (units = metres) and measure there.
Trips are in different countries, so the zone is chosen PER TRIP.
"""
import numpy as np
from pyproj import Transformer

EARTH_R = 6_371_000.0  # metres


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat). 326xx = north, 327xx = south."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def make_projector(lon_ref: float, lat_ref: float) -> Transformer:
    """Transformer lon/lat (WGS84) -> x/y metres. always_xy=True means input order is (lon, lat)."""
    return Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg(lon_ref, lat_ref)}", always_xy=True)


def project(transformer: Transformer, lon, lat):
    x, y = transformer.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x), np.asarray(y)


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (vectorised). Used as a projection-free cross-check."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))
