"""Distance helpers in metres.

Longitude/latitude are angles, so distances are never measured in raw degrees.
We project each trip onto a flat x/y plane in metres centred on that trip.
"""
import numpy as np

EARTH_RADIUS_M = 6_371_008.8  # mean Earth radius in metres


class LocalProjection:
    """Equirectangular projection around a centre point: lon/lat (degrees) -> x/y (metres).

    One trip covers at most tens of km, where this is accurate to well under 1%.
    The centre is chosen per trip because the trips are spread across the world,
    so no single fixed projection (e.g. one UTM zone) would fit all of them.
    """

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = lon0
        self.lat0 = lat0
        metres_per_degree = EARTH_RADIUS_M * np.pi / 180
        self.kx = metres_per_degree * np.cos(np.radians(lat0))  # 1 degree of longitude shrinks towards the poles
        self.ky = metres_per_degree                             # 1 degree of latitude is ~111 km everywhere

    @classmethod
    def around(cls, lonlat: np.ndarray) -> "LocalProjection":
        """Projection centred on the mean of the given (lon, lat) rows."""
        return cls(float(lonlat[:, 0].mean()), float(lonlat[:, 1].mean()))

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        """(n, 2) lon/lat in degrees -> (n, 2) x/y in metres."""
        x = (lonlat[:, 0] - self.lon0) * self.kx
        y = (lonlat[:, 1] - self.lat0) * self.ky
        return np.column_stack([x, y])


def point_segment_distances(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Distance (metres) from every point to every straight segment start->end.

    points: (n, 2), starts/ends: (m, 2), all in projected metres. Returns an (n, m) matrix.
    """
    seg = ends - starts                                   # (m, 2) segment direction vectors
    seg_len2 = (seg ** 2).sum(axis=1)                     # (m,) squared segment lengths
    seg_len2 = np.where(seg_len2 > 0, seg_len2, 1.0)      # avoid dividing by 0 for zero-length arcs
    rel = points[:, None, :] - starts[None, :, :]         # (n, m, 2) point relative to segment start
    # How far along the segment the closest point is (0 = start, 1 = end), clipped to the segment.
    t = np.clip((rel * seg[None, :, :]).sum(axis=2) / seg_len2, 0.0, 1.0)
    closest = starts[None, :, :] + t[:, :, None] * seg[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=2)
