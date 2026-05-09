import numpy as np

def haversine_vectorized(lat1: np.ndarray, lon1: np.ndarray,
                         lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Return distances (in meters) between arrays of (lat1, lon1) and (lat2, lon2)"""
    R = 6_371_000.0 # radius of Earth (in meters)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def bbox_diagonal_m(lats: np.ndarray, lons: np.ndarray) -> float:
    """Diagonal of the bounding box of a set of coordinates, in meters"""
    return haversine_vectorized(
        np.array([lats.min()]), np.array([lons.min()]),
        np.array([lats.max()]), np.array([lons.max()])
    )[0]