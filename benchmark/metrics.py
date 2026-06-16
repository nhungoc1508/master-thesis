"""
Evaluation metrics for frozen benchmark.

Process:
    - Obtain predictions as normalized d_lat/d_lon
    - Denormalize predictions to absolute lat/lon using per-trajectory denom params
    - Compute error as Haversine distance (meters) calculated at masked positions only
"""
from __future__ import annotations

import numpy as np

_EARTH_R = 6_371_000.0 # radius of Earth (in meters)

def denorm_coords(coords_n: np.ndarray, bbox_half: float,
                 lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Map normalized (d_lat_norm, d_lon_norm) back to absolute (lat, lon)

    Args:
        coords_n: (..., 2) array of normalized offsets
    Returns:
        (lat, lon): arrays of shape coords_n[..., 0]
    """
    lat = coords_n[..., 0] * bbox_half + lat0
    lon = coords_n[..., 1] * bbox_half + lon0
    return lat, lon

def haversine_m(lat1: np.ndarray, lon1: np.ndarray,
                lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * _EARTH_R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def recovery_error_m(pred_coords_n: np.ndarray, true_coords_n: np.ndarray,
                     mask: np.ndarray, denorm: dict) -> np.ndarray:
    """
    Per-position Haversine error (in meters) at masked positions for one trajectory

    Args:
        pred_coords_n / true_coords_n: (L, 2) normalized predicted / ground-truth offsets
        mask: (L,) bool, True = masked position
        denorm: dict with bbox_half, lat0, lon0
    Returns:
        (n_masked,) array of meter errors
    """
    plat, plon = denorm_coords(pred_coords_n, denorm['bbox_half'], denorm['lat0'], denorm['lon0'])
    tlat, tlon = denorm_coords(true_coords_n, denorm['bbox_half'], denorm['lat0'], denorm['lon0'])
    err = haversine_m(tlat, tlon, plat, plon)
    return err[mask]

def aggregate(errors_m: list[np.ndarray]) -> dict:
    """Aggregate per-trajectory error arrays into summary metrics"""
    if not errors_m:
        return {'n': 0}
    full_err = np.concatenate([e for e in errors_m if e.size > 0])
    return {
        'n_points': int(full_err.size),
        'n_trajectories': len(errors_m),
        'mae_m': float(np.mean(full_err)),
        'rmse_m': float(np.sqrt(np.mean(full_err ** 2))),
        'median_m': float(np.median(full_err)),
        'p90_m': float(np.percentile(full_err, 90))
    }