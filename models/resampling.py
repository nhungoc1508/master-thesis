"""
Interval Consistent Sampling (ICR) & Dynamic Multi-scale Resampling (DMR)

- ICR: resamples a trajectory to a canonical time interval by keeping the first point in each time bucket
- DMR: logarithmically subsamples sequences that still exceed max_len
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

Domain = Literal['urban', 'maritime']

ICR_INTERVAL = {'urban': 30}
MAX_LEN = {'urban': 128, 'maritime': 256}

def icr(df: pd.DataFrame, domain: Domain, ts_col: str = 'ts_unix') -> pd.DataFrame:
    """Resample to canonical interval, keep first point in each d_t_can bucket"""
    interval_s = ICR_INTERVAL[domain]
    df = df.copy().sort_values(ts_col)
    df['_bucket'] = df[ts_col] // interval_s
    df = df.groupby('_bucket', sort=True).first().reset_index(drop=True)
    df = df.drop(columns=['_bucket'], errors='ignore')
    return df

def _dmr_keep_indices(n: int, max_len: int, n_min: int = 10) -> np.ndarray:
    """Return sorted indices for logarithmic subsampling of a sequence of length n"""
    if n <= max_len:
        return np.arange(n)
    r_min = max_len / n
    n_max = n
    log_range = math.log(n_max - n_min + 1)
    rates = []
    for i in range(n):
        if i <= n_min:
            r = 1.0
        elif i >= n_max:
            r = r_min
        else:
            r = 1.0 - (1.0 - r_min) * math.log(i - n_min + 1) / log_range
        rates.append(r)
    rates = np.array(rates)
    keep = np.random.rand(n) < rates
    idxs = np.where(keep)[0]
    if len(idxs) < max_len:
        remaining = np.setdiff1d(np.arange(n), idxs)
        extra = np.random.choice(remaining, max_len - len(idxs), replace=False)
        idxs = np.sort(np.concatenate([idxs, extra]))
    elif len(idxs) > max_len:
        idxs = np.sort(np.random.choice(idxs, max_len, replace=False))
    return idxs

def dmr(points: np.ndarray, max_len: int, n_min: int = 10) -> np.ndarray:
    n = len(points)
    if n <= max_len:
        return points
    idxs = _dmr_keep_indices(n, max_len, n_min)
    return points[idxs]

def normalize_trajectory(lats: np.ndarray, lons: np.ndarray, ts: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Convert absolute lat/lon/ts to relative offsets normalized to [-1, 1] / log-scale
    Returns (d_lat, d_lon, d_t_norm, bbox_half, log_max_dt)
        d_lat & d_lon are divided by bbox_half
        d_t_n = log1p(d_t_s) / log1p(max_d_t_s)
    """
    d_lat = lats - lats[0]
    d_lon = lons - lons[0]
    d_t = (ts - ts[0]).astype(float)

    bbox_half = max(
        max(abs(d_lat.max()), abs(d_lat.min()), 1e-8),
        max(abs(d_lon.max()), abs(d_lon.min()), 1e-8)
    )
    d_lat_n = d_lat / bbox_half
    d_lon_n = d_lon / bbox_half

    log_max_dt = math.log1p(d_t.max()) if d_t.max() > 0 else 1.0
    d_t_n = np.log1p(d_t) / log_max_dt

    return d_lat_n, d_lon_n, d_t_n, float(bbox_half), float(log_max_dt)