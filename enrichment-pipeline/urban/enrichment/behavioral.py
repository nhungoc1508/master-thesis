"""
Kinematic feature enrichment

Columns added:
    speed_ms            float   instantaneous speed in m/s
    acceleration_ms2    float   signed acceleration in m/s^2
    behavioral_phase    str     moving | slow | stopped

Phase thresholds:
    speed < stop_ms     stopped
    speed < slow_ms     slow
    otherwise           moving
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.geo import haversine_vectorized

def enrich(df: pd.DataFrame,
           stop_speed_ms: float = 0.5,
           slow_speed_ms: float = 1.5) -> pd.DataFrame:
    """Add kinematic columns to a trajectory DataFrame"""
    df = df.sort_values(['trajectory_id', 'point_idx']).copy()

    speed_out = np.full(len(df), np.nan, dtype=np.float32)
    accel_out = np.full(len(df), np.nan, dtype=np.float32)
    phase_out = np.empty(len(df), dtype=object)
    phase_out[:] = 'moving'

    grp_bounds = df.groupby('trajectory_id', sort=False).indices

    for tid, idx in grp_bounds.items():
        if len(idx) < 2:
            speed_out[idx] = 0.0
            accel_out[idx] = 0.0
            phase_out[idx] = 'stopped'
            continue

        lats = df['lat'].values[idx]
        lons = df['lon'].values[idx]
        ts = df['timestamp'].values[idx].astype(np.float64)

        dist = haversine_vectorized(lats[:-1], lons[:-1], lats[1:], lons[1:])
        dt = np.diff(ts)
        dt = np.where(dt <= 0, 1e-3, dt)
        spd = (dist / dt).astype(np.float32)

        # Assign speed
        seg_speed = np.empty(len(idx), dtype=np.float32)
        seg_speed[:-1] = spd
        seg_speed[-1] = spd[-1]

        # Assign acceleration
        accel = np.zeros(len(idx), dtype=np.float32)
        if len(spd) > 1:
            dspd = np.diff(spd)
            dt2 = dt[:-1]
            accel[1:-1] = (dspd / dt2).astype(np.float32)
            accel[0] = accel[1]
            accel[-1] = accel[-2]
        
        speed_out[idx] = seg_speed
        accel_out[idx] = accel

        phases = np.full(len(idx), 'moving', dtype=object)
        phases[seg_speed < slow_speed_ms] = 'slow'
        phases[seg_speed < stop_speed_ms] = 'stopped'
        phase_out[idx] = phases

    df['speed_ms'] = speed_out
    df['acceleration_ms2'] = accel_out
    df['behavioral_phase'] = phase_out
    return df