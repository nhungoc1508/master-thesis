"""
BerlinMOD-Brussels cleaner

Raw format: .parquet file (pre-cleaned in DuckDB) with columns
    tripid, lat, lon, ts

Implementation details:
    - Very densely sampled
    - Resample each trip to ~0.2 Hz using greedy selection: a point is kept
        if it is >= _RESAMPLE_DT seconds after the last kept point
    - Synthetic data so no need to filter GPS noise
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq

from .base import BaseCleaner, QualityConfig

logger = logging.getLogger(__name__)

_RESAMPLE_DT = 5.0 # resample to ~ 0.2 Hz
_FETCH_CHUNK = 500_000

def _resample_trajectory(ts: np.ndarray, lats: np.ndarray, lons: np.ndarray):
    """Vectorized greedy resampling"""
    if len(ts) == 0:
        return ts, lats, lons
    keep = np.zeros(len(ts), dtype=bool)
    keep[0] = True
    last = ts[0]
    for i in range(1, len(ts)):
        if ts[i] - last >= _RESAMPLE_DT:
            keep[i] = True
            last = ts[i]
    return ts[keep], lats[keep], lons[keep]

class BrusselsCleaner(BaseCleaner):
    source         = 'berlinmod_brussels'
    city           = 'brussels'
    transport_mode = 'car'

    def __init__(self, config: QualityConfig | None = None):
        super().__init__(config)
        self.cfg.max_speed_kmh = max(self.cfg.max_speed_kmh, 300.0)

    def iter_raw(self, data_path: Path) -> Iterator[dict]:
        data_path = Path(data_path)
        
        current_tid: int | None = None
        buf_ts: list[float] = []
        buf_lat: list[float] = []
        buf_lon: list[float] = []

        pf = pq.ParquetFile(data_path)
        for batch in pf.iter_batches(batch_size=_FETCH_CHUNK):
            tids = batch.column('tripid').to_pylist()
            lats = batch.column('lat').to_pylist()
            lons = batch.column('lon').to_pylist()
            tss = batch.column('ts').to_pylist()

            for tid, lat, lon, ts in zip(tids, lats, lons, tss):
                tid = int(tid)
                if tid != current_tid:
                    if current_tid is not None:
                        yield self._make_traj(current_tid, buf_ts, buf_lat, buf_lon)
                    current_tid = tid
                    buf_ts, buf_lat, buf_lon = [], [], []
                buf_ts.append(ts)
                buf_lat.append(lat)
                buf_lon.append(lon)

        if current_tid is not None:
            yield self._make_traj(current_tid, buf_ts, buf_lat, buf_lon)

    @staticmethod
    def _make_traj(tripid: int, ts_list: list[float], lat_list: list[float], lon_list: list[float]) -> dict:
        ts_r, lat_r, lon_r = _resample_trajectory(np.array(ts_list), np.array(lat_list), np.array(lon_list))
        return {
            'trajectory_id': f'berlinmod_brussels_{tripid}',
            'lats': lat_r.tolist(),
            'lons': lon_r.tolist(),
            'timestamps': ts_r.astype(np.int64).tolist()
        }