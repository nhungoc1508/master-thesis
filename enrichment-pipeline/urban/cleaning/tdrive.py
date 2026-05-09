"""
T-Drive cleaner.

Raw format (no header, coma-separated, one file per taxi):
    taxi_id, timestamps, longitude, latitude

Trip segmentation:
    - Each file = continuous GPS log for one taxi over a full week
    - Segment based on idle gaps: GAP_THRESHOLD_S (default: 1800s = 30min)

Quality filters:
    - Relaxed min points requirement: 10 -> 5
    - 0 coors & coors outside of Beijing are dropped before segmentation

Source: https://www.kaggle.com/datasets/arashnic/tdriver
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterator

from .base import BaseCleaner, QualityConfig

logger = logging.getLogger(__name__)

_CST     = timezone(timedelta(hours=8)) # Beijing is UTC+8
_GAP_S   = 1800
_LAT_MIN = 38.0
_LAT_MAX = 41.5
_LON_MIN = 115.0
_LON_MAX = 118.0

class TDriveCleaner(BaseCleaner):
    source         = 'tdrive'
    city           = 'beijing'
    transport_mode = 'taxi'

    def __init__(self, config: QualityConfig | None = None,
                 gap_threshold_s: int = _GAP_S):
        if config is None:
            config = QualityConfig(min_points=5, min_duration_s=300.0)
        super().__init__(config)
        self._gap_s = gap_threshold_s

    def iter_raw(self, data_path: Path) -> Iterator[dict]:
        data_path = Path(data_path)
        if data_path.is_dir():
            txt_files = sorted(data_path.glob('*.txt'))
            if not txt_files:
                raise FileNotFoundError(f'No .txt files found in {data_path}')
            logger.info('T-Drive: processing %d taxi files', len(txt_files))
            for f in txt_files:
                yield from self._iter_file(f)
        else:
            yield from self._iter_file(data_path)

    def _iter_file(self, file_path: Path) -> Iterator[dict]:
        points: list[tuple[int, float, float]] = [] # (ts_utc, lat, lon)

        with open(file_path) as fh:
            for line in fh:
                parts = line.strip().split(',')
                if len(parts) != 4:
                    continue
                _, ts_str, lon_str, lat_str = parts
                try:
                    lat = float(lat_str)
                    lon = float(lon_str)
                    dt = datetime.strptime(ts_str.strip(), '%Y-%m-%d %H:%M:%S').replace(tzinfo=_CST)
                    ts = int(dt.timestamp())
                except (ValueError, OverflowError):
                    continue

                if not (_LAT_MIN <= lat <= _LAT_MAX and _LON_MIN <= lon <= _LON_MAX):
                    continue

                points.append((ts, lat, lon))

        if not points:
            return
        
        points.sort(key=lambda x: x[0])

        taxi_id = file_path.stem
        seg_idx = 0
        seg_start = 0

        for i in range(1, len(points)):
            if points[i][0] - points[i-1][0] > self._gap_s:
                yield self._make_traj(taxi_id, seg_idx, points[seg_start:i])
                seg_idx += 1
                seg_start = i
        
        yield self._make_traj(taxi_id, seg_idx, points[seg_start:])
    
    @staticmethod
    def _make_traj(taxi_id: str, seg_idx: int, points: list) -> dict:
        return {
            'trajectory_id': f'tdrive_{taxi_id}_{seg_idx}',
            'lats':          [p[1] for p in points],
            'lons':          [p[2] for p in points],
            'timestamps':    [p[0] for p in points],
        }