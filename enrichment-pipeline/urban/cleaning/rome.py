"""
Rome Taxi cleaner.

Raw format: no header, semicolon-separated, single file with all taxis interleaved
    taxi_id;timestamp;POINT(lat lon)

Filtering:
    - Apply a point-level outlier pass
    - Any point whose distance to previous kept point implies a speed above
        _MAX_STEP_SPEED_MS (default 55 m/s = 200 km/h) is discarded

Trip segmentation:
    - On idle gaps (default 1800s = 30min)

Source: https://ieee-dataport.org/open-access/crawdad-romataxi
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterator

from utils.geo import haversine_vectorized
from .base import BaseCleaner, QualityConfig

logger = logging.getLogger(__name__)

_CET = timezone(timedelta(hours=1)) # Rome time zone in February = UTC+1
_GAP_S = 1800
_MAX_STEP_SPEED_MS = 55.0

class RomeCleaner(BaseCleaner):
    source         = 'rome'
    city           = 'rome'
    transport_mode = 'taxi'

    def __init__(self, config: QualityConfig | None = None,
                 gap_threshold_s: int = _GAP_S):
        if config is None:
            config = QualityConfig(max_speed_kmh=500.0)
        else:
            config.max_speed_kmh = 500.0
        super().__init__(config)
        self._gap_s = gap_threshold_s

    def iter_raw(self, data_path: Path) -> Iterator[dict]:
        data_path = Path(data_path)
        txt_file = data_path if data_path.is_file() else data_path / 'text_february.txt'
        if not txt_file.exists():
            raise FileNotFoundError(f'.txt file not found in {data_path}')
        logger.info('Rome: reading %s in a single pass', txt_file.name)

        # ---------- Single pass: collect points by taxi_id ----------
        taxi_points: dict[int, list[tuple[int, float, float]]] = defaultdict(list)

        with open(txt_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(';')
                if len(parts) != 3:
                    continue
                try:
                    tid = int(parts[0])
                    ts_str = parts[1][:19] # 'YYYY-MM-DD HH:MM:SS'
                    ts = int(datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                             .replace(tzinfo=_CET).timestamp())
                    coord_str = parts[2][6:-1] # 'POINT(...)'
                    lat_s, lon_s = coord_str.split()
                    lat, lon = float(lat_s), float(lon_s)
                except (ValueError, IndexError):
                    continue
                taxi_points[tid].append((ts, lat, lon))

        logger.info('Rome: %d taxis loaded, segmenting next', len(taxi_points))

        # ---------- Segment each taxi on gaps, with per-point noise removal ----------
        for taxi_id in sorted(taxi_points):
            points = sorted(taxi_points[taxi_id], key=lambda x: x[0])
            seg_idx = 0
            seg_start = 0
            for i in range(1, len(points)):
                if points[i][0] - points[i-1][0] > self._gap_s:
                    yield self._make_traj(taxi_id, seg_idx,
                                          self._denoise(points[seg_start:i]))
                    seg_idx += 1
                    seg_start = i
            yield self._make_traj(taxi_id, seg_idx,
                                  self._denoise(points[seg_start:]))
            
    @staticmethod
    def _denoise(
        points: list[tuple[int, float, float]]
    ) -> list[tuple[int, float, float]]:
        """
        Drop GPS points whose step speed exceeds the threshold or
        whose timestamp duplicates the previous kept point (same second,
        caused by sub-second truncation)
        """
        if len(points) < 2:
            return points
        clean = [points[0]]
        for ts, lat, lon in points[1:]:
            prev_ts, prev_lat, prev_lon = clean[-1]
            dt = ts - prev_ts
            if dt == 0: # same-second duplicate
                continue
            dist = haversine_vectorized(prev_lat, prev_lon, lat, lon)
            if dist / dt <= _MAX_STEP_SPEED_MS:
                clean.append((ts, lat, lon))
        return clean

    @staticmethod
    def _make_traj(taxi_id: int, seg_idx: int,
                   points: list[tuple[int, float, float]]) -> dict:
        return {
            'trajectory_id': f'rome_{taxi_id}_{seg_idx}',
            'lats':          [p[1] for p in points],
            'lons':          [p[2] for p in points],
            'timestamps':    [p[0] for p in points]
        }