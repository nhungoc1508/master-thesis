"""
Cabspotting cleaner.

Raw format: no header, space-separated, one file per cab
    lat lon occupancy timestamp

Trip segmentation:
    - Files store GPS points in reverse chronological order
    - One file = continuois 24-day GPS log for one cab
    - Segment trips by GAP_THRESHOLD_S (default 1800s = 30min)
    - Points outside of the SF Bay Area bounding box are dropped before segmenting

Source: https://ieee-dataport.org/open-access/crawdad-epflmobility
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from .base import BaseCleaner, QualityConfig

logger = logging.getLogger(__name__)

_GAP_S = 1800
_LAT_MIN = 36.5
_LAT_MAX = 38.5
_LON_MIN = -123.5
_LON_MAX = -121.5

class CabspottingCleaner(BaseCleaner):
    source         = 'cabspotting'
    city           = 'san_francisco'
    transport_mode = 'taxi'

    def __init__(self, config: QualityConfig | None = None,
                 gap_threshold_s: int = _GAP_S):
        super().__init__(config or QualityConfig())
        self._gap_s = gap_threshold_s

    def iter_raw(self, data_path: Path) -> Iterator[dict]:
        data_path = Path(data_path)
        if data_path.is_dir():
            txt_files = sorted(data_path.glob('*.txt'))
            if not txt_files:
                raise FileNotFoundError(f'No .txt files found in {data_path}')
            logger.info('Cabspotting: processing %d cab files', len(txt_files))
            for f in txt_files:
                yield from self._iter_file(f)
        else:
            yield from self._iter_file(data_path)

    def _iter_file(self, file_path: Path) -> Iterator[dict]:
        points: list[tuple[int, float, float]] = [] # (ts_utc, lat, lon)

        with open(file_path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    ts = int(parts[3])
                except (ValueError, IndexError):
                    continue

                if not (_LAT_MIN <= lat <= _LAT_MAX and _LON_MIN <= lon <= _LON_MAX):
                    continue

                points.append((ts, lat ,lon))

        if not points:
            return
        
        points.sort(key=lambda x: x[0])

        cab_id = file_path.stem
        seg_idx = 0
        seg_start = 0
        
        for i in range(1, len(points)):
            if points[i][0] - points[i-1][0] > self._gap_s:
                yield self._make_traj(cab_id, seg_idx, points[seg_start:i])
                seg_idx += 1
                seg_start = i

        yield self._make_traj(cab_id, seg_idx, points[seg_start:])

    @staticmethod
    def _make_traj(cab_id: str, seg_idx: int, points: list) -> dict:
        return {
            'trajectory_id': f'cabspotting_{cab_id}_{seg_idx}',
            'lats':          [p[1] for p in points],
            'lons':          [p[2] for p in points],
            'timestamps':    [p[0] for p in points],
        }