"""
GeoLife cleaner.

Raw format: one .plt file per recording session per user,
    6 lines of header then CSV format
    lat, lon, 0, altitude_feet, date_serial, date_str, time_str

Trip segmentation:
    - Segmenting on 1800s (30 min) gaps

Transport mode:
    - 69/182 users have a tab-separated labels.txt file with format:
        Start Time, End Time, Transportation Mode
    - Times are UTC
    - A segment is assigned a transport mode if its entire time span falls
        within exactly one label interval
    - Otherwise transport_mode = None
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .base import BaseCleaner, QualityConfig

logger = logging.getLogger(__name__)

_GAP_S = 1800
_LAT_MIN = 38.5
_LAT_MAX = 41.5
_LON_MIN = 114.5
_LON_MAX = 118.5

def _parse_labels(labels_path: Path) -> list[tuple[int, int, str]]:
    """Return list of (start_utc, end_utc, mode) from a labels.txt file"""
    intervals: list[tuple[int, int, str]] = []
    with open(labels_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('Start'):
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            try:
                t0 = int(datetime.strptime(parts[0], '%Y/%m/%d %H:%M:%S')
                         .replace(tzinfo=timezone.utc).timestamp())
                t1 = int(datetime.strptime(parts[1], '%Y/%m/%d %H:%M:%S')
                         .replace(tzinfo=timezone.utc).timestamp())
                intervals.append((t0, t1, parts[2].strip()))
            except ValueError:
                pass
    return intervals

def _assign_mode(seg_start: int, seg_end: int,
                 intervals: list[tuple[int, int, str]]) -> str | None:
    """Return mode string if the segment is fully contained in one interval"""
    for t0, t1, mode in intervals:
        if t0 <= seg_start and seg_end <= t1:
            return mode
    return 'unknown'

class GeoLifeCleaner(BaseCleaner):
    source         = 'geolife'
    city           = 'beijing'

    def __init__(self, config: QualityConfig | None = None,
                 gap_threshold_s: int = _GAP_S):
        super().__init__(config or QualityConfig())
        self._gap_s = gap_threshold_s

    def iter_raw(self, data_path: Path) -> Iterator[dict]:
        data_path = Path(data_path)
        if (data_path / 'Trajectory').is_dir():
            # Single user directory
            yield from self._iter_user(data_path)
        else:
            # Top-level directory containing numbered user subdirs
            user_dirs = sorted(d for d in data_path.iterdir() if d.is_dir())
            logger.info('GeoLife: processing %d user directories', len(user_dirs))
            for user_dir in user_dirs:
                if (user_dir / 'Trajectory').is_dir():
                    yield from self._iter_user(user_dir)

    def _iter_user(self, user_dir: Path) -> Iterator[dict]:
        user_id = user_dir.name
        traj_dir = user_dir / 'Trajectory'
        labels_path = user_dir / 'labels.txt'
        intervals: list[tuple[int, int, str]] = []
        if labels_path.exists():
            intervals = _parse_labels(labels_path)

        for plt_file in sorted(traj_dir.glob('*.plt')):
            yield from self._iter_file(plt_file, user_id, intervals)

    def _iter_file(self, plt_file: Path, user_id: str,
                   intervals: list[tuple[int, int, str]]) -> Iterator[dict]:
        points: list[tuple[int, float, float]] = [] # (ts_utc, lat, lon)

        with open(plt_file) as fh:
            for _ in range(6): # skip 6 header lines
                fh.readline()
            for line in fh:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    ts = int(datetime.strptime(
                        f'{parts[5]} {parts[6]}', '%Y-%m-%d %H:%M:%S'
                    ).replace(tzinfo=timezone.utc).timestamp())
                except (ValueError, OverflowError):
                    continue
                points.append((ts, lat, lon))
        
        if not points:
            return
        
        points.sort(key=lambda x: x[0])

        # Segment on 30-min gaps
        seg_idx = 0
        seg_start = 0
        for i in range(1, len(points)):
            if points[i][0] - points[i-1][0] > self._gap_s:
                yield from self._emit(user_id, plt_file.stem, seg_idx,
                                      points[seg_start:i], intervals)
                seg_idx += 1
                seg_start = i
        yield from self._emit(user_id, plt_file.stem, seg_idx,
                              points[seg_start:], intervals)
        
    def _emit(self, user_id: str, stem: str, seg_idx: int,
              points: list[tuple[int, float, float]],
              intervals: list[tuple[int, int, str]]) -> Iterator[dict]:
        if not points:
            return
        
        lats = [p[1] for p in points]
        lons = [p[2] for p in points]

        centroid_lat = sum(lats) / len(lats)
        centroid_lon = sum(lons) / len(lons)
        if not (_LAT_MIN <= centroid_lat <= _LAT_MAX and _LON_MIN <= centroid_lon <= _LON_MAX):
            return
        
        tss = [p[0] for p in points]
        mode = _assign_mode(tss[0], tss[-1], intervals) if intervals else 'unknown'

        yield {
            'trajectory_id':  f'geolife_{user_id}_{stem}_{seg_idx}',
            'lats':           lats,
            'lons':           lons,
            'timestamps':     tss,
            'transport_mode': mode
        }