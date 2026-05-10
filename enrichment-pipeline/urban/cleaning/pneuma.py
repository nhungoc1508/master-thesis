"""
pNEUMA dataset cleaner.

Raw format (semicolon-separated, one row per vehicle:
    track_id; type; traveled_d; avg_speed; (lat; lon; speed; lon_acc; lat_acc; time) x N
    - First 4 fields: per-vehicle metadata
    - Remaining fields repeat in groups of 6, one observation per group
    - `time`: elapsed seconds from the recording session start

File naming convention:
    20181024_d1_0830_0900.csv = date: 2018-10-24, drone 1, session 08:30 - 09:00

Points are downsampled to ~1 Hz (every 25th point) to keep timestamps in the interget format

Vehicle type mapping:
    Car, Taxi      -> car / taxi
    Motorcycle     -> motorcycle
    Bus            -> bus
    Medium Vehicle -> medium_vehicle
    Heavy Vehicle  -> heavy_vehicle

Source: https://open-traffic.epfl.ch/
"""
from __future__ import annotations

import datetime
import logging
import re
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

from .base import BaseCleaner, QualityConfig

logger = logging.getLogger(__name__)

# Filename: YYYYMMDD_dN_HHMM_
_FNAME_RE = re.compile(r'(\d{4})(\d{2})(\d{2})_d\w+_(\d{2})(\d{2})_')

_TYPE_MAP = {
    'Car':            'car',
    'Taxi':           'taxi',
    'Motorcycle':     'motorcycle',
    'Bus':            'bus',
    'Medium Vehicle': 'medium_vehicle',
    'Heavy Vehicle':  'heavy_vehicle'
}

_ATHENS_TZ       = ZoneInfo('Europe/Athens')
_DOWNSAMPLE_STEP = 25
_FIELDS_PER_PT   = 6
_META_FIELDS     = 4

class PNEUMACleaner(BaseCleaner):
    source         = 'pneuma'
    city           = 'athens'
    transport_mode = 'car'

    def __init__(self, config : QualityConfig | None = None,
                 default_base_ts: int | None = None,
                 downsample_step: int = _DOWNSAMPLE_STEP):
        super().__init__(config)
        self._default_base_ts = default_base_ts or int(
            datetime.datetime(2018, 10, 24, 8, 30, tzinfo=_ATHENS_TZ).timestamp()
        )
        self._downsample_step = downsample_step

    def iter_raw(self, data_path: Path) -> Iterator[dict]:
        """Accept a single .csv for a directory of .csv files"""
        data_path = Path(data_path)
        if data_path.is_dir():
            csv_files = sorted(data_path.glob('*.csv'))
            if not csv_files:
                raise FileNotFoundError(f'No .sv files found in {data_path}')
            logger.info('pNEUMA: processing %d .csv files from %s', len(csv_files), data_path)
            for f in csv_files:
                yield from self._iter_file(f)
        else:
            yield from self._iter_file(data_path)

    def _iter_file(self, file_path: Path) -> Iterator[dict]:
        base_ts = self._parse_base_ts(file_path)
        step = self._downsample_step
        file_tag = file_path.stem # e.g., '20181024_d1_0830_0900'

        with open(file_path) as fh:
            fh.readline() # skip header
            for line in fh:
                line = line.rstrip('\n')
                if not line:
                    continue
                fields = [f.strip() for f in line.split(';')]
                if len(fields) < _META_FIELDS + _FIELDS_PER_PT:
                    continue
                    
                track_id = fields[0]
                vtype = fields[1].strip()
                gps_fields = fields[_META_FIELDS:]

                lats, lons, timestamps = [], [], []
                n_pts = len(gps_fields) // _FIELDS_PER_PT
                for i in range(0, n_pts, step):
                    off = i * _FIELDS_PER_PT
                    try:
                        lat = float(gps_fields[off])
                        lon = float(gps_fields[off + 1])
                        t_s = float(gps_fields[off + 5])
                    except (ValueError, IndexError):
                        continue
                    lats.append(lat)
                    lons.append(lon)
                    timestamps.append(int(base_ts + round(t_s)))
                
                if not lats:
                    continue

                yield {
                    'trajectory_id':  f'pneuma_{file_tag}_{track_id}',
                    'lats':           lats,
                    'lons':           lons,
                    'timestamps':     timestamps,
                    'transport_mode': _TYPE_MAP.get(vtype, 'car')
                }

    def _parse_base_ts(self, path: Path) -> int:
        m = _FNAME_RE.search(path.stem)
        if not m:
            logger.warning('Cannot parse date/time from filename %s, using default: %s', path.name, self._default_base_ts)
            return self._default_base_ts
        year, month, day, hour, minute = (int(x) for x in m.groups())
        dt = datetime.datetime(year, month, day, hour, minute, tzinfo=_ATHENS_TZ)
        return int(dt.timestamp())