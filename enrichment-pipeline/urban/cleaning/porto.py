"""
Porto dataset cleaner.

Raw format (train.csv):
    TRIP_ID, CALL_TYPE, ORIGIN_CALL, ORIGIN_STAND, TAXI_ID,
    TIMESTAMP (unix s, trip start), DAY_TYPE, MISSING_DATA,
    POLYLINE (JSON [[lon, lat], ...], 15-second intervals)
    Coordinates in WGS-84
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

from .base import BaseCleaner

logger = logging.getLogger(__name__)

SAMPLING_S = 15 # fixed interval for Porto

class PortoCleaner(BaseCleaner):
    source         = 'porto'
    city           = 'porto'
    transport_mode = 'taxi'

    def iter_raw(self, data_path: Path) -> Iterator[dict]:
        reader = pd.read_csv(
            data_path,
            low_memory=False,
            chunksize=50_000,
        )
        for chunk in reader:
            chunk = chunk[chunk['MISSING_DATA'] == False].reset_index(drop=True)
            for _, row in chunk.iterrows():
                try:
                    polyline = json.loads(row['POLYLINE'])
                except (json.JSONDecodeError, TypeError):
                    continue

                if not polyline or len(polyline) < 2:
                    continue

                start_ts = int(row['TIMESTAMP'])
                lons = [float(p[0]) for p in polyline]
                lats = [float(p[1]) for p in polyline]
                timestamps = [start_ts + i * SAMPLING_S for i in range(len(polyline))]

                yield {
                    'trajectory_id': f'porto_{row["TRIP_ID"]}',
                    'lats': lats,
                    'lons': lons,
                    'timestamps': timestamps
                }