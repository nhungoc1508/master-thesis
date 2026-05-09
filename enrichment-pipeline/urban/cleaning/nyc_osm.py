"""
NYC cleaner.

Raw format:
    traj_id, user, lat, lon, time

Enriched labels:
    tid, move_id, uid, label

Transport mode:
    - 0 = walk, 1 = bike, 2 = bus, 3 = car, 4 = subway, 5 = train, 6 = taxi
    - enriched_moves contains only move segments
    - For each trajectory, the label with the most GPS points across all
        its move segments is assigned to the whole trajectory

Resampling: too dense -> downsamples to ~1s by keeping
    the first point in each 1-second time bucket

Source: https://zenodo.org/records/15658129
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import numpy as np

from .base import BaseCleaner, QualityConfig

logger = logging.getLogger(__name__)

_LABEL_MAP = {0: 'walk', 1: 'bike', 2: 'bus', 3: 'car',
              4: 'subway', 5: 'train', 6: 'taxi'}
_RAW_FILE = 'input/raw_trajectories_nyc_matbuilder.parquet'
_MOVES_FILE = 'output/enriched_moves.parquet'

def _build_mode_map(moves_path: Path) -> dict[str, str | None]:
    """Return {tid_str: mode_str} using point-weight majority label"""
    if not moves_path.exists():
        logger.warning('enriched_moves.parquet not found at %s; transport_mode will be unknown',
                       moves_path)
        return {}
    
    logger.info('NYC OSM: reading enriched_moves for mode labels')
    moves = pq.read_table(moves_path, columns=['tid', 'label']).to_pandas()
    counts = moves.groupby(['tid', 'label']).size().reset_index(name='n')

    result: dict[str, str | None] = {}
    for tid, grp in counts.groupby('tid'):
        top = grp.loc[grp['n'].idxmax()]
        # Detect tie: more than one label shares the max count
        if (grp['n'] == top['n']).sum() > 1:
            result[str(tid)] = 'unknown'
        else:
            result[str(tid)] = _LABEL_MAP.get(int(top['label']))
    return result

class NYCOSMCleaner(BaseCleaner):
    source         = 'nyc_osm'
    city           = 'new_york'

    def __init__(self, config: QualityConfig | None = None,
                 resample_s: int = 1):
        super().__init__(config or QualityConfig())
        self.resample_s = resample_s

    def iter_raw(self, data_path: Path) -> Iterator[dict]:
        data_path = Path(data_path)
        raw_path = data_path / _RAW_FILE
        moves_path = data_path / _MOVES_FILE

        mode_map = _build_mode_map(moves_path)

        logger.info('NYC OSM: loading raw parquet')
        tbl = pq.read_table(raw_path, columns=['traj_id', 'time', 'lat', 'lon'])
        df = tbl.to_pandas()

        df['ts'] = df['time'].astype('int64') // 1_000_000_000
        df.drop(columns=['time'], inplace=True)
        df.sort_values(['traj_id', 'ts'], inplace=True)

        n_trajs = df['traj_id'].nunique()
        logger.info('NYC OSM: processing %d trajectories', n_trajs)

        for traj_id, grp in df.groupby('traj_id', sort=False):
            ts_arr = grp['ts'].values
            lat_arr = grp['lat'].values
            lon_arr = grp['lon'].values

            ts_d, lat_d, lon_d = self._downsample(ts_arr, lat_arr, lon_arr)

            yield {
                'trajectory_id':  f'nyc_{traj_id}',
                'lats':           lat_d,
                'lons':           lon_d,
                'timestamps':     ts_d,
                'transport_mode': mode_map.get(str(traj_id)),
            }

    def _downsample(self, ts: np.ndarray,
                    lats: np.ndarray, lons: np.ndarray) -> tuple[list, list, list]:
        """Keep the first point in each 1-second bucket"""
        out_ts, out_lat, out_lon = [], [], []
        last_bucket: int | None = None
        for t, la, lo in zip(ts, lats, lons):
            bucket = int(t) // self.resample_s
            if bucket != last_bucket:
                out_ts.append(int(t))
                out_lat.append(float(la))
                out_lon.append(float(lo))
                last_bucket = bucket
        return out_ts, out_lat, out_lon
