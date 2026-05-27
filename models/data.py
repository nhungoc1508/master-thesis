"""
Urban: raw -> canonical -> ENRICHED -> described -> encoded
    Model input = enriched .parquet
Maritime: raw -> ingested -> annotated -> segmented -> enriched -> CANONICAL -> described -> encoded
    Model inout = canonical .parquet

Returned dict keys:
    x: (max_len, input_dim) float32 = [d_lat, d_lon, d_t_n, kin]
    coords: (max_len, 2) float32 = [d_lat_n, d_lon_n]
    tau: (max_len, 4) float32 = [DoW_n, HoD_n, MoH_n, d_t_n]
    pad_mask: (max_len,) bool True = padding
    traj_len: int
    trajectory_id: str
    domain: int, 0 = urban, 1 = maritime
    e_sem: (max_len, sem_dim) float32 or None
    kinematics: (max_len, 3) float32 or None = [speed_n, heading_n, turn_n]
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from resampling import ICR_INTERVAL, normalize_trajectory

logger = logging.getLogger(__name__)

Domain = Literal['urban', 'maritime']

class TrajectoryDataset(Dataset):

    def __init__(self, parquet_paths: list[Path | str], domain: Domain = 'urban', max_len: int = 128,
                 input_dim: int = 3, sem_npy_paths: list[Path | str | None] | None = None,
                 include_kinematics: bool = False):
        """
        Args:
            parquet_paths: canonical .parquet files (one or more)
            domain: 'urban' or 'maritime'
            max_len: maximum sequence length after ICR + DMR
            input_dim: 3 = [d_lat, d_lon, d_t]; 6 = + [speed, heading, turn]
            sem_npy_paths: one .npy path per parquet_path or None to disable semantic embeddings
            include_kinematics: if True, also return 'kinematics' key
        """
        self.domain = domain
        self.max_len = max_len
        self.input_dim = input_dim
        self.include_kinematics = include_kinematics

        if sem_npy_paths is None:
            sem_npy_paths = [None] * len(parquet_paths)
        if len(sem_npy_paths) != len(parquet_paths):
            raise ValueError('sem_npy_paths must have the same length as parquet_paths')

        self.trajectories: list[pd.DataFrame] = []
        self._sem_memmaps: list[np.memmap | None] = []
        
        self._load(parquet_paths, sem_npy_paths)

    def _load(self, paths: list[Path | str], sem_npy_paths: list[Path | str | None]):
        for path, sem_path in zip(paths, sem_npy_paths):
            path = Path(path)
            logger.info('Reading %s', path.name)
            df = pd.read_parquet(path)
            logger.info('\t%d rows, %d unique trajectories', len(df),
                        df['trajectory_id'].nunique() if 'trajectory_id' in df.columns else -1)

            if 'ts_unix' not in df.columns and 'timestamp' in df.columns:
                df = df.rename(columns={'timestamp': 'ts_unix'})

            required = {'trajectory_id', 'lat', 'lon', 'ts_unix'}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f'{path} missing columns: {missing}')

            sem_mm: np.memmap | None = None
            if sem_path is not None:
                sem_path = Path(sem_path)
                if sem_path.exists():
                    sem_mm = np.lib.format.open_memmap(sem_path, mode='r')
                    logger.info('\tsem embeddings: %s  shape=%s', sem_path.name, sem_mm.shape)
                else:
                    logger.warning('\tsem .npy not found: %s', sem_path)

            df = df.reset_index(drop=True)
            df['_abs_row'] = df.index

            if self.domain == 'urban':
                interval_s = ICR_INTERVAL['urban']
                df = df.sort_values(['trajectory_id', 'ts_unix'])
                df['_bucket'] = df['ts_unix'] // interval_s
                n_rows_before = len(df)
                df = (df.groupby(['trajectory_id', '_bucket'], sort=False)
                        .first()
                        .reset_index())
                df = df.drop(columns=['_bucket'], errors='ignore')
                logger.info('\tICR: %d -> %d rows (%.1f%% kept)',
                            n_rows_before, len(df), 100 * len(df) / n_rows_before)

            n_before = len(self.trajectories)
            for _, grp in df.groupby('trajectory_id', sort=False):
                grp = grp.sort_values('ts_unix').reset_index(drop=True)
                if len(grp) < 2:
                    continue
                self.trajectories.append(grp)
                self._sem_memmaps.append(sem_mm)

            n_added = len(self.trajectories) - n_before
            logger.info('\t%d trajectories kept (total so far: %d)',
                        n_added, len(self.trajectories))

    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> dict:
        df = self.trajectories[idx]
        sem_mm = self._sem_memmaps[idx]
        
        lats = df['lat'].values.astype(np.float32)
        lons = df['lon'].values.astype(np.float32)
        ts = df['ts_unix'].values.astype(np.float64)
        abs_rows = df['_abs_row'].values.copy()

        d_lat, d_lon, d_t, _, _ = normalize_trajectory(lats, lons, ts)
        coords = np.stack([d_lat, d_lon], axis=1).astype(np.float32)
        tau = self._build_tau(df, d_t)
        kin = self._get_kinematics_raw(df)

        x = np.stack([d_lat, d_lon, d_t], axis=1).astype(np.float32)
        if self.input_dim == 6 and kin is not None:
            x = np.concatenate([x, kin], axis=1)

        # DMR: subsample if longer than max_len
        n = len(x)
        if n > self.max_len:
            idxs = _dmr_indices(n, self.max_len)
            x = x[idxs]
            coords = coords[idxs]
            tau = tau[idxs]
            abs_rows = abs_rows[idxs]
            if kin is not None:
                kin = kin[idxs]
        
        traj_len = len(x)
        x_pad, pad_mask = _pad(x, self.max_len)
        coords_pad, _ = _pad(coords, self.max_len)
        tau_pad, _ = _pad(tau, self.max_len)

        item = {
            'x': torch.from_numpy(x_pad),
            'coords': torch.from_numpy(coords_pad),
            'tau': torch.from_numpy(tau_pad),
            'pad_mask': torch.from_numpy(pad_mask),
            'traj_len': traj_len,
            'trajectory_id': str(df['trajectory_id'].iloc[0]),
            'domain': 0 if self.domain == 'urban' else 1
        }

        if self.include_kinematics:
            if kin is not None:
                kin_pad, _ = _pad(kin, self.max_len)
            else:
                kin_pad = np.zeros((self.max_len, 3), dtype=np.float32)
            item['kinematics'] = torch.from_numpy(kin_pad)
        else:
            item['kinematics'] = None

        item['e_sem'] = self._load_sem(sem_mm, abs_rows, traj_len)

        return item
    
    @staticmethod
    def _build_tau(df: pd.DataFrame, d_t_norm: np.ndarray) -> np.ndarray:
        ts_int = df['ts_unix'].values.astype(np.int64)
        dt_index = pd.to_datetime(ts_int, unit='s', utc=True)
        moh = (dt_index.minute.values / 59.0).astype(np.float32) # [0, 1]

        if 'day_of_week' in df.columns and 'hour_of_day' in df.columns:
            dow = df['day_of_week'].values.astype(np.float32) / 6.0
            hod = df['hour_of_day'].values.astype(np.float32) / 23.0
        else:
            dow = (dt_index.dayofweek.values / 6.0).astype(np.float32)
            hod = (dt_index.hour.values / 23.0).astype(np.float32)

        return np.stack([dow, hod, moh, d_t_norm.astype(np.float32)], axis=1)
    
    @staticmethod
    def _get_kinematics_raw(df: pd.DataFrame) -> np.ndarray | None:
        if not all(c in df.columns for c in ('SOG', 'COG', 'ROT')):
            return None
        sog = np.nan_to_num(df['SOG'].values.astype(np.float32), nan=0.0)
        cog = np.nan_to_num(df['COG'].values.astype(np.float32), nan=0.0)
        rot = np.nan_to_num(df['ROT'].values.astype(np.float32), nan=0.0)
        speed_n = np.clip(sog / 30.0, 0.0, 1.0)
        heading_n = np.clip(cog / 180.0 - 1.0, -1.0, 1.0)
        turn_n = np.clip(rot / 720.0, -1.0, 1.0)
        return np.stack([speed_n, heading_n, turn_n], axis=1).astype(np.float32)
    
    def _load_sem(self, sem_mm: np.memmap | None, abs_rows: np.ndarray,
                  traj_len: int) -> torch.Tensor | None:
        if sem_mm is None:
            return None
        rows = abs_rows[:traj_len]
        e = sem_mm[rows].astype(np.float32) # (traj_len, sem_dim)
        sem_dim = e.shape[1]
        padded = np.zeros((self.max_len, sem_dim), dtype=np.float32)
        padded[:traj_len] = e
        return torch.from_numpy(padded)
    
# ========== Helpers ==========

def _pad(arr: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad (n, d) array to (max_len, d), return (padded, pad_mask)"""
    n, d = arr.shape
    padded = np.zeros((max_len, d), dtype=arr.dtype)
    padded[:n] = arr
    mask = np.ones(max_len, dtype=bool)
    mask[:n] = False
    return padded, mask

def _dmr_indices(n: int, max_len: int, n_min: int = 10) -> np.ndarray:
    r_min = max_len / n
    log_range = math.log(n - n_min + 1) if n > n_min else 1.0
    rates = np.array([
        1.0 if i <= n_min
        else r_min if i >= n
        else 1.0 - (1.0 - r_min) * math.log(i - n_min + 1) / log_range
        for i in range(n)
    ])
    keep = np.random.rand(n) < rates
    idxs = np.where(keep)[0]
    if len(idxs) < max_len:
        remaining = np.setdiff1d(np.arange(n), idxs)
        extra = np.random.choice(remaining, max_len - len(idxs), replace=False)
        idxs = np.sort(np.concatenate([idxs, extra]))
    elif len(idxs) > max_len:
        idxs = np.sort(np.random.choice(idxs, max_len, replace=False))
    return idxs

def collate_fn(batch: list[dict]) -> dict:
    out = dict()
    for k in batch[0].keys():
        vals = [item[k] for item in batch]
        if vals[0] is None:
            out[k] = None
        elif isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], int):
            out[k] = torch.tensor(vals, dtype=torch.long)
        else:
            out[k] = vals
    return out