"""
Loader for frozen benchmark units produced by freeze_benchmark.py.

Each source dataset is its own unit dir. This loader composes ANY set of units
into one dataset, so you can run experiments on a single city, all-urban,
urban→maritime transfer, etc., from the same frozen artifacts. Every model
(this one + baselines) loads through this, guaranteeing identical inputs/targets.

Per item (see freeze_benchmark for field meanings):
    x_spatial, coords : (L, 2)   tau : (L, 4)   kinematics : (L, 3)
    pad_mask, pos_mask : (L,)    domain_id : int   traj_len : int
    e_sem : (L, sem_dim) or None
    target_coords : (L, 2)       denorm : dict   dataset : str
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def find_units(split_dir: str | Path, domains: list[str] | None = None,
               datasets: list[str] | None = None) -> list[Path]:
    """
    Discover frozen unit dirs under a split, optionally filtered.

    split_dir: e.g. benchmark/frozen/test
    domains:   keep only these domains (e.g. ['urban']); None = all.
    datasets:  keep only these dataset names (e.g. ['porto_enriched']); None = all.
    Returns sorted list of unit directories (each contains corpus.npz).
    """
    split_dir = Path(split_dir)
    units = []
    for corpus in sorted(split_dir.glob('*/*/corpus.npz')):
        unit = corpus.parent
        dom = unit.parent.name
        name = unit.name
        if domains and dom not in domains:
            continue
        if datasets and name not in datasets:
            continue
        units.append(unit)
    return units


class BenchmarkDataset(Dataset):
    def __init__(self, units: str | Path | list[str | Path], task: str = 'recovery',
                 with_sem: bool = True):
        """
        units: a single unit dir, or a list of unit dirs (compose a subset).
        task: 'recovery' or 'prediction' — selects the fixed mask.
        with_sem: load e_sem.npy if present.
        """
        if task not in ('recovery', 'prediction'):
            raise ValueError("task must be 'recovery' or 'prediction'")
        if isinstance(units, (str, Path)):
            units = [units]
        self.task = task
        self._units: list[dict] = []
        self._index: list[tuple[int, int]] = []     # (unit_idx, local_idx)

        mask_file = 'recovery_mask.npy' if task == 'recovery' else 'prediction_mask.npy'
        for u_idx, udir in enumerate(units):
            udir = Path(udir)
            z = np.load(udir / 'corpus.npz')
            sem_path = udir / 'e_sem.npy'
            unit = {
                'name': udir.name,
                'coords': z['coords'], 'tau': z['tau'], 'kin': z['kin'],
                'pad_mask': z['pad_mask'], 'traj_len': z['traj_len'], 'domain': z['domain'],
                'bbox': z['bbox_half'], 'lat0': z['lat0'], 'lon0': z['lon0'],
                't0': z['t0'], 'logdt': z['log_max_dt'],
                'pos_mask': np.load(udir / mask_file),
                'e_sem': (np.load(sem_path, mmap_mode='r')
                          if (with_sem and sem_path.exists()) else None),
            }
            self._units.append(unit)
            self._index += [(u_idx, i) for i in range(len(unit['coords']))]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        u, i = self._index[idx]
        U = self._units[u]
        coords = torch.from_numpy(U['coords'][i].copy())
        item = {
            'x_spatial': coords,
            'coords': coords,
            'tau': torch.from_numpy(U['tau'][i].copy()),
            'kinematics': torch.from_numpy(U['kin'][i].copy()),
            'pad_mask': torch.from_numpy(U['pad_mask'][i].copy()),
            'pos_mask': torch.from_numpy(U['pos_mask'][i].copy()),
            'domain_id': int(U['domain'][i]),
            'traj_len': int(U['traj_len'][i]),
            'dataset': U['name'],
            'target_coords': coords.clone(),
            'denorm': {
                'bbox_half': float(U['bbox'][i]), 'lat0': float(U['lat0'][i]),
                'lon0': float(U['lon0'][i]), 't0': float(U['t0'][i]),
                'log_max_dt': float(U['logdt'][i]),
            },
        }
        item['e_sem'] = (torch.from_numpy(np.asarray(U['e_sem'][i], dtype=np.float32))
                         if U['e_sem'] is not None else None)
        return item


def collate(batch: list[dict]) -> dict:
    """Stack tensors; keep scalars/dicts/strings as per-sample lists."""
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if vals[0] is None:
            out[k] = None
        elif isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, 0)
        elif isinstance(vals[0], int):
            out[k] = torch.tensor(vals, dtype=torch.long)
        else:                       # dict (denorm) / str (dataset)
            out[k] = vals
    return out
