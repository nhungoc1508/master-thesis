import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import TrajectoryMaskedAutoEncoder
from config import ModelConfig
from bench_dataset import BenchmarkDataset, find_units, collate
import metrics

_DOMAIN_NAME = {0: 'urban', 1: 'maritime'}

def load_model(ckpt_path: str, device: torch.device):
    """Instantiate model from a checkpoint and load weights"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_obj = ckpt.get('cfg', {})
    if isinstance(cfg_obj, dict):
        cfg = ModelConfig(**cfg_obj)
    else:
        cfg = cfg_obj
    
    model =TrajectoryMaskedAutoEncoder(cfg).to(device)
    
    if 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'Missing keys: {len(missing)} (e.g. {missing[:3]})')
    if unexpected:
        print(f'Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})')
    model.eval()
    if isinstance(ckpt, dict):
        epochs = ckpt.get('epoch')
    else:
        epochs = '?'
    print(f'Loaded model from {ckpt_path} (epochs={epochs})')

    return model

@torch.no_grad()
def evaluate_task(model, units, task, device, batch_size, num_workers, with_sem,
                  max_len=200):
    ds = BenchmarkDataset(units, task=task, with_sem=with_sem)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=num_workers)

    overall = []
    by_domain = defaultdict(list)
    by_dataset = defaultdict(list)

    for batch in loader:
        e_sem = batch['e_sem']
        out = model.forward(
            x_spatial=batch['x_spatial'].to(device),
            tau=batch['tau'].to(device),
            kinematics=batch['kinematics'].to(device),
            coords=batch['coords'].to(device),
            pad_mask=batch['pad_mask'].to(device),
            pos_mask=batch['pos_mask'].to(device),
            domain_ids=batch['domain_id'].to(device),
            e_sem=e_sem.to(device) if e_sem is not None else None
        )
        pred = out['pred'][..., :2].cpu().numpy()
        target = batch['target_coords'].numpy()
        pos = batch['pos_mask'].numpy()
        pad = batch['pad_mask'].numpy()
        tlen = batch['traj_len'].numpy()

        for b in range(pred.shape[0]):
            if max_len is not None and tlen[b] > max_len:
                continue
            mask = pos[b] & ~pad[b]
            if not mask.any():
                continue
            err = metrics.recovery_error_m(pred[b], target[b], mask, batch['denorm'][b])
            overall.append(err)
            by_domain[_DOMAIN_NAME[int(batch['domain_id'][b])]].append(err)
            by_dataset[batch['dataset'][b]].append(err)

    return {
        'overall': metrics.aggregate(overall),
        'by_domain': {k: metrics.aggregate(v) for k, v in sorted(by_domain.items())},
        'by_dataset': {k: metrics.aggregate(v) for k, v in sorted(by_dataset.items())}
    }

@torch.no_grad()
def predict_trajectories(model, unit_dir, task, device, with_sem=True,
                         indices=None, max_traj=None):
    unit_dir = Path(unit_dir)
    ds = BenchmarkDataset(unit_dir, task=task, with_sem=with_sem)
    traj_ids = json.loads((unit_dir / 'traj_ids.json').read_text())
    name = unit_dir.name

    if indices is None:
        if max_traj is None:
            n = len(ds)
        else:
            n = min(max_traj, len(ds))
        indices = list(range(n))
    
    records = []
    for idx in indices:
        it = ds[idx]
        e_sem = it['e_sem']
        out = model.forward(
            x_spatial=it['x_spatial'].unsqueeze(0).to(device),
            tau=it['tau'].unsqueeze(0).to(device),
            kinematics=it['kinematics'].unsqueeze(0).to(device),
            coords=it['coords'].unsqueeze(0).to(device),
            pad_mask=it['pad_mask'].unsqueeze(0).to(device),
            pos_mask=it['pos_mask'].unsqueeze(0).to(device),
            domain_ids=torch.tensor([it['domain_id']], device=device),
            e_sem=e_sem.unsqueeze(0).to(device) if e_sem is not None else None
        )
        traj_len = it['traj_len']
        pred_n = out['pred'][0, :traj_len, :2].cpu().numpy()
        true_n = it['target_coords'][:traj_len].numpy()
        mask = it['pos_mask'][:traj_len].numpy() & ~it['pad_mask'][:traj_len].numpy()
        denorm = it['denorm']
        tlat, tlon = metrics.denorm_coords(true_n, denorm['bbox_half'], denorm['lat0'], denorm['lon0'])
        plat, plon = metrics.denorm_coords(pred_n, denorm['bbox_half'], denorm['lat0'], denorm['lon0'])
        if mask.any():
            err = metrics.haversine_m(tlat[mask], tlon[mask], plat[mask], plon[mask])
        else:
            err = np.array([])

        records.append({
            'traj_id': traj_ids[idx] if idx < len(traj_ids) else str(idx),
            'dataset': name,
            'domain': _DOMAIN_NAME[int(it['domain_id'])],
            'traj_len': traj_len,
            'mask': mask,
            'masked_idx': np.where(mask)[0],
            'true_lat': tlat,
            'true_lon': tlon,
            'pred_lat': plat,
            'pred_lon': plon,
            'err_m': err
        })
    return records