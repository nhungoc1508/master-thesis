import json
import pickle
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

@torch.no_grad()
def predict_and_save(model, units, task, device, *, batch_size=256, num_workers=4,
                     with_sem=True, max_len=200, save_path=None, store_full_pred=True):
    if isinstance(units, (str, Path)):
        units = [units]
    units = [Path(u) for u in units]

    traj_ids_by_ds, ds_counter = {}, defaultdict(int)
    for u in units:
        tj = u / 'traj_ids.json'
        traj_ids_by_ds[u.name] = json.loads(tj.read_text()) if tj.exists() else None

    ds = BenchmarkDataset(units, task=task, with_sem=with_sem)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=num_workers)

    records = []
    overall, by_domain, by_dataset = [], defaultdict(list), defaultdict(list)

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
        doms = batch['domain_id'].numpy()
        names = batch['dataset']
        denorms = batch['denorm']

        for b in range(pred.shape[0]):
            name = names[b]
            tids = traj_ids_by_ds.get(name)
            j = ds_counter[name]; ds_counter[name] += 1
            traj_id = tids[j] if (tids is not None and j < len(tids)) else f'{name}:{j}'

            n = int(tlen[b])
            if max_len is not None and n > max_len:
                continue
            mask = pos[b, :n] & ~pad[b, :n]
            if not mask.any():
                continue

            dn = denorms[b]
            tlat, tlon = metrics.denorm_coords(target[b, :n], dn['bbox_half'], dn['lat0'], dn['lon0'])
            plat, plon = metrics.denorm_coords(pred[b, :n], dn['bbox_half'], dn['lat0'], dn['lon0'])
            err = metrics.haversine_m(tlat[mask], tlon[mask], plat[mask], plon[mask])

            overall.append(err)
            by_domain[_DOMAIN_NAME[int(doms[b])]].append(err)
            by_dataset[name].append(err)

            rec = {
                'uid': f'{name}#{j}',          # globally-unique (traj_id is NOT unique across days)
                'traj_id': traj_id,
                'dataset': name,
                'domain': _DOMAIN_NAME[int(doms[b])],
                'traj_len': n,
                'mask': mask,
                'masked_idx': np.where(mask)[0].astype(np.int32),
                'true_lat': tlat.astype(np.float32),
                'true_lon': tlon.astype(np.float32),
                'err_m': err.astype(np.float32),
            }
            if store_full_pred:
                rec['pred_lat'], rec['pred_lon'] = plat.astype(np.float32), plon.astype(np.float32)
            else:
                rec['pred_lat'], rec['pred_lon'] = plat[mask].astype(np.float32), plon[mask].astype(np.float32)
            records.append(rec)

    result = {
        'records': records,
        'metrics': {
            'overall': metrics.aggregate(overall),
            'by_domain': {k: metrics.aggregate(v) for k, v in sorted(by_domain.items())},
            'by_dataset': {k: metrics.aggregate(v) for k, v in sorted(by_dataset.items())},
        },
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as fh:
            pickle.dump({'records': records, 'metrics': result['metrics'],
                         'meta': {'task': task, 'with_sem': with_sem, 'max_len': max_len,
                                  'store_full_pred': store_full_pred}}, fh)
        print(f'Saved {len(records)} predictions -> {save_path}')

    return result

def load_predictions(save_path):
    with open(Path(save_path), 'rb') as fh:
        return pickle.load(fh)

# ===================== UniTraj baseline: predict + save =====================

_WT_MEAN = np.array([5.3311563533497974e-05, -7.49477039789781e-05], dtype=np.float64)
_WT_STD = np.array([0.049923088401556015, 0.040688566863536835], dtype=np.float64)

def load_unitraj_model(repo_dir, model_path=None, device='cpu', max_len=200):
    repo_dir = Path(repo_dir)
    model_path = Path(model_path) if model_path else repo_dir / 'model.pt'
    sys.path.insert(0, str(repo_dir / 'utils'))
    from unitraj import UniTraj
    model = UniTraj(trajectory_length=max_len, patch_size=1, embedding_dim=128,
                    encoder_layers=8, encoder_heads=4, decoder_layers=4,
                    decoder_heads=4, mask_ratio=0.5)
    state = torch.load(str(model_path), map_location=device)
    if isinstance(state, dict) and 'state_dict' in state and not any(k.startswith('encoder') for k in state):
        state = state['state_dict']
    model.load_state_dict(state)
    print(f'Loaded UniTraj from {model_path} (trajectory_length={max_len})')
    return model.to(device).eval()

def _unitraj_stats(recs, scope):
    if scope == 'worldtrace':
        return defaultdict(lambda: (_WT_MEAN, _WT_STD))
    by = defaultdict(list)
    for r in recs:
        by[r['dataset']].append(r['lonlat'] - r['lonlat'][0])
    stats = {}
    for name, offs in by.items():
        allo = np.concatenate(offs, axis=0)
        if scope == 'robust':
            mean = np.median(allo, 0)
            std = (np.percentile(allo, 84, 0) - np.percentile(allo, 16, 0)) / 2.0
        else:
            mean, std = allo.mean(0), allo.std(0)
        std = np.where(std < 1e-9, 1.0, std)
        stats[name] = (mean, std)
    return stats

@torch.no_grad()
def predict_and_save_unitraj(model, units, task, device, *, norm='robust', intervals='clamp',
                             clamp_dt=15.0, batch_size=256, save_path=None, store_full_pred=True):
    if isinstance(units, (str, Path)):
        units = [units]
    units = [Path(u) for u in units]
    max_len = int(model.encoder.num_tokens)

    traj_ids_by_ds, ds_counter = {}, defaultdict(int)
    for u in units:
        tj = u / 'traj_ids.json'
        traj_ids_by_ds[u.name] = json.loads(tj.read_text()) if tj.exists() else None

    ds = BenchmarkDataset(units, task=task, with_sem=False)
    recs = []
    for j in range(len(ds)):
        it = ds[j]
        name = it['dataset']
        tids = traj_ids_by_ds.get(name)
        c = ds_counter[name]; ds_counter[name] += 1
        traj_id = tids[c] if (tids is not None and c < len(tids)) else f'{name}:{c}'
        tlen = int(it['traj_len'])
        if tlen > max_len:
            continue
        coords = it['coords'][:tlen].numpy(); d = it['denorm']
        lat = coords[:, 0] * d['bbox_half'] + d['lat0']
        lon = coords[:, 1] * d['bbox_half'] + d['lon0']
        lonlat = np.stack([lon, lat], axis=1).astype(np.float64)
        dt = np.expm1(it['tau'][:tlen, 3].numpy().astype(np.float64) * d['log_max_dt'])
        pos = it['pos_mask'][:tlen].numpy(); pad = it['pad_mask'][:tlen].numpy()
        masked = np.where(pos & ~pad)[0]
        if len(masked) == 0 or len(masked) >= tlen:
            continue
        recs.append({'uid': f'{name}#{c}', 'traj_id': traj_id, 'dataset': name,
                     'domain': _DOMAIN_NAME[int(it['domain_id'])],
                     'lonlat': lonlat, 'dt': dt, 'masked': masked, 'tlen': tlen})

    stats = _unitraj_stats(recs, norm)

    buckets = defaultdict(list)
    for i, r in enumerate(recs):
        buckets[len(r['masked'])].append(i)

    records = []
    overall, by_domain, by_dataset = [], defaultdict(list), defaultdict(list)
    for K, idxs in buckets.items():
        for b0 in range(0, len(idxs), batch_size):
            chunk = idxs[b0:b0 + batch_size]
            B = len(chunk)
            traj = np.zeros((B, 2, max_len), np.float32)
            interv = np.zeros((B, max_len), np.float32)
            mask_idx = np.zeros((B, K), np.int64)
            for bi, ri in enumerate(chunk):
                r = recs[ri]; mean, std = stats[r['dataset']]; t = r['tlen']
                traj[bi, :, :t] = ((r['lonlat'] - r['lonlat'][0] - mean) / std).T.astype(np.float32)
                dt = np.clip(r['dt'], 0.0, clamp_dt) if intervals == 'clamp' else r['dt']
                interv[bi, :t] = dt.astype(np.float32)
                mask_idx[bi] = r['masked']
            interv_t = None if intervals == 'none' else torch.from_numpy(interv).to(device)
            pred, _ = model(torch.from_numpy(traj).to(device), interv_t,
                            torch.from_numpy(mask_idx).to(device))
            pred = pred.cpu().numpy()
            for bi, ri in enumerate(chunk):
                r = recs[ri]; mean, std = stats[r['dataset']]; t = r['tlen']; m = r['masked']
                pworld = pred[bi].T[:t] * std + mean + r['lonlat'][0]
                tlat, tlon = r['lonlat'][:, 1], r['lonlat'][:, 0]
                plat, plon = pworld[:, 1], pworld[:, 0]
                err = metrics.haversine_m(tlat[m], tlon[m], plat[m], plon[m])
                overall.append(err)
                by_domain[r['domain']].append(err)
                by_dataset[r['dataset']].append(err)
                mask_full = np.zeros(t, dtype=bool); mask_full[m] = True
                rec = {'uid': r['uid'], 'traj_id': r['traj_id'], 'dataset': r['dataset'], 'domain': r['domain'],
                       'traj_len': t, 'mask': mask_full, 'masked_idx': m.astype(np.int32),
                       'true_lat': tlat.astype(np.float32), 'true_lon': tlon.astype(np.float32),
                       'err_m': err.astype(np.float32)}
                if store_full_pred:
                    rec['pred_lat'], rec['pred_lon'] = plat.astype(np.float32), plon.astype(np.float32)
                else:
                    rec['pred_lat'], rec['pred_lon'] = plat[m].astype(np.float32), plon[m].astype(np.float32)
                records.append(rec)

    result = {'records': records, 'metrics': {
        'overall': metrics.aggregate(overall),
        'by_domain': {k: metrics.aggregate(v) for k, v in sorted(by_domain.items())},
        'by_dataset': {k: metrics.aggregate(v) for k, v in sorted(by_dataset.items())}}}

    if save_path is not None:
        save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as fh:
            pickle.dump({'records': records, 'metrics': result['metrics'],
                         'meta': {'model': 'unitraj', 'task': task, 'norm': norm,
                                  'intervals': intervals, 'clamp_dt': clamp_dt, 'max_len': max_len,
                                  'store_full_pred': store_full_pred}}, fh)
        print(f'Saved {len(records)} UniTraj predictions -> {save_path}')
    return result