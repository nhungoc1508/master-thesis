"""
Shared utilities for vanilla baselines
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_dataset import BenchmarkDataset, find_units, collate
import metrics

_DOMAIN = {0: 'urban', 1: 'maritime'}

class InputFeaturizer(nn.Module):
    """
    (coords, tau, kin, domain) -> (B, L, d)
    Masked positions get a learned spatial mask token + zeroed kinematics; temporal features (tau) are kept
    """

    def __init__(self, d_model, use_domain=False):   # domain-agnostic by default -> clean domain transfer
        super().__init__()
        self.coord_proj = nn.Linear(2, d_model)
        self.tau_proj = nn.Linear(4, d_model)
        self.kin_proj = nn.Linear(3, d_model)
        self.mask_spatial = nn.Parameter(torch.randn(d_model) * 0.02)
        self.domain_emb = nn.Embedding(2, d_model) if use_domain else None
        self.norm = nn.LayerNorm(d_model)

    def forward(self, coords, tau, kin, domain_id, hide_mask):
        e = self.coord_proj(coords)
        m = hide_mask.unsqueeze(-1)
        e = torch.where(m, self.mask_spatial.view(1, 1, -1).to(e.dtype), e)
        e = e + self.tau_proj(tau)
        k = self.kin_proj(kin)
        e = e + torch.where(m, torch.zeros_like(k), k)
        if self.domain_emb is not None:
            e = e + self.domain_emb(domain_id).unsqueeze(1)
        return self.norm(e)

def make_loader(units, task, bs, nw, shuffle):
    return DataLoader(BenchmarkDataset(units, task=task, with_sem=False),
                      batch_size=bs, shuffle=shuffle, collate_fn=collate, num_workers=nw)

def masked_mse(pred, target, mask):
    m = mask.unsqueeze(-1)
    return (((pred - target) ** 2) * m).sum() / m.float().sum().clamp(min=1) / pred.shape[-1]

@torch.no_grad()
def evaluate(predict_batch, units, task, device, bs=128, nw=2):
    loader = make_loader(units, task, bs, nw, False)
    overall, by_dom, by_ds = [], defaultdict(list), defaultdict(list)
    for b in loader:
        pred = predict_batch(b)
        target = b['target_coords'].numpy()
        pos = b['pos_mask'].numpy(); pad = b['pad_mask'].numpy()
        for i in range(pred.shape[0]):
            mask = pos[i] & ~pad[i]
            if not mask.any():
                continue
            err = metrics.recovery_error_m(pred[i], target[i], mask, b['denorm'][i])
            overall.append(err)
            by_dom[_DOMAIN[int(b['domain_id'][i])]].append(err)
            by_ds[b['dataset'][i]].append(err)
    return {'overall': metrics.aggregate(overall),
            'by_domain': {k: metrics.aggregate(v) for k, v in sorted(by_dom.items())},
            'by_dataset': {k: metrics.aggregate(v) for k, v in sorted(by_ds.items())}}

def print_block(title, res):
    o = res['overall']
    print(f"\n=== {title} ===")
    print(f"  OVERALL mae={o.get('mae_m', float('nan')):.1f}m median={o.get('median_m', float('nan')):.1f}m "
          f"p90={o.get('p90_m', float('nan')):.1f}m (n_traj={o.get('n_trajectories', 0)})")
    for d, m in res['by_domain'].items():
        print(f"  [{d}] mae={m['mae_m']:.1f}m median={m['median_m']:.1f}m n_traj={m['n_trajectories']}")

def gather_masked(batch, device):
    pos = batch['pos_mask']; pad = batch['pad_mask']
    mask = pos & ~pad
    B, L = mask.shape
    Mmax = max(1, int(mask.sum(1).max().item()))
    tau_m = torch.zeros(B, Mmax, 4); tgt_m = torch.zeros(B, Mmax, 2)
    valid = torch.zeros(B, Mmax, dtype=torch.bool); idx_list = []
    tau = batch['tau']; tgt = batch['target_coords']
    for i in range(B):
        idxs = mask[i].nonzero(as_tuple=False).squeeze(1)
        n = len(idxs)
        if n:
            tau_m[i, :n] = tau[i, idxs]; tgt_m[i, :n] = tgt[i, idxs]; valid[i, :n] = True
        idx_list.append(idxs)
    return (tau_m.to(device), tgt_m.to(device), valid.to(device), idx_list, Mmax)

def save_ckpt(path, payload):
    """Persist a trained baseline: {'model': state_dict, 'arch': {...}, ...}. Train once, eval many."""
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(p))

def load_ckpt(path, map_location='cpu'):
    return torch.load(str(path), map_location=map_location, weights_only=False)