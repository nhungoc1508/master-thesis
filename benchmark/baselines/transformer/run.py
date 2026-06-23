"""
Vanilla Transformer baseline on the frozen benchmark

A single masked encoder serves both tasks via the frozen masks:
    - Recovery = interior points masked -> reconstruct
    - Prediction = last-N points masked -> forecast

Trained end-to-end supervised on the task, no SSL stage
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parents[1]))

from bench_dataset import BenchmarkDataset, find_units, collate
import metrics
import common

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
_DOMAIN ={0: 'urban', 1: 'maritime'}

def _sinusoid(L, d, device):
    pos = torch.arange(L, device=device).unsqueeze(1).float()
    i = torch.arange(0, d, 2, device=device).float()
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(L, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe

class MaskedTransformer(nn.Module):
    def __init__(self, d_model=256, n_layers=6, n_heads=8, dropout=0.1):
        super().__init__()
        self.coord_proj = nn.Linear(2, d_model)
        self.tau_proj = nn.Linear(4, d_model)
        self.kin_proj = nn.Linear(3, d_model)
        self.mask_spatial = nn.Parameter(torch.randn(d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, coords, tau, kin, domain_id, pad_mask, hide_mask):
        e = self.coord_proj(coords)
        m = hide_mask.unsqueeze(-1)
        e = torch.where(m, self.mask_spatial.view(1, 1, -1).to(e.dtype), e)
        e = e + self.tau_proj(tau)
        k = self.kin_proj(kin)
        e = e + torch.where(m, torch.zeros_like(k), k)
        e = self.norm(e) + _sinusoid(e.shape[1], e.shape[2], e.device).unsqueeze(0)
        h = self.encoder(e, src_key_padding_mask=pad_mask)
        return self.head(h)

def _loader(units, task, bs, nw, shuffle):
    return DataLoader(BenchmarkDataset(units, task=task, with_sem=False),
                      batch_size=bs, shuffle=shuffle, collate_fn=collate, num_workers=nw)

def train(model, units, task, device, epochs, bs, nw, lr):
    loader = _loader(units, task, bs, nw, True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for ep in range(1, epochs + 1):
        tot = nb = 0.0
        for b in loader:
            coords = b['coords'].to(device); tau = b['tau'].to(device)
            kin = b['kinematics'].to(device); pad = b['pad_mask'].to(device)
            pos = b['pos_mask'].to(device); dom = b['domain_id'].to(device)
            tgt = b['target_coords'].to(device)
            pred = model(coords, tau, kin, dom, pad, hide_mask=pos)
            mask = (pos & ~pad).unsqueeze(-1)
            loss = (((pred - tgt) ** 2) * mask).sum() / mask.float().sum().clamp(min=1) / 2
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        logger.info('[Transformer %s] epoch %3d/%d | mse=%.6f', task, ep, epochs, tot / max(nb, 1))

@torch.no_grad()
def evaluate(model, units, task, device, bs, nw):
    loader = _loader(units, task, bs, nw, False)
    model.eval()
    overall, by_dom, by_ds = [], defaultdict(list), defaultdict(list)
    for b in loader:
        pred = model(b['coords'].to(device), b['tau'].to(device), b['kinematics'].to(device),
                     b['domain_id'].to(device), b['pad_mask'].to(device),
                     hide_mask=b['pos_mask'].to(device)).cpu().numpy()
        target = b['target_coords'].numpy(); pos = b['pos_mask'].numpy(); pad = b['pad_mask'].numpy()
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir')                  # not required in eval-only (--ckpt) mode
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--task', default='recovery', choices=['recovery', 'prediction'])
    ap.add_argument('--domains', nargs='*', default=None)
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', default=None)
    ap.add_argument('--save-ckpt', default=None)    # after training, persist weights here
    ap.add_argument('--ckpt', default=None)         # load weights + skip training (eval-only)
    args = ap.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    test_units = find_units(args.test_dir, domains=args.domains, datasets=args.datasets)
    if not test_units:
        raise FileNotFoundError('No frozen test units found (check dir/filters).')
    logger.info('Device: %s | task=%s', device, args.task)

    if args.ckpt: # ----- eval-only (region/domain transfer) -----
        ck = common.load_ckpt(args.ckpt, device); a = ck['arch']
        model = MaskedTransformer(a['d_model'], a['layers'], a['heads']).to(device)
        model.load_state_dict(ck['model'])
        logger.info('Loaded checkpoint %s (trained task=%s)', args.ckpt, a.get('task'))
    else: # ----- train -----
        train_units = find_units(args.train_dir, domains=args.domains, datasets=args.datasets) if args.train_dir else None
        if not train_units:
            raise FileNotFoundError('No frozen train units found (pass --train-dir, or --ckpt for eval-only).')
        model = MaskedTransformer(args.d_model, args.layers, args.heads).to(device)
        logger.info('--- Training (supervised masked reconstruction) ---')
        train(model, train_units, args.task, device, args.epochs, args.batch_size, args.num_workers, args.lr)
        if args.save_ckpt:
            common.save_ckpt(args.save_ckpt, {'model': model.state_dict(),
                                              'arch': dict(d_model=args.d_model, layers=args.layers,
                                                           heads=args.heads, task=args.task)})
            logger.info('Saved checkpoint -> %s', args.save_ckpt)
    res = evaluate(model, test_units, args.task, device, args.batch_size, args.num_workers)
    o = res['overall']
    print(f"\n=== Transformer • {args.task} ===")
    print(f"  OVERALL mae={o.get('mae_m', float('nan')):.1f}m median={o.get('median_m', float('nan')):.1f}m "
          f"p90={o.get('p90_m', float('nan')):.1f}m (n_traj={o.get('n_trajectories',0)})")
    for d, m in res['by_domain'].items():
        print(f"  [{d}] mae={m['mae_m']:.1f}m median={m['median_m']:.1f}m n_traj={m['n_trajectories']}")

if __name__ == '__main__':
    main()