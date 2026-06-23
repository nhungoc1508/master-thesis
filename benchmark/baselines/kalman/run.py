"""
DOMAIN: MARITIME
TASKS: PREDICTION + RECOVERY
Kalman filter baseline on the frozen benchmark

State [x, y, vx, vy], constant-velocity transition (dt = index step). Observations
only at visible (~pos_mask) positions. A forward filter + RTS backward smoother
estimate every position.

Read the masked positions:
  - recovery (interior masks): the smoother uses both sides -> interpolation/smoothing
  - prediction (last-N masks): no future obs -> the smoother reduces to forward
        constant-velocity extrapolation
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench_dataset import BenchmarkDataset, find_units
import metrics

_DOMAIN = {0: 'urban', 1: 'maritime'}

def kalman_cv_smooth(coords, visible, n, q, r):
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    Q = q * np.eye(4); R = r * np.eye(2)
    vis_idx = np.where(visible[:n])[0]
    if len(vis_idx) == 0:
        return coords[:n].copy()
    x = np.array([coords[vis_idx[0], 0], coords[vis_idx[0], 1], 0.0, 0.0])
    P = np.eye(4)
    xs_pred, Ps_pred, xs_filt, Ps_filt = [], [], [], []
    for t in range(n):
        x = F @ x; P = F @ P @ F.T + Q
        xs_pred.append(x.copy()); Ps_pred.append(P.copy())
        if visible[t]:
            z = coords[t]
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z - H @ x)
            P = (np.eye(4) - K @ H) @ P
        xs_filt.append(x.copy()); Ps_filt.append(P.copy())
    xs_s = [a.copy() for a in xs_filt]
    for t in range(n - 2, -1, -1):
        Ck = Ps_filt[t] @ F.T @ np.linalg.inv(Ps_pred[t + 1])
        xs_s[t] = xs_filt[t] + Ck @ (xs_s[t + 1] - xs_pred[t + 1])
    return np.array(xs_s)[:, :2]

def evaluate(units, task, q, r):
    ds = BenchmarkDataset(units, task=task, with_sem=False)
    overall, by_dom, by_ds = [], defaultdict(list), defaultdict(list)
    for j in range(len(ds)):
        it = ds[j]; n = int(it['traj_len'])
        coords = it['coords'][:n].numpy()
        pos = it['pos_mask'][:n].numpy(); pad = it['pad_mask'][:n].numpy()
        masked = pos & ~pad; visible = (~pos) & ~pad
        if not masked.any():
            continue
        pred_n = kalman_cv_smooth(coords, visible, n, q, r)
        pred_full = np.zeros((n, 2), np.float32); pred_full[masked] = pred_n[masked]
        err = metrics.recovery_error_m(pred_full, it['target_coords'][:n].numpy(), masked, it['denorm'])
        overall.append(err)
        by_dom[_DOMAIN[int(it['domain_id'])]].append(err)
        by_ds[it['dataset']].append(err)
    return {'overall': metrics.aggregate(overall),
            'by_domain': {k: metrics.aggregate(v) for k, v in sorted(by_dom.items())},
            'by_dataset': {k: metrics.aggregate(v) for k, v in sorted(by_ds.items())}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--task', default='recovery', choices=['recovery', 'prediction'])
    ap.add_argument('--domains', nargs='*', default=None)
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--q', type=float, default=1e-4)
    ap.add_argument('--r', type=float, default=1e-4)
    args = ap.parse_args()
    units = find_units(args.test_dir, domains=args.domains, datasets=args.datasets)
    if not units:
        raise FileNotFoundError('No frozen units found (check dir/filters).')
    res = evaluate(units, args.task, args.q, args.r)
    o = res['overall']
    print(f"\n=== Kalman • {args.task} ===")
    print(f"  OVERALL mae={o.get('mae_m', float('nan')):.1f}m median={o.get('median_m', float('nan')):.1f}m "
          f"p90={o.get('p90_m', float('nan')):.1f}m (n_traj={o.get('n_trajectories',0)})")
    for d, m in res['by_domain'].items():
        print(f"  [{d}] mae={m['mae_m']:.1f}m median={m['median_m']:.1f}m n_traj={m['n_trajectories']}")

if __name__ == '__main__':
    main()