"""
Linear interpolation floor on frozen benchmark

Per masked position, in the normalized offset space:
  - interior (visible points on both sides)  -> linear interpolation by index
  - tail / last-N (visible only on the left) -> constant-velocity extrapolation
  - head (visible only on the right) -> constant-velocity extrapolation
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

from bench_dataset import BenchmarkDataset, find_units
import metrics

_DOMAIN = {0: 'urban', 1: 'maritime'}

def _interp_traj(coords, visible, masked):
    pred = coords.copy()
    vis_idx = np.where(visible)[0]
    if len(vis_idx) == 0:
        return pred
    for i in np.where(masked)[0]:
        left = vis_idx[vis_idx < i]
        right = vis_idx[vis_idx > i]
        if len(left) and len(right): # interior -> interpolate
            lo, hi = left[-1], right[0]
            f = (i - lo) / (hi - lo)
            pred[i] = coords[lo] * (1 - f) + coords[hi] * f
        elif len(left) >= 2: # tail -> constant velocity
            a, b = left[-2], left[-1]
            pred[i] = coords[b] + (i - b) * (coords[b] - coords[a])
        elif len(left) == 1:
            pred[i] = coords[left[-1]]
        elif len(right) >= 2: # head -> constant velocity
            a, b = right[0], right[1]
            pred[i] = coords[a] - (a - i) * (coords[b] - coords[a])
        else:
            pred[i] = coords[right[0]]
    return pred

def evaluate(units, task):
    ds = BenchmarkDataset(units, task=task, with_sem=False)
    overall, by_dom, by_ds = [], defaultdict(list), defaultdict(list)
    for j in range(len(ds)):
        it = ds[j]
        tlen = int(it['traj_len'])
        coords = it['coords'][:tlen].numpy()
        pos = it['pos_mask'][:tlen].numpy(); pad = it['pad_mask'][:tlen].numpy()
        masked = pos & ~pad; visible = (~pos) & ~pad
        if not masked.any():
            continue
        pred = _interp_traj(coords, visible, masked)
        target = it['target_coords'][:tlen].numpy()
        err = metrics.recovery_error_m(pred, target, masked, it['denorm'])
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
    args = ap.parse_args()
    units = find_units(args.test_dir, domains=args.domains, datasets=args.datasets)
    if not units:
        raise FileNotFoundError('No frozen units found (check dir/filters).')
    res = evaluate(units, args.task)
    o = res['overall']
    print(f"\n=== Linear • {args.task} ===")
    print(f"  OVERALL mae={o.get('mae_m', float('nan')):.1f}m median={o.get('median_m', float('nan')):.1f}m "
          f"p90={o.get('p90_m', float('nan')):.1f}m (n_traj={o.get('n_trajectories',0)})")
    for d, m in res['by_domain'].items():
        print(f"  [{d}] mae={m['mae_m']:.1f}m median={m['median_m']:.1f}m n_traj={m['n_trajectories']}")

if __name__ == '__main__':
    main()