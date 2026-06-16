"""
Freeze a benchmark split into fixed tensors for training/evaluating baselines

Different tasks (recovery & prediction) share corpus, differ only by masks

Output layout:
    [out_dir]/[split]/[domain]/[dataset]/:
        corpus.npz
        recovery_mask.npy
        prediction_mask.npy
        e_sem.npy (optional)
        traj_ids.json
        manifest.json

Usage:
    python freeze_benchmark.py \\
        --urban-parquet [...]
        --maritime-parquet [...]
        --urban-sem-npy [...]
        --maritime-sem-npy [...]
        --split test
        --out-dir [...]
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

from data import TrajectoryDataset
from masking import make_pos_mask

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

def _parquets(p: str | None) -> list[Path]:
    """Get paths to all .parquet files"""
    if not p:
        return []
    base = Path(p)
    return sorted(base.glob('**/*.parquet')) if base.is_dir() else [base]

def _sem_for(parquet: Path, sem_dir: str | None, domain: str) -> Path | None:
    """Find matching .npy file for a .parquet file"""
    if not sem_dir:
        return None
    sem_dir = Path(sem_dir)
    stem = parquet.stem
    if domain == 'maritime':
        stem = stem.removesuffix('_canonical')
    for cand in (sem_dir / f'{stem}_sem.npy', sem_dir / f'{stem}_described_sem.npy'):
        if cand.exists():
            return cand
    logger.warning('No sem .npy found for %s', parquet.name)
    return None

def _dataset_name(parquet: Path, domain: str) -> str:
    """Get folder name for this source dataset's frozen unit"""
    stem = parquet.stem
    if domain == 'maritime':
        stem = stem.removesuffix('_canonical')
    return stem

def freeze_one(parquet: Path, sem_dir: str | None, domain: str, args) -> None:
    """Freeze one source dataset into its own unit directory"""
    name = _dataset_name(parquet, domain)
    out_dir = Path(args.out_dir) / args.split / domain / name
    if (out_dir / 'corpus.npz').exists() and not args.overwrite:
        logger.info('[%s/%s] already frozen, skipping', domain, name)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    sem_path = _sem_for(parquet, sem_dir, domain)
    ds = TrajectoryDataset(
        [parquet], domain=domain, max_len=args.max_len, input_dim=args.input_dim,
        sem_npy_path=[sem_path], include_kinematics=True
    )
    N = len(ds)
    if N == 0:
        logger.warning('[%s/%s] no trajectories, skipping', domain, name)
        return
    logger.info('[%s/%s] freezing %d trajectories', domain, name, N)

    col_names = ['coords', 'tau', 'kin', 'pad', 'rec', 'prd', 'sem', 'traj_len', 'bbox_half', 'lat0', 'lon0', 't0', 'log_max_dt', 'tid']
    cols = {k: [] for k in col_names}
    for i in range(N):
        np.random.seed(args.seed + i)
        item = ds[i]
        dn = ds.trajectories[i]['denorm']
        tlen = int(item['traj_len'])
        coords = item['coords'].numpy()
        cols['coords'].append(coords)
        cols['tau'].append(item['tau'].numpy())
        if item['kinematics'] is not None:
            kin = item['kinematics'].numpy()
        else:
            kin = np.zeros((args.max_len, 3), np.float32)
        cols['kin'].append(kin)
        cols['pad'].append(item['pad_mask'].numpy())
        if item['e_sem'] is not None:
            sem = item['e_sem'].numpy()
        else:
            sem = None
        cols['sem'].append(sem)
        cols['traj_len'].append(tlen)
        cols['tid'].append(item['trajectory_id'])
        for k in ('bbox_half', 'lat0', 'lon0', 't0', 'log_max_dt'):
            cols[k].append(dn[k])

        mrng = np.random.default_rng(args.seed * 1000 + i)
        rec = np.zeros(args.max_len, dtype=bool)
        rec[:tlen] = make_pos_mask(args.recovery_mode, tlen, coords=coords[:tlen], rng=mrng)
        prd = np.zeros(args.max_len, dtype=bool)
        prd[:tlen] = make_pos_mask('last_n', tlen, coords=coords[:tlen], rng=mrng)
        cols['rec'].append(rec)
        cols['prd'].append(prd)

    np.savez(
        out_dir / 'corpus.npz',
        coords=np.stack(cols['coords']).astype(np.float32),
        tau=np.stack(cols['tau']).astype(np.float32),
        kin=np.stack(cols['kin']).astype(np.float32),
        pad_mask=np.stack(cols['pad']),
        traj_len=np.array(cols['traj_len'], np.int32),
        domain=np.full(N, 0 if domain == 'urban' else 1, np.int8),
        bbox_half=np.array(cols['bbox_half'], np.float64),
        lat0=np.array(cols['lat0'], np.float64),
        lon0=np.array(cols['lon0'], np.float64),
        t0=np.array(cols['t0'], np.float64),
        log_max_dt=np.array(cols['log_max_dt'], np.float64)
    )
    np.save(out_dir / 'recovery_mask.npy', np.stack(cols['rec']))
    np.save(out_dir / 'prediction_mask.npy', np.stack(cols['prd']))
    (out_dir / 'traj_ids.json').write_text(json.dumps(cols['tid'], indent=4))

    has_sem = args.with_sem and cols['sem'][0] is not None
    if has_sem:
        np.save(out_dir / 'e_sem.npy', np.stack(cols['sem']).astype(np.float16))

    (out_dir / 'manifest.json').write_text(json.dumps({
        'dataset': name, 'domain': domain, 'split': args.split,
        'n_trajectories': N, 'max_len': args.max_len,
        'input_dim': args.input_dim, 'seed': args.seed,
        'recovery_mode': args.recovery_mode, 'prediction_mode': 'last_n',
        'has_sem': bool(has_sem), 'source_parquet': str(parquet),
        'date': str(date.today())
    }, indent=4))
    logger.info('[%s/%s] -> %s', domain, name, out_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urban-parquet', default=None)
    parser.add_argument('--maritime-parquet', default=None)
    parser.add_argument('--urban-sem-npy', default=None)
    parser.add_argument('--maritime-sem-npy', default=None)
    parser.add_argument('--split', required=True)
    parser.add_argument('--out-dir', default='frozen')
    parser.add_argument('--max-len', type=int, default=256)
    parser.add_argument('--input-dim', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--recovery-mode', default='spatial_random',
                        choices=['spatial_random', 'block', 'key_point'])
    parser.add_argument('--no-sem', dest='with_sem', action='store_false')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    units = ([(p, args.urban_sem_npy, 'urban') for p in _parquets(args.urban_parquet)] +
             [(p, args.maritime_sem_npy, 'maritime') for p in _parquets(args.maritime_parquet)])
    if not units:
        raise FileNotFoundError('No .parquet files found')
    logger.info('Freezing %d source datasets into %s/%s/', len(units), args.out_dir, args.split)
    for parquet, sem_dir, domain in units:
        freeze_one(parquet, sem_dir, domain, args)
    logger.info('Complete: %s units frozen', len(units))

if __name__ == '__main__':
    main()