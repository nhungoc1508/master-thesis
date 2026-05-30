#!/usr/bin/env python3
"""
PHASE 3: Split train / val /test sets dataset-wise & chronologically

Implementation details:
    - Split a canonical/enriched .parquet file into 3 non-overlapping subsets
        based on the start timestamp of each trajectory
    - Default split: 70% train - 15% val - 15% test
    - Use accompanying manifest file to save memory
    - Cutoff is at the trajectory level:
        1. Compute each trajectory's start time
        2. Sort all trajectories by start time
        3. Partition sorted dataset into 3 sets in 70/15/15 ratio

Usage examples:
    Default 70/15/15 split:
        python run_split.py data/enriched/porto_enriched.parquet

    Custom ratios:
        python run_split.py data/canonical/porto.parquet --train 0.8 --val 0.1 --test 0.1
    
    Custom output directory:
        python run_split.py data/enriched/porto_enriched.parquet --output-dir data/splits/

Output files (written to the same input dir unless --output-dir is specified):
    {stem}_train.parquet
    {stem}_val.parquet
    {stem}_test.parquet
    {stem}_split_summary.csv
"""
import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault('NUMEXPR_MAX_THREADS', '1')

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s    %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def _traj_order_from_manifest(manifest_path: Path) -> pd.DataFrame:
    """Sort trajectories by timestamps using the manifest"""
    logger.info('Using manifest %s for chronological ordering', manifest_path)
    m = pd.read_csv(manifest_path, usecols=['trajectory_id', 'start_ts'])
    return m.sort_values('start_ts').reset_index(drop=True)

def _traj_order_from_parquet(input_path: Path) -> pd.DataFrame:
    """Fallback: sort trajectories by timestamps using the full .parquet"""
    logger.info('No manifest found, scanning .parquet for chronological ordering')
    pf = pq.ParquetFile(input_path)
    records: dict[str, int] = {} # mapping trajectory_id -> start timestamp

    for batch in pf.iter_batches(columns=['trajectory_id', 'timestamp']):
        tbl = pa.Table.from_batches([batch])
        df = tbl.to_pandas()
        mins = df.groupby('trajectory_id', sort=False)['timestamp'].min()
        for tid, ts in mins.items():
            if tid not in records or ts < records[tid]:
                records[tid] = ts

    traj_order = (
        pd.DataFrame({'trajectory_id': list(records.keys()),
                      'start_ts': list(records.values())})
            .sort_values('start_ts')
            .reset_index(drop=True)
    )
    return traj_order

def _write_split(input_path: Path, output_path: Path, ids: set[str]) -> int:
    """
    Filter input_path row-group by row-group, keep rows in `ids`, write to output_path
    Returns row count written to files
    """
    pf = pq.ParquetFile(input_path)
    schema = pf.schema_arrow
    writer = None
    total = 0

    for batch in pf.iter_batches():
        tbl = pa.Table.from_batches([batch], schema=schema)
        # Emit only points included in the `ids` list
        mask = pa.array(
            [tid in ids for tid in tbl.column('trajectory_id').to_pylist()]
        )
        filtered = tbl.filter(mask)
        if filtered.num_rows == 0:
            continue
        if writer is None:
            writer = pq.ParquetWriter(output_path, schema)
        writer.write_table(filtered)
        total += filtered.num_rows
    
    if writer:
        writer.close()

    return total

def split(input_path: Path, output_dir: Path,
          train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15) -> dict[str, Path]:
    """
    Split trajectories chronologically & write to 3 .parquet files
    Returns a dict { 'train': output path, 'val': output path, 'test': output path }
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f'Split ratios must sum up to 1.0, instead got '
            f'{train_ratio + val_ratio + test_ratio:.4f}'
        )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    # ---------- Load trajectory order ----------
    # Manifest file is in data/canonical/, so it will be missing for enriched files
    manifest_base_path = input_path.parent.parent / 'canonical'
    if input_path.parent.name == 'enriched':
        file_stem = input_path.stem.replace('_enriched', '')
    else:
        file_stem = input_path.stem
    manifest_path = manifest_base_path / f'{file_stem}.manifest.csv'
    if manifest_path.exists():
        traj_order = _traj_order_from_manifest(manifest_path)
    else:
        traj_order = _traj_order_from_parquet(input_path)

    n = len(traj_order)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    logger.info(
        '%d total trajectories -> train %d (%.0f%%) | val %d (%.0f%%) | test %d (%.0f%%)',
        n, n_train, 100 * train_ratio, n_val, 100 * val_ratio, n_test, 100 * test_ratio
    )

    splits = {
        'train': set(traj_order.iloc[: n_train]['trajectory_id']),
        'val': set(traj_order.iloc[n_train : n_train + n_val]['trajectory_id']),
        'test': set(traj_order.iloc[n_train + n_val :]['trajectory_id'])
    }

    # ---------- Helper function: get time boundaries ----------
    def ts_range(ids: set[str]) -> tuple[str, str]:
        rows = traj_order[traj_order['trajectory_id'].isin(ids)]
        lo = pd.to_datetime(rows['start_ts'].min(), unit='s', utc=True)
        hi = pd.to_datetime(rows['start_ts'].max(), unit='s', utc=True)
        return lo.strftime('%Y-%m-%d'), hi.strftime('%Y-%m-%d')
    
    # ---------- Write splits using row-group streaming ----------
    summary_rows = []
    output_paths: dict[str, Path] = {}

    for split_name, ids in splits.items():
        out_path = output_dir / f'{stem}_{split_name}.parquet'
        output_paths[split_name] = out_path
        n_rows = _write_split(input_path, out_path, ids)

        t_lo, t_hi = ts_range(ids)
        logger.info(
            '\t%-5s %6d trajectories %9d points [%s - %s] -> %s',
            split_name, len(ids), n_rows, t_lo, t_hi, out_path.name
        )
        summary_rows.append({
            'split':          split_name,
            'n_trajectories': len(ids),
            'n_points':       n_rows,
            'start_date':     t_lo,
            'end_date':       t_hi
        })
    
    # ---------- Write summary .csv ----------
    summary_path = output_dir / f'{stem}_split_summary.csv'
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    logger.info('Split summary written to %s', summary_path)

    return output_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--train', type=float, default=0.70, metavar='RATIO')
    parser.add_argument('--val', type=float, default=0.15, metavar='RATIO')
    parser.add_argument('--test', type=float, default=0.15, metavar='RATIO')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    split(
        input_path  = input_path,
        output_dir  = output_dir,
        train_ratio = args.train,
        val_ratio   = args.val,
        test_ratio  = args.test
    )

if __name__ == '__main__':
    main()