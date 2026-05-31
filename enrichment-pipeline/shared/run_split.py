#!/usr/bin/env python3
"""
Split a .parquet file + paired .npy embeddings file into train/val/test splits,
splitting chronologically at the trajectory level.

Usage:
    Parquet only (no semantic embeddings):
        python run_split.py data/enriched/porto_enriched.parquet

    Parquet + npy:
        python run_split.py data/enriched/porto_enriched.parquet \
            --npy data/encoded/porto_enriched_described_sem.npy

    Custom ratios:
        python run_split.py porto_enriched.parquet \
            --npy porto_enriched_described_sem.npy \
            --train 0.8 --val 0.1 --test 0.1

    Custom output directory:
        python run_split.py porto_enriched.parquet \
            --npy porto_enriched_described_sem.npy \
            --output-dir data/splits/

Output files (written to same directory as input unless --output-dir set):
    {stem}_train.parquet        {stem}_train_sem.npy
    {stem}_val.parquet          {stem}_val_sem.npy
    {stem}_test.parquet         {stem}_test_sem.npy
    {stem}_split_summary.csv
"""
import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault('NUMEXPR_MAX_THREADS', '1')

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s    %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

_NP_CHUNK = 100_000   # rows per chunk when writing npy splits

def _ts_column(input_path: Path) -> str:
    """Detect whether the parquet uses 'ts_unix' (maritime) or 'timestamp' (urban)."""
    names = pq.ParquetFile(input_path).schema_arrow.names
    for candidate in ('ts_unix', 'timestamp'):
        if candidate in names:
            return candidate
    raise ValueError(
        f'{input_path.name}: no timestamp column found (tried ts_unix, timestamp). '
        f'Columns present: {names}'
    )

def _traj_order(input_path: Path) -> pd.DataFrame:
    """
    Scan the parquet to find each trajectory's earliest timestamp.
    Returns a DataFrame sorted by start_ts (ascending),
    columns: trajectory_id, start_ts.
    """
    ts_col = _ts_column(input_path)
    logger.info('Scanning %s for trajectory order (ts column: %s)', input_path.name, ts_col)

    pf = pq.ParquetFile(input_path)
    records: dict[str, int] = {}

    for batch in pf.iter_batches(columns=['trajectory_id', ts_col]):
        df = pa.Table.from_batches([batch]).to_pandas()
        mins = df.groupby('trajectory_id', sort=False)[ts_col].min()
        for tid, ts in mins.items():
            if tid not in records or ts < records[tid]:
                records[tid] = int(ts)

    return (
        pd.DataFrame({'trajectory_id': list(records.keys()),
                      'start_ts':      list(records.values())})
          .sort_values('start_ts')
          .reset_index(drop=True)
    )

def _collect_row_indices(input_path: Path,
                         tid_to_split: dict[str, str]) -> dict[str, list[int]]:
    """
    Single lightweight pass over only the trajectory_id column
    Returns {split_name: [sorted absolute row indices]} for npy slicing
    """
    split_names = set(tid_to_split.values())
    indices: dict[str, list[int]] = {s: [] for s in split_names}
    abs_row = 0

    pf = pq.ParquetFile(input_path)
    for batch in pf.iter_batches(columns=['trajectory_id']):
        tids = batch.column('trajectory_id').to_pylist()
        for i, tid in enumerate(tids):
            sname = tid_to_split.get(str(tid))
            if sname is not None:
                indices[sname].append(abs_row + i)
        abs_row += len(tids)

    return indices

def _write_parquet_split(input_path: Path, output_path: Path, ids: set[str]) -> int:
    """
    Stream input_path row-group by row-group, keep only rows whose
    trajectory_id is in `ids`, write to output_path
    Returns the number of rows written.
    """
    pf = pq.ParquetFile(input_path)
    schema = pf.schema_arrow
    writer = None
    total = 0

    for batch in pf.iter_batches():
        tbl = pa.Table.from_batches([batch], schema=schema)
        mask = pa.array([tid in ids for tid in tbl.column('trajectory_id').to_pylist()])
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

def _write_npy_split(src_path: Path, dst_path: Path, indices: list[int]) -> None:
    """
    Write the rows at `indices` from the source npy memmap to a new npy memmap.
    Source is never fully loaded; reads and writes in chunks of _NP_CHUNK rows.
    """
    src = np.lib.format.open_memmap(src_path, mode='r')
    n, dim = len(indices), src.shape[1]
    idx_arr = np.asarray(indices, dtype=np.int64)  # already in ascending order

    dst = np.lib.format.open_memmap(dst_path, mode='w+', dtype=src.dtype, shape=(n, dim))
    for start in range(0, n, _NP_CHUNK):
        end = min(start + _NP_CHUNK, n)
        dst[start:end] = src[idx_arr[start:end]]
    dst.flush()

def split(
    input_path: Path,
    npy_path: Path | None,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, Path]:
    """
    Split trajectories chronologically and write parquet (+ optional npy) splits.
    Returns {split_name: output_parquet_path}.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f'Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.4f}'
        )

    # Validate npy row count matches parquet
    if npy_path is not None:
        npy_n = np.lib.format.open_memmap(npy_path, mode='r').shape[0]
        parquet_n = pq.ParquetFile(input_path).metadata.num_rows
        if npy_n != parquet_n:
            raise ValueError(
                f'Row count mismatch: parquet has {parquet_n} rows '
                f'but npy has {npy_n} rows. '
                f'Ensure the npy was generated from the same parquet file.'
            )
        logger.info('Row count check passed: %d rows in both parquet and npy', parquet_n)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    # 1. Determine chronological trajectory order
    traj_order = _traj_order(input_path)
    n = len(traj_order)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val
    logger.info(
        '%d trajectories -> train %d (%.0f%%) | val %d (%.0f%%) | test %d (%.0f%%)',
        n, n_train, 100 * train_ratio, n_val, 100 * val_ratio, n_test, 100 * test_ratio
    )

    splits_ids: dict[str, set[str]] = {
        'train': set(traj_order.iloc[:n_train]['trajectory_id']),
        'val':   set(traj_order.iloc[n_train : n_train + n_val]['trajectory_id']),
        'test':  set(traj_order.iloc[n_train + n_val:]['trajectory_id']),
    }

    # 2. Collect row indices (needed for npy; one lightweight pass)
    row_indices: dict[str, list[int]] | None = None
    if npy_path is not None:
        tid_to_split = {tid: sname
                        for sname, ids in splits_ids.items()
                        for tid in ids}
        logger.info('Collecting row indices for npy splitting')
        row_indices = _collect_row_indices(input_path, tid_to_split)

    # Helper: human-readable date range for a split
    def ts_range(ids: set[str]) -> tuple[str, str]:
        rows = traj_order[traj_order['trajectory_id'].isin(ids)]
        lo = pd.to_datetime(rows['start_ts'].min(), unit='s', utc=True)
        hi = pd.to_datetime(rows['start_ts'].max(), unit='s', utc=True)
        return lo.strftime('%Y-%m-%d'), hi.strftime('%Y-%m-%d')

    # 3. Write parquet splits (3 streaming passes)
    summary_rows = []
    output_paths: dict[str, Path] = {}

    for sname, ids in splits_ids.items():
        parquet_dir = output_dir / sname / 'parquet'
        parquet_dir.mkdir(parents=True, exist_ok=True)
        out_parquet = parquet_dir / f'{stem}_{sname}.parquet'
        output_paths[sname] = out_parquet
        n_rows = _write_parquet_split(input_path, out_parquet, ids)
        t_lo, t_hi = ts_range(ids)
        logger.info(
            '\t%-5s  %6d trajectories  %9d rows  [%s - %s]  ->  %s',
            sname, len(ids), n_rows, t_lo, t_hi, out_parquet.relative_to(output_dir)
        )
        summary_rows.append({
            'split':          sname,
            'n_trajectories': len(ids),
            'n_rows':         n_rows,
            'start_date':     t_lo,
            'end_date':       t_hi,
            'npy_path':       '',
        })

    # 4. Write npy splits
    if npy_path is not None and row_indices is not None:
        logger.info('Writing npy splits from %s', npy_path.name)
        for sname, indices in row_indices.items():
            if not indices:
                logger.warning('Split %s has no rows in npy; skipping', sname)
                continue
            npy_dir = output_dir / sname / 'npy'
            npy_dir.mkdir(parents=True, exist_ok=True)
            out_npy = npy_dir / f'{stem}_{sname}_sem.npy'
            _write_npy_split(npy_path, out_npy, indices)
            logger.info('\t%-5s  %d rows -> %s', sname, len(indices), out_npy.relative_to(output_dir))
            for row in summary_rows:
                if row['split'] == sname:
                    row['npy_path'] = str(out_npy)

    # 5. Write summary CSV
    summary_path = output_dir / f'{stem}_split_summary.csv'
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    logger.info('Summary -> %s', summary_path)

    return output_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--npy', default=None, metavar='PATH')
    parser.add_argument('--output-dir', default=None, metavar='DIR')
    parser.add_argument('--train', type=float, default=0.70, metavar='RATIO')
    parser.add_argument('--val',   type=float, default=0.15, metavar='RATIO')
    parser.add_argument('--test',  type=float, default=0.15, metavar='RATIO')
    args = parser.parse_args()

    input_path = Path(args.input)
    npy_path   = Path(args.npy) if args.npy else None
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    split(
        input_path  = input_path,
        npy_path    = npy_path,
        output_dir  = output_dir,
        train_ratio = args.train,
        val_ratio   = args.val,
        test_ratio  = args.test,
    )

if __name__ == '__main__':
    main()
