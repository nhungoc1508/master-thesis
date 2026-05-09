#!/usr/bin/env python3
"""
PHASE 1: Clean a raw mobility dataset and write canonical .parquet.

Usage examples:
    Clean whole dataset:
        python run_clean porto data/porto/train.csv data/canonical/porto.parquet

    Clean + sample 120K trajectories (grid-based):
        python run_clean.py porto data/porto/train.csv data/canonical/porto.parquet \
            --sample 120000
    
    Tune grid resolution (default 20x20):
        python run_clean.py porto data/porto/train.csv data/canonical/porto.parquet \
            --sample 120000 --grid-rows 25 --grid-cols 25

Output:
    Canonical long-format .parquet (one row per GPS points) with columns:
        trajectory_id, point_idx, lat, lon, timestamp, city, source, transport_mode
    Manifest .csv: {stem}.manifest.csv, one row per trajectory
        bbox midpoint, timestamps
"""
import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault('NUMEXPR_MAX_THREADS', '1')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
    datefmt='%H:%M:%S'
)

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cleaning import (
    PortoCleaner, PNEUMACleaner, TDriveCleaner, GeoLifeCleaner,
    grid_sample_ids
)

from cleaning.base import QualityConfig

logger = logging.getLogger(__name__)

CLEANERS = {
    'porto': PortoCleaner,
    'pneuma': PNEUMACleaner,
    'tdrive': TDriveCleaner,
    'geolife': GeoLifeCleaner
}

def _filter_parquet_by_ids(input_path: Path, output_path: Path, ids: set[str]) -> int:
    """
    Read input_path row-group by row-group, keep only rows whose trajectory_id
    is in `ids` and write to output_path
    Return total rows written
    """
    pf = pq.ParquetFile(input_path)
    schema = pf.schema_arrow
    writer = None
    total = 0

    for batch in pf.iter_batches():
        tbl = pa.Table.from_batches([batch], schema=schema)
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

def main():
    parser = argparse.ArgumentParser(description='Clean a raw trajectory dataset')
    parser.add_argument('source', choices=list(CLEANERS), help='Dataset name')
    parser.add_argument('input', help='Path to raw data file')
    parser.add_argument('output', help='Path to canonical parquet output')
    parser.add_argument('--city', default=None, help='City name override')
    
    # Quality filters
    parser.add_argument('--min-points', type=int, default=10)
    parser.add_argument('--min-duration', type=float, default=60.0)
    parser.add_argument('--max-speed', type=float, default=180.0)
    parser.add_argument('--min-displacement', type=float, default=100.0)

    # Grid-based sampling
    parser.add_argument('--sample', type=int, default=None, metavar='N')
    parser.add_argument('--grid-rows', type=int, default=20)
    parser.add_argument('--grid-cols', type=int, default=20)
    parser.add_argument('--grid-seed', type=int, default=42)

    args = parser.parse_args()

    cfg = QualityConfig(
        min_points         = args.min_points,
        min_duration_s     = args.min_duration,
        max_speed_kmh      = args.max_speed,
        min_displacement_m = args.min_displacement,
    )

    cleaner_cls = CLEANERS[args.source]
    cleaner = cleaner_cls(config=cfg)

    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = output_path.with_suffix('').with_suffix('.manifest.csv')

    # ---------- Stage 1a: Clean (streaming write + manifest) ----------
    n_points = cleaner.run(input_path, output_path)
    logger.info('Cleaning done: %d points written to %s', n_points, output_path)

    # ---------- Stage 1b: Grid sampling (optional) ----------
    if args.sample is not None:
        logger.info('Reading manifest for grid sampling')
        manifest = pd.read_csv(manifest_path)

        n_before = len(manifest)
        sampled_ids = grid_sample_ids(
            manifest, n=args.sample,
            grid_rows=args.grid_rows, grid_cols=args.grid_cols, seed=args.grid_seed,
        )
        n_after = len(sampled_ids)

        if n_after < n_before:
            tmp_path = output_path.with_suffix('.tmp.parquet')
            rows_kept = _filter_parquet_by_ids(output_path, tmp_path, sampled_ids)
            tmp_path.replace(output_path)
            manifest = manifest[manifest['trajectory_id'].isin(sampled_ids)]
            manifest.to_csv(manifest_path, index=False)

            logger.info(
                'Grid sampling done: %d -> %d trajectories (%d points) -> %s',
                n_before, n_after, rows_kept, output_path,
            )
    
    n_rows = pq.read_metadata(output_path).num_rows
    print(f'Done: {output_path} ({n_rows} rows)')

if __name__ == '__main__':
    main()