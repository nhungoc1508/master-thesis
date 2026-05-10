#!/usr/bin/env python3
"""
Inspect canonical/enriched .parquet files and report:
    - Number of GPS points
    - Number of trajectories

Usage:
    python scripts/inspect_parquet_stats.py --dir canonical
    python scripts/inspect_parquet_stats.py --dir enriched

Output:
    data/logs/stats_[canonical|enriched]_[mmdd]_[hhmm].json
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DIR_MAP = {
    'canonical': _PROJECT_ROOT / 'data' / 'canonical',
    'enriched': _PROJECT_ROOT / 'data' / 'enriched',
}

def _run_duckdb(query: str) -> list[dict]:
    result = subprocess.run(
        ['duckdb', '-json', '-c', query],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)

def _stats_for_file(path: Path) -> dict:
    query = f"SELECT COUNT(*) AS n_points, COUNT(DISTINCT trajectory_id) AS n_trajectories FROM read_parquet('{path}')"
    rows = _run_duckdb(query)
    row = rows[0]
    return {
        'file':           path.name,
        'n_points':       int(row['n_points']),
        'n_trajectories': int(row['n_trajectories'])
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', choices=['canonical', 'enriched'], required=True)
    args = parser.parse_args()
    data_dir = _DIR_MAP[args.dir]
    if not data_dir.exists():
        sys.exit(f'Directory not found: {data_dir}')
    
    parquet_files = sorted(data_dir.glob('*.parquet'))
    if not parquet_files:
        sys.exit(f'No .parquet files found in {data_dir}')
    
    logger.info('Inspecting %d file(s) in %s', len(parquet_files), data_dir)

    results = []
    total_points = 0
    total_trajectories = 0

    for pf in parquet_files:
        logger.info('\t%s', pf.name)
        try:
            stats = _stats_for_file(pf)
            results.append(stats)
            total_points += stats['n_points']
            total_trajectories += stats['n_trajectories']
            logger.info('\t\t%d points, %d trajectories', stats['n_points'], stats['n_trajectories'])
        except Exception as exc:
            logger.error('\t\tFAILED: %s', exc)
            results.append({'file': pf.name, 'error': str(exc)})

    output = {
        'dir':                args.dir,
        'generated_at':       datetime.now().isoformat(timespec='seconds'),
        'total_points':       total_points,
        'total_trajectories': total_trajectories,
        'datasets':           results
    }

    logs_dir = _PROJECT_ROOT / 'data' / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%m%d_%H%M')
    out_path = logs_dir / f'stats_{args.dir}_{ts}.json'
    out_path.write_text(json.dumps(output, indent=4))

    logger.info('Written to %s', out_path)
    print(f'\nTotal: {total_points:,} points across {total_trajectories:,} trajectories')
    print(f'Output: {out_path}')

if __name__ == '__main__':
    main()