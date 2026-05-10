#!/usr/bin/env python3
"""
Inspect canonical .parquet files and report:
    - Month & year pairs
    - Count of trajectories within each pair
    - Sorted in descending order

Usage:
    python scripts/inspect_parquet_months.py

Output: one .json file for each .parquet file
    data/logs/months_[mmdd]_[HHMM]/[dataset]_month.json
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
_CANONICAL_DIR = _PROJECT_ROOT / 'data' / 'canonical'

def _run_duckdb(query: str) -> list[dict]:
    result = subprocess.run(
        ['duckdb', '-json', '-c', query],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)

def _stats_for_file(path: Path) -> list[dict]:
    query = f"""
WITH monthYear AS (
    SELECT
        trajectory_id,
        extract('year' from to_timestamp(timestamp)) y,
        extract('month' from to_timestamp(timestamp)) m
    FROM read_parquet('{path}')
)
SELECT y, m, COUNT(DISTINCT trajectory_id) freq
FROM monthYear GROUP BY y, m ORDER BY freq DESC
"""
    rows = _run_duckdb(query)
    return [{'year': int(r['y']), 'month': int(r['m']), 'freq': int(r['freq'])} for r in rows]

def main():
    data_dir = _CANONICAL_DIR
    if not data_dir.exists():
        sys.exit(f'Directory not found: {data_dir}')
    
    parquet_files = sorted(data_dir.glob('*.parquet'))
    if not parquet_files:
        sys.exit(f'No .parquet files found in {data_dir}')
    
    logger.info('Inspecting %d file(s) in %s', len(parquet_files), data_dir)

    ts = datetime.now().strftime('%m%d_%H%M')
    logs_dir = _PROJECT_ROOT / 'data' / 'logs' / f'months_{ts}'
    logs_dir.mkdir(parents=True, exist_ok=True)

    for pf in parquet_files:
        logger.info('\t%s', pf.name)
        results = []
        json_name = pf.name.split('.')[0]
        out_path = logs_dir / f'{json_name}.json'
        try:
            results = _stats_for_file(pf)
        except Exception as exc:
            logger.error('\t\tFAILED: %s', exc)
            results.append({'file': pf.name, 'error': str(exc)})
        output = {
            'file':    pf.name,
            'results': results
        }
        out_path.write_text(json.dumps(output, indent=4))

        logger.info('\t\tWritten to %s', out_path)

if __name__ == '__main__':
    main()