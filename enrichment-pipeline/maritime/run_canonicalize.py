"""
STAGE 5 - Resample enriched points to the canonical interval (5 min)

Usage examples:
    Canonicalize a specific enriched .parquet:
        python run_canonicalize.py data/enriched/aisdk_2023-10-01_enriched.parquet

    Use custom interval:
        python run_canonicalize.py data/enriched/aisdk_2023-10-01_enriched.parquet --interval 300
    
    Use custom output directory:
        python run_canonicalize.py data/enriched/aisdk_2023-10-01_enriched.parquet --output-dir data/canonical/
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

from canonicalize.resample import run as run_canonicalize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output-dir', default='data/canonical/')
    parser.add_argument('--interval', type=int, default=None)
    parser.add_argument('--config', default='config/pipeline_config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.interval is not None:
        cfg.setdefault('canonicalize', {})['interval_s'] = args.interval

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    files = sorted(input_path.glob('*_enriched.parquet')) if input_path.is_dir() else [input_path]
    if not files:
        logger.error('No *_enriched.parquet files found at %s', input_path)
        sys.exit(1)
    
    for pq_file in files:
        logger.info('----- Canonicalizing %s -----', pq_file.name)
        out = run_canonicalize(pq_file, output_dir, cfg)
        logger.info('\tCanonicalizing completed -> %s', out)

if __name__ == '__main__':
    main()