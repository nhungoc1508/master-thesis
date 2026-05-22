"""
PHASE 4: Enrich a segment .parquet file

Usage examples:
    Enrich one .parquet file:
        python run_enrich.py data/segmented/aisdk_2023-10-01_segmented.parquet
    
    Run specific stages on all .parquet file in a directory:
        python run_enrich.py data/segmented/ --stages temporal kinematic spatial
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

from enrich.pipeline import run as run_enrich, _ALL_STAGES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output-dir', default='data/enriched/')
    parser.add_argument('--stages', nargs='+', choices=_ALL_STAGES, default=None)
    parser.add_argument('--config', default='config/pipeline_config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    files = sorted(input_path.glob('*_segmented.parquet')) if input_path.is_dir() else [input_path]
    if not files:
        logger.error('No *_segmented.parquet files found at %s', input_path)
        sys.exit(1)
    
    for parquet_file in files:
        logger.info('----- Enriching %s -----', parquet_file.name)
        out = run_enrich(parquet_file, output_dir, cfg, stages=args.stages)
        logger.info('\tEnrichment completed -> %s', out)

if __name__ == '__main__':
    main()