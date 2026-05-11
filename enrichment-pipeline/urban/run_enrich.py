#!/usr/bin/env python3
"""
PHASE 2: Enrich a canonical .parquet file

Usage examples:
    Full pipeline:
        python run_enrich.py data/canonical/porto.parquet

    Run specific stages
        python run_enrich.py data/canonical/porto.parquet --stages temporal behavioral

    Custom output directory and config
        python run_enrich.py data/canonical/porto.parquet --output-dir data/enriched/ --config config/enrichment_config.yaml
"""
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
    datefmt='%H:%M:%S'
)

from enrichment.pipeline import EnrichmentPipeline, ALL_STAGES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--stages', nargs='+', choices=ALL_STAGES, default=None, metavar='STAGE')
    parser.add_argument('--output-dir', default='data/enriched')
    parser.add_argument('--cache-dir', default='data')
    parser.add_argument('--config', default='config/enrichment_config.yaml')
    args = parser.parse_args()

    pipeline = EnrichmentPipeline.from_config(args.config, cache_root=args.cache_dir)
    input_path = Path(args.input)
    inputs = (sorted(input_path.glob('*.parquet'))
              if input_path.is_dir() else [input_path])
    
    for inp in inputs:
        out = pipeline.run(inp, output_dir=args.output_dir, stages=args.stages)

if __name__ == '__main__':
    main()