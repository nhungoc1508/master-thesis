"""
PHASE 1: Ingest and clean raw AIS .csv

Usage examples:
    Clean one .csv file:
        python run_ingest.py data/raw/aisdk_2026-01-01.csv

    Custom output directory:
        python run_ingest.py data/raw/aisdk_2026-01-01.csv --output-dir data/ingested/

    Clean all .csv files in directory:
        python run_ingest.py data/raw/
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from ingest.preprocess import run as run_ingest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output-dir', default='data/ingested/')
    parser.add_argument('--config', default='config/pipeline_config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    files = sorted(input_path.glob('*.csv')) if input_path.is_dir() else [input_path]
    if not files:
        logger.error('No .csv file found at %s', input_path)
        sys.exit(1)
    
    for csv_file in files:
        logger.info('----- Ingesting %s -----', csv_file.name)
        parquet_out, txt_out, vessel_info_out = run_ingest(csv_file, output_dir, cfg)
        logger.info('\tfull_ais -> %s', parquet_out)
        logger.info('\tannotate_in -> %s', txt_out)
        logger.info('\tvessel_info -> %s', vessel_info_out)

if __name__ == '__main__':
    main()