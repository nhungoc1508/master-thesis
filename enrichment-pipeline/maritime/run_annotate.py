"""
PHASE 2: Annotate AIS positions using Archimedes C++ binary

Original repository: https://github.com/M3-Archimedes/AIS-trajectory-annotation

Requires the binary to be complied first

Usage examples:
    python run_annotate.py \
        data/ingested/aisdk-2025-01-01_for_annotation.txt \ # sorted .txt file
        data/ingested/aisdk-2025-01-01_vessel_info.csv      # vessel info .csv
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

from annotate.runner import run as run_annotate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('vessel_info')
    parser.add_argument('--output-dir', default='data/annotated')
    parser.add_argument('--config', default='config/pipeline_config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    txt_file = Path(args.input)
    vessel_info = Path(args.vessel_info)
    output_dir = Path(args.output_dir)

    if not txt_file.exists():
        logger.error('_for_annotation.txt not found at %s', txt_file)
        sys.exit(1)
    if not vessel_info.exists():
        logger.error('_vessel_info.csv not found at %s', vessel_info)
        sys.exit(1)

    logger.info('----- Annotating %s -----', txt_file.name)
    out = run_annotate(txt_file, vessel_info, output_dir, cfg)
    logger.info('\tAnnotation completed -> %s', out)

if __name__ == '__main__':
    main()