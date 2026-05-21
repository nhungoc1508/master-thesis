"""
STAGE 3: Join annotated output (from Stage 2) with full .parquet (from Stafe 1),
    then segment into trips (per-MMSI)

Usage examples:
    Segment using one annotation .csv + one full .parquet
        python run_segment.py \
            data/annotated/aisdk-2025-01-01_annotated.csv \
            data/ingested/aisdk-2025-01-01_full_ais.parquet
    
    Run and save additional pre-filtered .parquet:
        python run_segment.py \
            data/annotated/aisdk-2025-01-01_annotated.csv \
            data/ingested/aisdk-2025-01-01_full_ais.parquet --keep-all
    
    Segment using all *_annotated.csv and *_full_ais.parquet
        python run_segment.py data/annotated data/ingested
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

from segment.join import run as run_join
from segment.trips import run as run_trips

def _find_pairs(anno_dir: Path, ais_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for anno_csv in sorted(anno_dir.glob('*_annotated.csv')):
        stem = anno_csv.stem.replace('_annotated', '')
        ais_parquet = ais_dir / f'{stem}_full_ais.parquet'
        if not ais_parquet.exists():
            logger.warning('No matching _full_ais.parquet for %s; skipping', anno_csv.name)
            continue
        pairs.append((anno_csv, ais_parquet))
    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('annotated')
    parser.add_argument('full_ais')
    parser.add_argument('--join-dir', default='data/segmented/joined/')
    parser.add_argument('--output-dir', default='data/segmented/')
    parser.add_argument('--config', default='config/pipeline_config.yaml')
    parser.add_argument('--keep-all', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    anno_path = Path(args.annotated)
    ais_path = Path(args.full_ais)

    if anno_path.is_dir() and ais_path.is_dir():
        pairs = _find_pairs(anno_path, ais_path)
    else:
        pairs = [(anno_path, ais_path)]

    if not pairs:
        logger.error('No matching annotation/AIS file pairs found')
        sys.exit(1)

    for anno_csv, ais_parquet in pairs:
        logger.info('----- Segmenting %s -----', anno_csv.stem)
        joined = run_join(anno_csv, ais_parquet, Path(args.join_dir), cfg)
        segmented = run_trips(joined, Path(args.output_dir), cfg, keep_all=args.keep_all)
        logger.info('\tSegmentation completed -> %s', segmented)

if __name__ == '__main__':
    main()