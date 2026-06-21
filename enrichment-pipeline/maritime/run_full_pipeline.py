"""
Run full maritime pipeline for all AIS .csv files found in input directory

Usage examples:
    python run_full_pipeline.py /home/ubuntu/thesis/data/ais/raw

Input directory has structure:
    raw/
    ├── aisdk-2025-01-01
    │   └── aisdk-2025-01-01.csv
    ├── aisdk-2025-02-01
    │   └── aisdk-2025-02-01.csv
    ...

Processing order:
    0. [Out of scope] Download raw AIS data
    1. Data ingestion
        Input:
            - [stem].csv
        Output:
            - data/ingested/[stem]_full_ais.parquet
            - data/ingested/[stem]_for_annotation.txt
            - data/ingested/[stem]_vessel_info.csv
    2. Data annotation
        Input:
            - data/ingested/[stem]_for_annotation.txt
            - data/ingested/[stem]_vessel_info.csv
        Output:
            - data/annotated/[stem]_annotated.csv
    3. Data segmentation
        Input:
            - data/annotated/[stem]_annotated.csv
            - data/ingested/[stem]_full_ais.parquet
        Output:
            - data/segmented/joined/[stem]_joined.parquet
            - data/segmented/[stem]_segmented.parquet
            - (Optional, for inspection) data/segmented/[stem]_segmented_all.parquet
    4. Data enrichment
        Input:
            - data/segmented/[stem]_segmented.parquet
        Output:
            - data/enriched/[stem]_enriched.parquet
    5. Data canonicalization
        Input:
            - data/enriched/[stem]_enriched.parquet
        Output:
            - data/canonical/[stem]_canonical.parquet
    6. Data description
        Input:
            - data/canonical/[stem]_canonical.parquet
        Output:
            - data/described/[stem]_described.parquet
"""
import argparse
import logging
import re
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
from annotate.runner import run as run_annotate
from segment.join import run as run_join
from segment.trips import run as run_trips
from enrich.pipeline import run as run_enrich
from canonicalize.resample import run as run_canonicalize
from run_describe import describe_file

_DATE_PATTERN = re.compile(r'^aisdk-\d{4}-\d{2}-\d{2}$')

def _cleanup(paths: list) -> None:
    for p in paths:
        p = Path(p)
        if p.exists():
            p.unlink()
            logger.info('\tRemoved: %s', p)


def _process_one(csv_file: Path, base_dir: Path, cfg: dict) -> None:
    stem = csv_file.stem  # e.g. aisdk-2025-01-01

    ingested_dir  = base_dir / 'data' / 'ingested'
    annotated_dir = base_dir / 'data' / 'annotated'
    join_dir      = base_dir / 'data' / 'segmented' / 'joined'
    segmented_dir = base_dir / 'data' / 'segmented'
    enriched_dir  = base_dir / 'data' / 'enriched'
    canonical_dir = base_dir / 'data' / 'canonical'
    described_dir = base_dir / 'data' / 'described'

    # ----- Stage 1: Ingest -----
    logger.info('[1/6] Ingesting %s', csv_file.name)
    parquet_out, txt_out, vessel_info_out = run_ingest(csv_file, ingested_dir, cfg)

    # ----- Stage 2: Annotate -----
    logger.info('[2/6] Annotating %s', stem)
    annotated_csv = run_annotate(txt_out, vessel_info_out, annotated_dir, cfg)

    # ----- Stage 3: Segment -----
    logger.info('[3/6] Segmenting %s', stem)
    joined = run_join(annotated_csv, parquet_out, join_dir, cfg)
    segmented = run_trips(joined, segmented_dir, cfg)

    # ----- Stage 4: Enrich -----
    logger.info('[4/6] Enriching %s', stem)
    enriched = run_enrich(segmented, enriched_dir, cfg, None)

    # ----- Stage 5: Canonicalize -----
    logger.info('[5/6] Canonicalizing %s', stem)
    canonical = run_canonicalize(enriched, canonical_dir, cfg)

    # ----- Stage 6: Describe -----
    logger.info('[6/6] Describing %s', stem)
    describe_file(canonical, described_dir)

    # ----- Cleanup heavy intermediates -----
    logger.info('Cleaning up intermediates for %s', stem)
    _cleanup([
        parquet_out,       # data/ingested/[stem]_full_ais.parquet
        txt_out,           # data/ingested/[stem]_for_annotation.txt
        vessel_info_out,   # data/ingested/[stem]_vessel_info.csv
        annotated_csv,     # data/annotated/[stem]_annotated.csv
        joined,            # data/segmented/joined/[stem]_joined.parquet
        segmented,         # data/segmented/[stem]_segmented.parquet
        segmented_dir / f'{stem}_segmented_all.parquet',  # optional inspection file
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir', nargs='?', default=None)
    parser.add_argument('--input-csv', default=None)
    parser.add_argument('--base-dir', default=None)
    parser.add_argument('--config', default='config/pipeline_config.yaml')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if not args.raw_dir and not args.input_csv:
        parser.error('Provide either RAW_DIR (aisdk-[date] layout) or --input-csv PATH.')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).parent

    # ----- Single-file mode (bypasses aisdk-[date] discovery; e.g., NOAA US) -----
    if args.input_csv:
        csv_file = Path(args.input_csv)
        if not csv_file.is_file():
            logger.error('--input-csv not found: %s', csv_file)
            sys.exit(1)
        logger.info('Single-file mode: %s', csv_file)
        try:
            _process_one(csv_file, base_dir, cfg)
            logger.info('Completed %s', csv_file.stem)
        except Exception:
            logger.exception('Failed for %s', csv_file.stem)
            sys.exit(1)
        return

    # ----- Directory mode (DMA aisdk-[date] layout) -----
    raw_dir  = Path(args.raw_dir)

    subdirs = sorted(
        d for d in raw_dir.iterdir()
        if d.is_dir() and _DATE_PATTERN.match(d.name)
    )

    if not subdirs:
        logger.error('No aisdk-[date] subdirectories found in %s', raw_dir)
        sys.exit(1)

    logger.info('Found %d date directories to process', len(subdirs))

    described_dir = base_dir / 'data' / 'described'
    n_ok = n_skip = n_fail = 0

    for i, subdir in enumerate(subdirs, 1):
        stem = subdir.name

        csv_files = sorted(subdir.glob('*.csv'))
        if not csv_files:
            logger.warning('[%d/%d] No .csv in %s — skipping', i, len(subdirs), subdir)
            n_fail += 1
            continue
        if len(csv_files) > 1:
            logger.warning('[%d/%d] Multiple .csv in %s — using %s',
                           i, len(subdirs), subdir, csv_files[0].name)

        logger.info('========== [%d/%d] %s ==========', i, len(subdirs), stem)
        try:
            _process_one(csv_files[0], base_dir, cfg)
            n_ok += 1
            logger.info('Completed %s', stem)
        except Exception:
            logger.exception('Failed for %s — continuing', stem)
            n_fail += 1

    logger.info('Done. ok=%d  skipped=%d  failed=%d', n_ok, n_skip, n_fail)


if __name__ == '__main__':
    main()