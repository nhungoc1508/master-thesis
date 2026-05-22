"""
STAGE 6 - Generate natural-language description for point-level enrichment

Usage examples:
    Describe a specific canonical .parquet:
        python run_describe.py data/canonical/aisdk_2023-10-01_canonical.parquet

    Describe all files in a directory:
        python run_describe.py data/canonical/
    
    Use custom output directory:
        python run_describe.py data/canonical/ --output-dir data/described/
"""
import argparse
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

from template.descriptor import generate_point_descriptor

def describe_file(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'{input_path.stem}_described.parquet'
    
    pf = pq.ParquetFile(input_path)
    writer = None
    total_rows = 0

    for batch in pf.iter_batches():
        df = pa.Table.from_batches([batch]).to_pandas()
        df['description'] = [generate_point_descriptor(r.to_dict()) for _, r in df.iterrows()]
        df = df[['trajectory_id', 'point_idx', 'description']]
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, tbl.schema)
        writer.write_table(tbl)
        total_rows += len(df)
    
    if writer:
        writer.close()

    logger.info('Described %d points -> %s', total_rows, out_path)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output-dir', default='data/described/')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    files = sorted(input_path.glob('*_canonical.parquet')) if input_path.is_dir() else [input_path]
    if not files:
        logger.error('No *_canonical.parquet files found at %s', input_path)
        sys.exit(1)
    
    for pq_file in files:
        logger.info('----- Describing %s -----', pq_file.name)
        out = describe_file(pq_file, output_dir)
        logger.info('\t-> %s', out)

if __name__ == '__main__':
    main()