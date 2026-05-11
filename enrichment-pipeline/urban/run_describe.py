#!/usr/bin/env python3
"""
PHASE 3: Generate natural-language description for point-level enrichment
    Currently uses simple templates

Usage examples:
    Full pipeline with default output (data/described/)
        python run_describe.py data/enriched/porto_enriched.parquet

    Custom output directory
        python run_describe.py data/enriched/porto_enriched.parquet --output-dir path/to/dir/
"""
import argparse
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from template.descriptor import generate_point_descriptor

def describe(input_path: Path, output_dir: Path) -> Path:
    """Generate natural-language descriptions for every GPS point in an enriched .parquet"""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f'Enriched data not found: {input_path}')

    output_path = output_dir / f'{input_path.stem}_described.parquet'
    logger.info('Describing %s -> %s', input_path.name, output_path)

    pf = pq.ParquetFile(input_path)
    writer = None
    total_rows = 0

    for batch in pf.iter_batches():
        df = pa.Table.from_batches([batch]).to_pandas()
        df['description'] = [
            generate_point_descriptor(row.to_dict()) for _, row in df.iterrows()
        ]
        out_tbl = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, out_tbl.schema)
        writer.write_table(out_tbl)
        total_rows += len(df)

    if writer:
        writer.close()

    logger.info('Done: saved %d rows -> %s', total_rows, output_path)
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output-dir', default='data/described/')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    describe(input_path, output_dir)

    logger.info('All done.')

if __name__ == '__main__':
    main()