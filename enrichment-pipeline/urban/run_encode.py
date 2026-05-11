#!/usr/bin/env python3
"""
PHASE 4: Pre-compute semantic embeddings for per-point description

Implementation details:
    - Read described .parquet files (cols: trajectory_id, point_idx, description)
    - Encode every description using a frozen text encoder (current model used: baai/bge-m3)
    - Write 2 files:
        - data/encoded/[stem]_sem.npy           float16 memmap, shape (N, 1024)
        - data/encoded/[stem]_sem_meta.json     row count, shape, dtype, source path
    - Row order in .npy file corresponds to row order in the source .parquet

Usage examples:
    Encode one single file
        python run_encode.py data/described/porto_enriched_described.parquet

    Custom output directory
        python run_encode.py data/described/porto_enriched_described.parquet --output-dir path/to/dir/
    
    Encode all .parquet files under one directory
        python run_encode.py --all data/described/
    
    Specify a different model
        python run_encode.py data/described/porto_enriched_described.parquet --model BAAI/bge-large-en-v1.5
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from template.encoder import SemanticEncoder

_READ_CHUNK = 50_000

def encode_file(input_path: Path, output_dir: Path,
                encoder: SemanticEncoder, overwrite: bool = False) -> Path:
    """Encode all descriptions in one .parquet file and return path to .npy file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    npy_path = output_dir / f'{stem}_sem.npy'
    meta_path = output_dir / f'{stem}_sem_meta.json'

    if npy_path.exists() and not overwrite:
        logger.info('Skipping %s, already encoded at %s', input_path.name, npy_path)
        return npy_path
    
    if not input_path.exists():
        raise FileNotFoundError(f'Described .parquet not found: {input_path}')

    logger.info('Encoding %s', input_path.name)

    # ---------- First pass: count total rows ----------
    pf = pq.ParquetFile(input_path)
    n_rows = pf.metadata.num_rows
    dim = encoder.embed_dim
    logger.info('\t%d rows, output shape (%d, %d)',
                n_rows, n_rows, dim)
    
    # ---------- Allocate output memmap ----------
    out_array = np.lib.format.open_memmap(
        npy_path,
        mode='w+',
        dtype=np.float16,
        shape=(n_rows, dim)
    )

    # ---------- Second pass: encode in chunks ----------
    ts_start = time.time()
    cursor = 0
    for batch in pf.iter_batches(batch_size=_READ_CHUNK, columns=['description']):
        texts = batch.column('description').to_pylist()
        logger.info('\tProcessing rows %d - %d / %d', cursor, cursor + len(texts) - 1, n_rows)
        embs = encoder.encode(texts, show_progress=False)
        out_array[cursor : cursor + len(texts)] = embs
        cursor += len(texts)
    
    dt = time.time() - ts_start
    dt_str = time.strftime('%H:%M:%S', time.gmtime(dt))
    out_array.flush()

    # ---------- Write metadata ----------
    meta = {
        'source': str(input_path.resolve()),
        'model': encoder.model_name,
        'n_rows': n_rows,
        'shape': [n_rows, dim],
        'dtype': 'float16',
        'normalized': encoder.normalize,
        'time': dt
    }
    meta_path.write_text(json.dumps(meta, indent=4))

    logger.info('Done: %s (%d rows)', npy_path, n_rows)
    logger.info('Time taken: %s', dt_str)
    return npy_path

def main():
    parser = argparse.ArgumentParser()
    # Input can be single file OR -all [dir]
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('input', nargs='?')
    group.add_argument('--all', metavar='DIR')

    parser.add_argument('--output-dir', default='data/encoded/')
    parser.add_argument('--model', default='BAAI/bge-m3')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--device', default=None)
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    encoder = SemanticEncoder(
        model_name = args.model,
        device = args.device,
        batch_size = args.batch_size
    )

    if args.all:
        source_dir = Path(args.all)
        files = sorted(source_dir.glob('*_described.parquet'))
        if not files:
            logger.error('No *_described.parquet files found in %s', source_dir)
            return
        logger.info('Encoding %d files from %s', len(files), source_dir)
        for i, f in enumerate(files):
            logger.info('[%d/%d] %s', i+1, len(files), f.name)
            encode_file(f, output_dir, encoder, overwrite=args.overwrite)
    else:
        encode_file(Path(args.input), output_dir, encoder, overwrite=args.overwrite)

    logger.info('All done.')

if __name__ == '__main__':
    main()