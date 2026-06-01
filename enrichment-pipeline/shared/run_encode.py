#!/usr/bin/env python3
"""
PHASE 4: Pre-compute semantic embeddings for per-point description

Implementation details:
    - Read described .parquet files (cols: trajectory_id, point_idx, description)
    - Encode every description using a frozen text encoder (current model used: google/embeddinggemma-300m)
    - Write 2 files:
        - data/encoded/[stem]_sem.npy           float16 memmap, shape (N, embed_dim)
        - data/encoded/[stem]_sem_meta.json     row count, shape, dtype, source path
    - Row order in .npy file corresponds to row order in the source .parquet

Resume support:
    If a partial .npy + sibling .progress file exist (from a previously interrupted
    run), encoding resumes from the saved cursor. The .progress file is deleted once
    encoding completes successfully.

HuggingFace backup:
    Pass --hf-repo [repo-id] to upload the .npy file to a HuggingFace dataset repo
    every --upload-every parquet chunks (default: 10). A final upload is always
    performed on completion.

Usage examples:
    Encode one file with HF backup every 10 chunks
        python run_encode.py data/described/porto_enriched_described.parquet \\
            --trust-remote-code --truncate-dim 256 \\
            --hf-repo nhungoc1508/ms-thesis

    Resume an interrupted run
        python run_encode.py data/described/porto_enriched_described.parquet \\
            --trust-remote-code --truncate-dim 256 --resume

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

import gc
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from encoder import SemanticEncoder

_READ_CHUNK = 20_000

# ----- Checkpoint helpers -----

def _progress_path(npy_path: Path) -> Path:
    return npy_path.with_suffix('.progress')

def _load_cursor(npy_path: Path) -> int:
    """Return saved cursor from .progress file, or 0 if absent."""
    p = _progress_path(npy_path)
    if p.exists():
        try:
            return int(p.read_text().strip())
        except (ValueError, IOError):
            return 0
    return 0

def _save_cursor(npy_path: Path, cursor: int) -> None:
    _progress_path(npy_path).write_text(str(cursor))

def _delete_cursor(npy_path: Path) -> None:
    p = _progress_path(npy_path)
    if p.exists():
        p.unlink()

# ----- HuggingFace helpers -----

def _hf_upload(npy_path: Path, repo_id: str, include_progress: bool = True) -> None:
    """
    Upload npy_path (+ .progress file) to a HuggingFace dataset repo.
    Overwrites any existing file at the same path.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()

        logger.info('Uploading %s to hf://%s ...', npy_path.name, repo_id)
        api.upload_file(
            path_or_fileobj=str(npy_path),
            path_in_repo=npy_path.name,
            repo_id=repo_id,
            repo_type='dataset',
        )

        prog = _progress_path(npy_path)
        if include_progress and prog.exists():
            api.upload_file(
                path_or_fileobj=str(prog),
                path_in_repo=prog.name,
                repo_id=repo_id,
                repo_type='dataset',
            )

        logger.info('Upload complete.')
    except Exception as exc:
        logger.warning('HuggingFace upload failed (will retry next interval): %s', exc)

# --- Core encoding -----

def encode_file(
    input_path: Path,
    output_dir: Path,
    encoder: SemanticEncoder,
    overwrite: bool = False,
    resume: bool = True,
    hf_repo: str | None = None,
    upload_every: int = 10,
) -> Path:
    """Encode all descriptions in one .parquet file and return path to .npy file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    npy_path  = output_dir / f'{stem}_sem.npy'
    meta_path = output_dir / f'{stem}_sem_meta.json'

    # Decide whether to resume, skip, or start fresh
    has_npy      = npy_path.exists()
    has_progress = _progress_path(npy_path).exists()

    if has_npy and not has_progress and not overwrite:
        logger.info('Skipping %s - already complete at %s', input_path.name, npy_path)
        return npy_path

    if not input_path.exists():
        raise FileNotFoundError(f'Described .parquet not found: {input_path}')

    # Count rows and prepare memmap
    pf     = pq.ParquetFile(input_path)
    n_rows = pf.metadata.num_rows
    dim    = encoder.embed_dim
    dtype  = np.float16
    size_gb = n_rows * dim * np.dtype(dtype).itemsize / 1e9
    logger.info('Encoding %s  (%d rows, shape %d×%d, %.2f GB)',
                input_path.name, n_rows, n_rows, dim, size_gb)

    cursor      = 0
    memmap_mode = 'w+'

    if resume and has_npy and has_progress:
        saved = _load_cursor(npy_path)
        if 0 < saved < n_rows:
            cursor      = saved
            memmap_mode = 'r+'
            logger.info('Resuming from row %d / %d (%.1f%%)',
                        cursor, n_rows, 100 * cursor / n_rows)
        else:
            logger.info('Progress file stale (cursor=%d), starting fresh', saved)

    out_array = np.lib.format.open_memmap(
        npy_path,
        mode=memmap_mode,
        dtype=dtype,
        shape=(n_rows, dim),
    )

    # Encode in chunks
    ts_start    = time.time()
    chunk_index = 0  # counts chunks processed in this run (not total)
    row_pos     = 0  # tracks absolute row position

    for batch in pf.iter_batches(batch_size=_READ_CHUNK, columns=['description']):
        batch_len = len(batch)

        # Skip batches already written in a previous run
        if row_pos + batch_len <= cursor:
            row_pos += batch_len
            continue

        texts = batch.column('description').to_pylist()
        logger.info('\tRows %d – %d / %d', cursor, cursor + len(texts) - 1, n_rows)

        embs = encoder.encode(texts, show_progress=False)
        out_array[cursor : cursor + len(texts)] = embs
        out_array.flush()
        cursor += len(texts)

        row_pos += len(texts)
        del texts, embs
        gc.collect()
        torch.cuda.empty_cache()

        _save_cursor(npy_path, cursor)
        chunk_index += 1

        # Periodic HF upload
        if hf_repo and chunk_index % upload_every == 0:
            _hf_upload(npy_path, hf_repo)

    # Finalise
    dt     = time.time() - ts_start
    dt_str = time.strftime('%H:%M:%S', time.gmtime(dt))

    meta = {
        'source':       str(input_path.resolve()),
        'model':        encoder.model_name,
        'n_rows':       n_rows,
        'shape':        [n_rows, dim],
        'dtype':        'float16',
        'normalized':   encoder.normalize,
        'truncate_dim': encoder.truncate_dim,
        'size_gb':      round(size_gb, 3),
        'time':         dt,
    }
    meta_path.write_text(json.dumps(meta, indent=4))
    _delete_cursor(npy_path)

    logger.info('Done: %s (%d rows)  time: %s', npy_path, n_rows, dt_str)

    # Final upload
    if hf_repo:
        _hf_upload(npy_path, hf_repo)

    return npy_path

def main():
    parser = argparse.ArgumentParser()
    # Input can be single file OR --all [dir]
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('input', nargs='?')
    group.add_argument('--all', metavar='DIR')

    parser.add_argument('--output-dir', default='data/encoded/')
    parser.add_argument('--model', default='google/embeddinggemma-300m')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--device', default=None)
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from a saved .progress file if present (default: on)')
    parser.add_argument('--no-resume', action='store_false', dest='resume',
                        help='Force fresh encoding even if a .progress file exists')
    parser.add_argument('--truncate-dim', type=int, default=256, metavar='N')
    parser.add_argument('--attn-impl', default=None,
                        choices=['sdpa', 'flash_attention_2'],
                        help='Attention implementation. '
                             'sdpa: PyTorch 2.0+ native, no install needed. '
                             'flash_attention_2: fastest on H100/A100, '
                             'requires: pip install flash-attn --no-build-isolation')
    parser.add_argument('--hf-repo', default=None, metavar='REPO_ID')
    parser.add_argument('--upload-every', type=int, default=10, metavar='N')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    encoder = SemanticEncoder(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
        truncate_dim=args.truncate_dim,
        attn_implementation=args.attn_impl,
    )

    kwargs = dict(
        overwrite=args.overwrite,
        resume=args.resume,
        hf_repo=args.hf_repo,
        upload_every=args.upload_every,
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
            encode_file(f, output_dir, encoder, **kwargs)
    else:
        encode_file(Path(args.input), output_dir, encoder, **kwargs)

    logger.info('All done.')

if __name__ == '__main__':
    main()