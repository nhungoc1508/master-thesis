"""
STAGE 3a: Join annotated .csv back to the full AIS record

Archimedes' ./annotate binary strips all fields except MMSI/lon/lat/t
This stage recovers fields from the original .csv: SOG, COG, Heading,
ROT, Navigational status, Ship type, Draught, Length, Width, Destination,
Name, ETA: merging on (MMSI, ts_unix)

Input format: annotated, space-delimited .csv
    id lon lat t speed heading annotation
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Columns to recover from the full .parquet
_RECOVER_COLS = [
    'SOG', 'COG', 'Heading', 'ROT',
    'Navigational status', 'Ship type', 'Draught',
    'Length', 'Width', 'Destination', 'Name'
]

def run(annotated_csv: Path | str, full_ais_parquet: Path | str,
        output_dir: Path | str, cfg: dict) -> Path:
    """
    Join annotations to AIS fields and write the merged data to a .parquet
    Returns path to the merged .parquet
    """
    annotated_csv = Path(annotated_csv)
    full_ais_parquet = Path(full_ais_parquet)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = annotated_csv.stem.replace('_annotated', '')
    out_path = output_dir / f'{stem}_joined.parquet'
    ingest_cfg = cfg.get('ingest', {}).get('raw_columns', {})
    col_mmsi = ingest_cfg.get('mmsi', 'MMSI')

    # ========== Load annotated .csv ==========
    logger.info('Reading annotated .csv: %s', annotated_csv)
    anno = pd.read_csv(
        annotated_csv,
        sep=' ',
        names=['mmsi', 'lon', 'lat', 'ts_unix', 'computed_speed_kn', 'computed_heading_deg', 'annotation'],
        header=0,
        dtype={'mmsi': 'Int64', 'ts_unix': 'Int64', 'annotation': str}
    )
    # Drop NOISE points
    noise_mask = anno['annotation'].str.strip() == 'NOISE'
    n_noise = noise_mask.sum()
    if n_noise:
        logger.info('Dropping %d NOISE points', n_noise)
    anno = anno[~noise_mask].copy()
    anno['annotation'] = anno['annotation'].fillna('')

    # ========== Load full AIS .parquet file ==========
    logger.info('Reading full AIS .parquet file: %s', full_ais_parquet)
    full_schema = pq.read_schema(full_ais_parquet)
    available = set(full_schema.names)

    load_cols = [col_mmsi, 'ts_unix'] + [c for c in _RECOVER_COLS if c in available]
    logger.info('List of columns to load from full .parquet: %s', ', '.join(load_cols))
    full = pd.read_parquet(full_ais_parquet, columns=load_cols)
    full = full.rename(columns={col_mmsi: 'mmsi'})
    full['mmsi'] = full['mmsi'].astype('Int64')
    full['ts_unix'] = full['ts_unix'].astype('Int64')

    # ========== Merge on (mmsi, ts_unix) ==========
    logger.info('Merging %d annotated + %d AIS rows', len(anno), len(full))
    merged = anno.merge(full, on=['mmsi', 'ts_unix'], how='left')

    # Report unmatched rows
    n_unmatched = merged['computed_speed_kn'].isna().sum()
    if n_unmatched:
        logger.warning('%d annotated rows had no matching AIS record', n_unmatched)

    logger.info('Writing joined .parquet: %s (%d rows)', out_path, len(merged))
    tbl = pa.Table.from_pandas(merged, preserve_index=False)
    pq.write_table(tbl, out_path, compression='snappy')

    return out_path