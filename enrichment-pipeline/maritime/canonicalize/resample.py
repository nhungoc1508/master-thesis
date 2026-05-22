"""
STAGE 5 - Resample enriched points to 5-minute interval

Implementation details:
    - For each (trajectory_id, ts_bucket) pair, keep the point whose
        ts_unix is closest to the bucket boundary
    - ts_bucket = FLOOR(ts_unix / interval_s)
    - After resampling, point_idx is recomputed within each trajectory
    - Short trips that reduce to < 2 points are dropped
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_KEEP_COLS = [
    # Canonical
    'trajectory_id', 'point_idx', 'lat', 'lon', 'ts_unix', 'city', 'source', 'transport_mode',
    # AIS kinematic
    'SOG', 'COG', 'Heading', 'ROT', 'Navigational status',
    # AIS static
    'Ship type', 'Draught', 'Name', 'Destination',
    # Annotation
    'annotation', 'computed_speed_kn', 'computed_heading_deg',
    # Kinematic
    'behavioral_phase', 'heading_to_cog_diff_deg',
    # Temporal
    'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'season', 'time_of_day_category',
    # Spatial
    'nearest_port_nm', 'nearest_port_name', 'port_proximity_label', 'in_port_zone',
    'in_mpa', 'mpa_name', 'in_tss', 'tss_name', 'sea_area_name', 'eez_country_iso', 'in_territorial_sea',
    # Bathymetry
    'water_depth_m',
    # Ocean
    'wave_height_m', 'current_speed_ms', 'current_dir_deg',
    # Geohash
    'geohash_5', 'geohash_7'
]

def run(enriched_parquet: Path | str, output_dir: Path | str,
        cfg: dict) -> Path:
    """
    Resample enriched points to canonical interval and write canonical .parquet
    Returns path to the canonical .parquet
    """
    enriched_parquet = Path(enriched_parquet)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = enriched_parquet.stem.replace('_enriched', '')
    out_path = output_dir / f'{stem}_canonical.parquet'
    
    interval_s = int(cfg.get('canonicalize', {}).get('interval_s', 300))

    logger.info('Loading enriched .parquet: %s', enriched_parquet)
    df = pd.read_parquet(enriched_parquet)
    n_before = len(df)
    trips_before = df['trajectory_id'].nunique()

    # ========== Floor-bucket resampling ==========
    df['_ts_bucket'] = (df['ts_unix'] // interval_s).astype(np.int64)
    # Within each (trajectory_id, bucket), keep the row closest to bucket start
    df['_bucket_start'] = df['_ts_bucket'] * interval_s
    df['_dist'] = (df['ts_unix'] - df['_bucket_start']).abs()

    df = df.sort_values('_dist').groupby(['trajectory_id', '_ts_bucket'], sort=False).first().reset_index()

    # Recompute point_idx after resampling
    df = df.sort_values(['trajectory_id', 'ts_unix'])
    df['point_idx'] = df.groupby('trajectory_id').cumcount()

    # Drop trips < 2 points after resampling
    trip_lengths = df.groupby('trajectory_id')['point_idx'].max()
    keep = trip_lengths[trip_lengths >= 1].index
    df = df[df['trajectory_id'].isin(keep)].copy()

    df = df.drop(columns=['_ts_bucket', '_bucket_start', '_dist'])

    # Select canonical columns that exist in the df
    cols_out = [c for c in _KEEP_COLS if c in df.columns]
    logger.debug('Canonical columns that exist in the df: %s', cols_out)
    df = df[cols_out].reset_index(drop=True)

    logger.info('Canonicalized: %d -> %d points (%d -> %d trips, interval=%ds)',
                n_before, len(df), trips_before, df['trajectory_id'].nunique(), interval_s)
    
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(tbl, out_path, compression='snappy')
    logger.info('Canonical .parquet written: %s', out_path)

    return out_path