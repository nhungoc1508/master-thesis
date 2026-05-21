"""
STAGE 3b: Trip segmentation from annotation labels

Trip definition: contiguous subsequence of a vessel's trajectory between two STOP events
    or between a STOP and a communitation GAP or the start/end of the record

Segmentation logic:
    - Begin a new trip segment when:
        - Processing the first point of a vessel
        - Encountering a STOP_END label
        - Encounter a GAP_END label
    - End the current trip segment when:
        - Encountering a STOP_START label
        - Encountering a GAP_START label
        - Reaching the last point of a vessel

Each point is assigned trajectory_id & point_idx

Quality gates: drop trips that fail minimum thresholds:
    min_points              20
    min_duration_s          300 (5 minutes)
    min_displacement_nm     0.5 (nautical miles)

Output .parquet adds columns: trajectory_id, point_idx, source, city, transport_mode
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from haversine import haversine, Unit

logger = logging.getLogger(__name__)

_SOURCE = 'aisdk'

_SHIP_TYPE_MAP: dict[tuple[int, int], str] = {
    (70, 79): 'cargo',
    (80, 89): 'tanker',
    (60, 69): 'passenger',
    (30, 30): 'fishing',
    (32, 33): 'fishing',
    (21, 22): 'tug',
    (31, 31): 'tug',
    (36, 37): 'pleasure_craft',
    (50, 59): 'special_craft',
}

_TEXT_TO_MODE: dict[str, str] = {
    'fishing':              'fishing',
    'cargo':                'cargo',
    'tanker':               'tanker',
    'passenger':            'passenger',
    'tug':                  'tug',
    'towing':               'tug',
    'towing long/wide':     'tug',
    'port tender':          'tug',
    'pleasure':             'pleasure_craft',
    'sailing':              'pleasure_craft',
    'hsc':                  'special_craft',
    'pilot':                'special_craft',
    'sar':                  'special_craft',
    'dredging':             'special_craft',
    'law enforcement':      'special_craft',
    'military':             'special_craft',
    'diving':               'special_craft',
    'anti-pollution':       'special_craft',
    'medical':              'special_craft',
    'other':                'unknown',
    'undefined':            'unknown',
    'reserved':             'unknown',
    'spare 1':              'unknown',
    'spare 2':              'unknown',
    'not party to conflict':'unknown',
}

def _ship_type_to_mode(type_code) -> str:
    if type_code is None:
        return 'unknown'
    text_key = str(type_code).strip().lower()
    if text_key in _TEXT_TO_MODE:
        return _TEXT_TO_MODE[text_key]
    try:
        code = int(type_code)
    except (TypeError, ValueError):
        return 'unknown'
    for (lo, hi), mode in _SHIP_TYPE_MAP.items():
        if lo <= code <= hi:
            return mode
    return 'unknown'

def _displacement_nm(lats: list[float], lons: list[float]) -> float:
    if len(lats) < 2:
        return 0.0
    return haversine((lats[0], lons[0]), (lats[-1], lons[-1]), unit=Unit.NAUTICAL_MILES)

def _segment_vessel(df: pd.DataFrame, min_gap_s: int) -> pd.Series:
    """Assign trip_idx to each row of a single-vessel DataFrame. df must be sorted by ts_unix asc"""
    trip_idx = np.zeros(len(df), dtype=np.int32)
    idx = 0
    prev_ts = None

    annotations = df['annotation'].tolist()
    timestamps = df['ts_unix'].tolist()

    for i, (anno, ts) in enumerate(zip(annotations, timestamps)):
        labels = set(anno.split(';')) if anno.strip() else set()
        if prev_ts is not None and (ts - prev_ts) >= min_gap_s:
            idx += 1
        if i > 0 and labels & {'STOP_END', 'GAP_END'}:
            idx += 1
        trip_idx[i] = idx
        prev_ts = ts
        if labels & {'STOP_START', 'GAP_START'}:
            idx += 1
    
    return pd.Series(trip_idx, index=df.index, dtype=np.int32)

def run(joined_parquet: Path | str, output_dir: Path | str,
        cfg: dict, keep_all: bool = False) -> Path:
    """
    Segment trips in the joined parquet and apply quality gates
    Returns path to the segmented .parquet
    If keep_all=True, also writes [stem]_segmented_all.parquet with all trips before quality filtering
    """
    joined_parquet = Path(joined_parquet)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = joined_parquet.stem.replace('_joined', '')
    out_path = output_dir / f'{stem}_segmented.parquet'
    all_path = output_dir / f'{stem}_segmented_all.parquet'

    seg_cfg = cfg.get('segment', {})
    min_gap = int(seg_cfg.get('min_gap_size', 3600))
    min_pts = int(seg_cfg.get('min_points', 20))
    min_dur = float(seg_cfg.get('min_duration_s', 300))
    min_disp = float(seg_cfg.get('min_displacement_nm', 0.5))

    logger.info('Loading joined .parquet: %s', joined_parquet)
    df = pd.read_parquet(joined_parquet)
    df = df.sort_values(['mmsi', 'ts_unix']).reset_index(drop=True)

    logger.info('Segmenting %d points accross %d vessels', len(df), df['mmsi'].nunique())

    # ========== Assign trip_idx per vessel ==========
    trip_idx_series = df.groupby('mmsi', sort=False, group_keys=False).apply(
        lambda g: _segment_vessel(g.sort_values('ts_unix'), min_gap),
        include_groups=False
    )
    df['_trip_idx'] = trip_idx_series.values

    # ========== Build trajectory_id ==========
    df['trajectory_id'] = (
        'aisdk_'
        + df['mmsi'].astype(str) + '_'
        + df['_trip_idx'].astype(str).str.zfill(4)
    )

    # ========== (Optional) save pre-filtering snapshot ==========
    if keep_all:
        df_all = df.copy()
        df_all['point_idx'] = df_all.groupby('trajectory_id').cumcount()
        df_all['source'] = _SOURCE
        type_col_all = 'Ship type' if 'Ship type' in df_all.columns else None
        if type_col_all:
            df_all['transport_mode'] = df_all[type_col_all].apply(_ship_type_to_mode)
        else:
            df_all['transport_mode'] = 'unknown'
        df_all['city'] = 'open_sea'
        df_all = df_all.drop(columns=['_trip_idx'])
        df_all = df_all.sort_values(['trajectory_id', 'point_idx']).reset_index(drop=True)
        logger.info('Writing pre-filtering .parquet: %s (%d points, %d trips)',
                    all_path, len(df_all), df_all['trajectory_id'].nunique())
        tbl_all = pa.Table.from_pandas(df_all, preserve_index=False)
        pq.write_table(tbl_all, all_path, compression='snappy')

    # ========== Quality filtering ==========
    def _passes_gates(grp: pd.DataFrame) -> bool:
        if len(grp) < min_pts:
            return False
        duration = grp['ts_unix'].iloc[-1] - grp['ts_unix'].iloc[0]
        if duration < min_dur:
            return False
        if _displacement_nm(grp['lat'].tolist(), grp['lon'].tolist()) < min_disp:
            return False
        return True
    
    keep_ids = {tid for tid, grp in df.groupby('trajectory_id', sort=False) if _passes_gates(grp)}
    n_before = df['trajectory_id'].nunique()
    df = df[df['trajectory_id'].isin(keep_ids)].copy()
    logger.info('Quality gates: %d/%d trips kept', len(keep_ids), n_before)

    # ========== Add canonical fields ==========
    df['point_idx'] = df.groupby('trajectory_id').cumcount()
    df['source'] = _SOURCE
    type_col = 'Ship type' if 'Ship type' in df.columns else None
    if type_col:
        df['transport_mode'] = df[type_col].apply(_ship_type_to_mode)
    else:
        df['transport_mode'] = 'unknown'

    # City = placeholder, sea_area_name will be enriched later
    df['city'] = 'open_sea'

    df = df.drop(columns=['_trip_idx'])
    df = df.sort_values(['trajectory_id', 'point_idx']).reset_index(drop=True)

    logger.info('Writing segmented .parquet: %s (%d points, %d trips)',
                out_path, len(df), df['trajectory_id'].nunique())
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(tbl, out_path, compression='snappy')

    return out_path