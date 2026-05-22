"""
Enrichment pipeline central orchestrator

Reads a segmented .parquet file (one row per GPS point) then runs enrichment stages
Writes a checkpoint after each stage so the pipeline can be resumed

Stages:
    temporal        temporal features derived from timestamp
    kinematic       derived from original AIS features (SOG, Heading, COG) and annotation labe;s
    spatial         World Port Index, OSM shapefiles (habors, MPA, TSS), VLIZ EEZ, VLIZ IHO sea areas
    bathymetry      GEBCO NetCDF
    ocean           Copernicus CMEMS API
    geohash         computed
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.archimedes import setup_path, load_context
from enrich.spatial import SpatialEnricher

logger = logging.getLogger(__name__)

_ALL_STAGES = ['spatial']

def _checkpoint_path(output_dir: Path, stem: str, stage: str) -> Path:
    return output_dir / f'{stem}_checkpoint_{stage}.parquet'

def _load_checkpoint(path: Path) -> pd.DataFrame | None:
    if path.exists():
        logger.info('Resuming from checkpoint: %s', path.name)
        return pd.read_parquet(path)
    return None

def _save_checkpoint(df: pd.DataFrame, path: Path):
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(tbl, path, compression='snappy')
    logger.info('Checkpoint saved: %s', path.name)

def run(segmented_parquet: Path | str, output_dir: Path | str,
        cfg: dict, stages: list[str]) -> Path:
    """
    Enrich a segmented .parquet file through all/some stages
    Returns path to the enriched .parquet file
    """
    segmented_parquet = Path(segmented_parquet)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = segmented_parquet.stem.replace('_segmented', '')
    final_out = output_dir / f'{stem}_enriched.parquet'
    
    stages = stages or _ALL_STAGES
    enrich_cfg = cfg.get('enrich', {})

    # Set up Archimedes Python path
    setup_path(cfg['archimedes']['python_src'])

    # Load geospatial context (ports, placemarks, MPAs, TSS, GEBCO)
    context = load_context(enrich_cfg.get('context', {}))

    # Instantiate enrichers
    port_cfg = enrich_cfg.get('port_proximity', {})
    context_cfg = enrich_cfg.get('context', {})
    spatial_enr = SpatialEnricher(context, port_cfg, context_cfg)

    # Determine starting point from latest checkpoint
    df = None
    start_from = 0
    for idx, stage in enumerate(_ALL_STAGES):
        ckpt = _load_checkpoint(_checkpoint_path(output_dir, stem, stage))
        if ckpt is not None:
            df = ckpt
            start_from = idx + 1
    
    if df is None:
        logger.info('Loading segmented .parquet: %s', segmented_parquet)
        df = pd.read_parquet(segmented_parquet)

    # ========== Run stages ==========
    stage_funcs = {
        'spatial': lambda d: spatial_enr.enrich(d)
    }

    for idx, stage in enumerate(_ALL_STAGES):
        if idx < start_from or stage not in stages:
            continue
        logger.info('Enrichment stage: %s (%d rows)', stage, len(df))
        df = stage_funcs[stage](df)
        _save_checkpoint(df, _checkpoint_path(output_dir, stem, stage))

    logger.info('Writing enriched .parquet: %s', final_out)
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(tbl, final_out, compression='snappy')

    # Clean up checkpoints
    for stage in _ALL_STAGES:
        ckpt = _checkpoint_path(output_dir, stem, stage)
        if ckpt.exists():
            ckpt.unlink()

    return final_out