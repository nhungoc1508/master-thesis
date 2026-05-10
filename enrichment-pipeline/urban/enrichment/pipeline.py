"""
Enrichment pipeline central orchestrator

Reads a canonical .parquet file (one row per GPS point) then
runs enrichment stages
Writes a checkpoint after each stage so the pipeline can be resumed

Stages:
    temporal        temporal features derived from timestamp
    behavioral      kinematic features + phases derived from GPS + timestamp
    road_network    OSM road type + name + speed limit
    poi             multi-radius POI category profiles
    land_use        OSM land use + functional zone
    weather         Open-Meteo/ERA5 weather
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

from . import temporal, behavioral
from .road_network import RoadNetworkEnricher
from .poi_profiles import POIProfileEnricher
from .land_use import LandUseEnricher
from .weather import WeatherEnricher, ERA5WeatherEnricher

logger = logging.getLogger(__name__)

ALL_STAGES = ['temporal', 'behavioral', 'road_network', 'poi', 'land_use', 'weather']

class EnrichmentPipeline:
    def __init__(self,
                 road_enricher:     RoadNetworkEnricher,
                 poi_enricher:      POIProfileEnricher,
                 lu_enricher:       LandUseEnricher,
                 weather_enricher:  WeatherEnricher | ERA5WeatherEnricher,
                 stop_speed_ms:     float = 0.5,
                 slow_speed_ms:     float = 1.5):
        self.road = road_enricher
        self.poi = poi_enricher
        self.lu = lu_enricher
        self.weather = weather_enricher
        self._stop_ms = stop_speed_ms
        self._slow_ms = slow_speed_ms

    @classmethod
    def from_config(cls, config_path: str | Path,
                    cache_root: str | Path = 'data') -> "EnrichmentPipeline":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        cache_root = Path(cache_root)
        city_queries = cfg.get('city_osmnx_queries', {})
        # (W, S, E, N) fallback bboxes for cities where Nominatim
        # doesn't return a polygon boundary
        city_bboxes = {
            city: tuple(v)
            for city, v in cfg.get('city_bboxes', {}).items()
        }
        beh = cfg.get('behavioral', {})
        w_cfg = cfg.get('weather', {})

        era5_path = w_cfg.get('era5_nc_path')
        if era5_path:
            weather_enricher: WeatherEnricher | ERA5WeatherEnricher = ERA5WeatherEnricher(
                nc_path = Path(era5_path)
            )
        else:
            weather_enricher = WeatherEnricher(
                cache_dir = cache_root / 'weather_cache',
                use_city_coords = w_cfg.get('use_city_coords', True),
                location_precision = w_cfg.get('location_precision', 1),
                request_delay_s = w_cfg.get('request_delay_s', 1.0)
            )

        return cls(
            weather_enricher = weather_enricher,
            road_enricher = RoadNetworkEnricher(
                cache_dir = cache_root / 'osm_cache' / 'road_network',
                city_queries = city_queries,
                city_bboxes = city_bboxes
            ),
            poi_enricher = POIProfileEnricher(
                cache_dir = cache_root / 'osm_cache' / 'poi',
                city_queries = city_queries,
                city_bboxes = city_bboxes
            ),
            lu_enricher = LandUseEnricher(
                cache_dir = cache_root / 'osm_cache' / 'land_use',
                city_queries = city_queries,
                city_bboxes = city_bboxes
            ),
            stop_speed_ms = beh.get('stop_speed_ms', 0.5),
            slow_speed_ms = beh.get('slow_speed_ms', 1.5)
        )

    # ---------- Public entry point ----------

    def run(self, input_path: str | Path, output_dir: str | Path,
            stages: Sequence[str] | None = None) -> Path:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stages = list(stages or ALL_STAGES)
        _validate_stages(stages)

        stem = input_path.stem
        ckpt_path = output_dir / f'{stem}_checkpoint.parquet'
        final_path = output_dir / f'{stem}_enriched.parquet'

        df, done_stages = self._load_checkpoint(ckpt_path, input_path)
        logger.into('Loaded %d rows, completed stages: %s', len(df), done_stages or '(none)')

        for stage in stages:
            if stage in done_stages:
                logger.info('\tSkipping %s (checkpoint)', stage)
                continue

            logger.info('\tRunning stage: %s', stage)
            df = self._run_stage(df, stage)
            df.to_parquet(ckpt_path, index=False)
            done_stages.append(stage)
            logger.info('\t%-12s done %s', stage, _stage_summary(df, stage))
        
        df.to_parquet(final_path, index=False)
        logger.info('Enrichment complete, .parquet file saved to: %s', final_path)
        return final_path
    
    def _run_stage(self, df: pd.DataFrame, stage: str) -> pd.DataFrame:
        if stage == 'temporal':
            return temporal.enrich(df)
        if stage == 'behavioral':
            return behavioral.enrich(df, self._stop_ms, self._slow_ms)
        if stage == 'road_network':
            return self.road.enrich(df)
        if stage == 'poi':
            return self.poi.enrich(df)
        if stage == 'land_use':
            return self.lu.enrich(df)
        if stage == 'weather':
            return self.weather.enrich(df)
        raise ValueError(f'Unknown stage: {stage}')
    
    @staticmethod
    def _load_checkpoint(ckpt_path: Path, fallback_path: Path):
        """Load checkpoint if exists, otherwise load canonical .parquet"""
        if ckpt_path.exists():
            df = pd.read_parquet(ckpt_path)
            # Infer completed stages
            done = _infer_done_stages(df)
            return df, done
        df = pd.read_parquet(fallback_path)
        return df, []
    
# ========== Helper functions ==========

_STAGES_SENTINEL_COL = {
    'temporal':     'hour_of_day',
    'behavioral':   'behavioral_phase',
    'road_network': 'road_type',
    'poi':          'poi_count_50m',
    'land_use':     'land_use',
    'weather':      'temperature_c'
}

def _infer_done_stages(df: pd.DataFrame) -> list[str]:
    return [s for s, col in _STAGES_SENTINEL_COL.items() if col in df.columns]

def _validate_stages(stages: list[str]) -> None:
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise ValueError(f'Unknown stages: {unknown}; valid: {ALL_STAGES}')

def _stage_summary(df: pd.DataFrame, stage: str) -> str:
    try:
        if stage == 'temporal':
            h = df['hour_of_day']
            wd = (~df['is_weekend']).mean()
            return (f'hours {int(h.min())} - {int(h.max())} '
                    f'weekday {wd:.0%} / weekend {1-wd:.0%}')
        
        if stage == 'behavioral':
            vc = df['behavioral_phase'].value_counts(normalize=True)
            parts = ' '.join(f'{k} {v:.0%}' for k, v in vc.items())
            spd = df['speed_ms']
            return f'{parts} | speed mean={spd.mean():.1f} max={spd.max():.1f} m/s'

        if stage == 'road_network':
            vc = df['road_type'].value_counts()
            top = ' '.join(f'{k}={v}' for k, v in vc.head(4).items())
            unk = int(vc.get('unknown', 0))
            return f'{top} (unknown={unk})'
        
        if stage == 'poi':
            for r in (50, 200, 500):
                col = f'poi_count_{r}m'
                if col in df.columns:
                    return (f'@{r}m: mean={df[col].mean():.1f} '
                            f'p50={df[col].median():.0f} '
                            f'max={int(df[col].max())}')

        if stage == 'land_use':
            vc = df['land_use'].value_counts()
            top = ' '.join(f'{k}={v}' for k, v in vc.head(5).items())
            return top
        
        if stage == 'weather':
            t = df['temperature_c']
            p = df['precipitation_mm']
            w = df['wind_speed_kmh']
            return (f'temp {t.min():.1f} - {t.max():.1f} '
                    f'precip max={p.max():.1f}mm '
                    f'wind max={w.max():.1f}km/h')
        
    except Exception as exc:
        raise Exception(f'Error while validating: {exc}')
    return ''