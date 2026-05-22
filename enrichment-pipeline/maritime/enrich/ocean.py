"""
Ocean feature enrichment

Uses the copernicusmarine Python client
Each API call is scoped to:
    - Geographic bbox of AIS batch + cfg buffer (default: 1 degree)
    - Time window of most cfg.batch_days days (default: 7)
    - Only needed variables (VHM0 for waves; uo/vo for currents)
    - Surface depth level only

Downloaded NetCDF tiles are cached

Columns added:
    wave_height_m           float       significant wave height (m), VHM0
    current_speed_ms        float       surface current speed (m/s)
    current_dir_deg         float       surface current direction (degrees from North)
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import copernicusmarine
import xarray as xr

logger = logging.getLogger(__name__)

def _bbox_from_df(df: pd.DataFrame, buffer: float) -> tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) with buffer added"""
    return (
        round(df['lon'].min() - buffer, 2),
        round(df['lat'].min() - buffer, 2),
        round(df['lon'].max() + buffer, 2),
        round(df['lat'].max() + buffer, 2)
    )

def _cache_key(dataset: str, bbox: tuple, t_start: str, t_end: str) -> str:
    raw = f'{dataset}_{bbox}_{t_start}_{t_end}'
    return hashlib.md5(raw.encode()).hexdigest()[:12]

def _fetch_window(dataset: str, variables: list[str], bbox: tuple,
                  t_start: datetime, t_end: datetime, cache_dir: Path) -> Path | None:
    """
    Download one time window Copernicus Marine Service and cache as NetCDF
    Returns local path
    """
    key = _cache_key(dataset, bbox, t_start.date().isoformat(), t_end.date().isoformat())
    nc_path = cache_dir / f'{key}.nc'

    if nc_path.exists():
        logger.info('Ocean cache hit: %s', nc_path.name)
        return nc_path
    
    min_lon, min_lat, max_lon, max_lat = bbox
    logger.info('Fetching CMEMS %s %s - %s bbox=(%s, %s, %s, %s)',
                dataset, t_start.date(), t_end.date(),
                min_lon, min_lat, max_lon, max_lat)
    try:
        ds = copernicusmarine.open_dataset(
            dataset_id=dataset,
            variables=variables,
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            start_datetime=t_start.strftime('%Y-%m-%dT%H:%M:%S'),
            end_datetime=t_end.strftime('%Y-%m-%dT%H:%M:%S'),
            maximum_depth=1.0
        )
        ds.load()
        ds.to_netcdf(nc_path, format='NETCDF3_64BIT', engine='scipy')
        ds.close()
        logger.info('Cached -> %s', nc_path)
        return nc_path
    except Exception as exc:
        logger.warning('CMEMS fetch failed for %s: %s', dataset, exc)
        nc_path.unlink(missing_ok=True)
        return None

def _load_ds(nc_path: Path):
    return xr.open_dataset(nc_path)

class OceanEnricher:
    def __init__(self, ocean_cfg: dict):
        self._wave_ds_id = ocean_cfg.get('wave_dataset', 'cmems_mod_glo_wav_my_0.2deg_PT3H-i')
        self._wave_var = ocean_cfg.get('wave_variable', 'VHM0')
        self._curr_ds_id = ocean_cfg.get('current_dataset', 'cmems_mod_glo_phy_my_0.083deg_P1D-m')
        self._curr_vars = ocean_cfg.get('current_variables', ['uo', 'vo'])
        self._cache_dir = Path(ocean_cfg.get('cache_dir', 'data/ocean_cache/'))
        self._batch_days = int(ocean_cfg.get('batch_days', 7))
        self._bbox_buf = float(ocean_cfg.get('bbox_buffer_deg', 1.0))
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        wave_h = np.full(n, np.nan)
        curr_s = np.full(n, np.nan)
        curr_d = np.full(n, np.nan)

        bbox = _bbox_from_df(df, self._bbox_buf)
        t_min = datetime.fromtimestamp(df['ts_unix'].min(), tz=timezone.utc)
        t_max = datetime.fromtimestamp(df['ts_unix'].max(), tz=timezone.utc)

        wave_datasets = dict()
        curr_datasets = dict()
        t_cursor = t_min.replace(hour=0, minute=0, second=0, microsecond=0)
        while t_cursor <= t_max:
            t_end_w = min(t_cursor + timedelta(days=self._batch_days), t_max + timedelta(hours=1))
            w_path = _fetch_window(self._wave_ds_id, [self._wave_var], bbox, t_cursor, t_end_w, self._cache_dir)
            if w_path:
                wave_datasets[t_cursor] = _load_ds(w_path)

            c_path = _fetch_window(self._curr_ds_id, self._curr_vars, bbox, t_cursor, t_end_w, self._cache_dir)
            if c_path:
                curr_datasets[t_cursor] = _load_ds(c_path)
            
            t_cursor += timedelta(days=self._batch_days)

        if not wave_datasets and not curr_datasets:
            logger.warning('No ocean data fetched; wave/current columns will be NaN')
            df['wave_height_m'] = np.nan
            df['current_speed_ms'] = np.nan
            df['current_dir_deg'] = np.nan
            return df
        
        # For each row, find which batch window covers it
        # Pick the latest window_start <= ts_unix
        dt_arr = pd.to_datetime(df['ts_unix'].values, unit='s')

        def _window_groups(datasets: dict) -> dict:
            """Return {window_key: positional_index_array} mapping"""
            if not datasets:
                return {}
            keys = sorted(datasets.keys())
            key_ts = np.array([k.timestamp() for k in keys])
            wi = np.clip(
                np.searchsorted(key_ts, df['ts_unix'].values, side='right') - 1,
                0, len(keys) - 1
            )
            return {keys[i]: np.where(wi == i)[0] for i in range(len(keys)) if (wi == i).any()}

        def _vec_sel(ds, var, positions):
            """Vectorized nearest-neighbor .sel() for a group of rows"""
            lats = xr.DataArray(df['lat'].values[positions], dims='points')
            lons = xr.DataArray(df['lon'].values[positions], dims='points')
            times = xr.DataArray(dt_arr[positions], dims='points')
            return ds[var].sel(
                latitude=lats, longitude=lons, time=times, method='nearest'
            ).squeeze().values.astype(float)
        
        # Wave height
        for wk, pos in _window_groups(wave_datasets).items():
            try:
                vals = _vec_sel(wave_datasets[wk], self._wave_var, pos)
                valid = ~np.isnan(vals)
                wave_h[pos[valid]] = np.round(vals[valid], 2)
            except Exception as exc:
                logger.warning('Wave lookup failed for window %s: %s', wk.date(), exc)

        # Current u/v -> speed & direction
        for wk, pos in _window_groups(curr_datasets).items():
            try:
                u = _vec_sel(curr_datasets[wk], self._curr_vars[0], pos)
                v = _vec_sel(curr_datasets[wk], self._curr_vars[1], pos)
                valid = ~(np.isnan(u) | np.isnan(v))
                curr_s[pos[valid]] = np.round(np.sqrt(u[valid]**2 + v[valid]**2), 3)
                curr_d[pos[valid]] = np.round((np.degrees(np.arctan2(u[valid], v[valid])) + 360) % 360, 1)
            except Exception as exc:
                logger.warning('Current speed & direction lookup failed for window %s: %s', wk.date(), exc)
            
        df['wave_height_m'] = wave_h
        df['current_speed_ms'] = curr_s
        df['current_dir_deg'] = curr_d
        return df