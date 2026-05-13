"""
Weather enrichment pipeline using Open-Meteo API & ERA5 NetCDF files

Columns added:
    temperature_c       float   temperature in degree Celcius
    precipitation_mm    float   mm
    wind_speed_kmh      float   km/h
    wind_direction_deg  float   degrees
    weather_code        int     WMO weather code
    weather_description str     human-readable WMO description

Implementation details:
    - Prioritize looking for local .nc files (ERA5 NetCDF)
    - If data for the month-year-city tuple is not available, use Open-Meteo
    - WMO weather codes are not available in ERA5, deriving a simple
        description using precipitation & temperature and setting weather_code = -1
"""
from __future__ import annotations

import calendar
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import xarray as xr

logger = logging.getLogger(__name__)

# ERA5 local directory mapping, each directory = one city and has 'YYYY-mm.nc' file(s)
# Prioritize checking local files before resorting to API calls
_CITY_ERA5_DIR: dict[str, str] = {
    'porto':         'data/weather_era5/porto',
    'athens':        'data/weather_era5/athens',
    'beijing':       'data/weather_era5/beijing',
    'san_francisco': 'data/weather_era5/san_francisco',
    'new_york':      'data/weather_era5/new_york',
    'rome':          ''
}

# Open-Meteo variables
_ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'
_VARIABLES = 'temperature_2m,precipitation,wind_speed_10m,wind_direction_10m,weather_code'

# City center coordinates for weather lookup
CITY_WEATHER_COORDS: dict[str, tuple[float, float]] = {
    "porto":         (41.15,  -8.61),
    "athens":        (37.98,  23.73),
    "beijing":       (39.91, 116.39),
    "san_francisco": (37.78, -122.42),
    "new_york":      (40.71,  -74.01),
    "rome":          (41.90,   12.50),
    "hanoi":         (21.03, 105.85),
    "brussels":      (50.85, 4.35),
}

# Mapping from WMO codes to readable strings
# https://open-meteo.com/en/docs#weather_variable_documentation
WMO_DESCRIPTIONS: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "heavy drizzle",
    56: "light freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains",
    80: "light rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with light hail", 99: "thunderstorm with heavy hail",
}

# ========== Weather enricher pipeline using Open-Meteo ==========

class WeatherEnricher:
    def __init__(self, cache_dir : Path, use_city_coords: bool = True,
                 location_precision: int = 1, request_delay_s: float = 1.0):
        """
        Params:
            cache_dir:          directory storing JSON weather cache files
            use_city_coords:    if True, use canonical city-center coordinates
            location_precision: decimal places for coord rounding when use_city_coords=False
            request_delay_s:    seconds to sleep between API calls
        """
        self.cache_dir = Path(cache_dir)
        self.use_city_coords = use_city_coords
        self.location_precision = location_precision
        self.request_delay_s = request_delay_s
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['_dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df['_date'] = df['_dt'].dt.date.astype(str)
        df['_hour'] = df['_dt'].dt.hour

        if self.use_city_coords:
            lookup = self._build_lookup_city(df)
            df['_key'] = list(zip(df['city'], df['_date'], df['_hour']))
        else:
            df['_lat_r'] = df['lat'].round(self.location_precision)
            df['_lon_r'] = df['lon'].round(self.location_precision)
            lookup = self._build_lookup_grid(df)
            df['_key'] = list(zip(df['_lat_r'], df['_lon_r'], df['_date'], df['_hour']))
        
        result_rows = [lookup.get(k, _MISSING) for k in df['_key']]
        cols = ['temperature_c', 'precipitation_mm', 'wind_speed_kmh',
                'wind_direction_deg', 'weather_code', 'weather_description']
        for i, col in enumerate(cols):
            df[col] = [r[i] for r in result_rows]
        
        drop_cols = ['_dt', '_date', '_hour', '_key']
        if '_lat_r' in df.columns:
            drop_cols += ['_lat_r', '_lon_r']
        df.drop(columns=drop_cols, inplace=True)
        return df
    
    # ---------- City-level lookup ----------

    def _build_lookup_city(self, df: pd.DataFrame) -> dict:
        lookup: dict = {}
        city_groups = df.groupby('city', sort=False)
        total = len(city_groups)

        for city_idx, (city, grp) in enumerate(city_groups, 1):
            city_str = str(city).lower()
            lat_r, lon_r = CITY_WEATHER_COORDS.get(
                city_str,
                (
                    round(grp['lat'].median(), self.location_precision),
                    round(grp['lon'].median(), self.location_precision)
                )
            )
            if city_str not in CITY_WEATHER_COORDS:
                logger.warning('City %d not in CITY_WEATHER_COORDS, using median coordinate (%.2f, %.2f)',
                               city_str, lat_r, lon_r)
                
            # Fetch one month per request
            year_months = sorted({(int(ts.year), int(ts.month)) for ts in grp['_dt']})
            n_months = len(year_months)

            for ym_idx, (year, month) in enumerate(year_months, 1):
                month_mask = (grp['_dt'].dt.year == year) & (grp['_dt'].dt.month == month)
                era5_nc = _find_era5_nc(city_str, year, month)

                if era5_nc is not None:
                    # ERA5 data available for this month
                    hourly_tuples = _load_era5_month(era5_nc, lat_r, lon_r)
                    for (date_str, hour), _ in grp.loc[month_mask].groupby(['_date', '_hour']):
                        key = (city_str, date_str, int(hour))
                        if key not in lookup:
                            lookup[key] = hourly_tuples.get(
                                f'{date_str}T{int(hour):02d}:00', _MISSING
                            )
                    src = 'ERA5'
                else:
                    # Fall back to Open-Meteo API
                    n_days = calendar.monthrange(year, month)[1]
                    start_dt = f'{year}-{month:02d}-01'
                    end_dt = f'{year}-{month:02d}-{n_days:02d}'
                    cache_key = f'city_{city_str}_{year}_{month:02d}'
                    hourly = self._fetch_or_load(lat_r, lon_r, start_dt, end_dt, cache_key)
                    for (date_str, hour), _ in grp.loc[month_mask].groupby(['_date', '_hour']):
                        key = (city_str, date_str, int(hour))
                        if key not in lookup:
                            w = hourly.get(f'{date_str}T{int(hour):02d}:00')
                            lookup[key] = _format_weather(w)
                    src = 'API'
                
                if ym_idx % 6 == 0 or ym_idx == n_months:
                    logger.info(
                        'Weather [city %d/%d] %s: %d/%d months (last: %04d-%02d via %s)',
                        city_idx, total, city_str, ym_idx, n_months, year, month, src
                    )

        return lookup
    
    # ---------- Grid-cell lookup ----------

    def _build_lookup_grid(self, df: pd.DataFrame) -> dict:
        lookup: dict = {}
        loc_groups = df.groupby(['_lat_r', '_lon_r'])
        total = len(loc_groups)

        for loc_idx, ((lat_r, lon_r), grp) in enumerate(loc_groups, 1):
            dates = sorted(grp['_date'].unique())
            start_date = dates[0]
            end_date = dates[-1]
            cache_key = f'{lat_r}_{lon_r}_{start_date}_{end_date}'
            hourly = self._fetch_or_load(lat_r, lon_r, start_date, end_date, cache_key)

            for (date_str, hour), _ in grp.groupby(['_date', '_hour']):
                key = (lat_r, lon_r, date_str, int(hour))
                if key not in lookup:
                    w = hourly.get(f'{date_str}T{int(hour):02d}:00')
                    lookup[key] = _format_weather(w)
                
            if loc_idx % 10 == 0:
                logger.info('Weather: %d/%d grid cells fetched', loc_idx, total)

        return lookup
    
    # ---------- Fetching/caching ----------

    def _fetch_or_load(
            self, lat: float, lon: float,
            start_date: str, end_date: str,
            cache_key: str
    ) -> dict:
        cache_file = self.cache_dir / f'{cache_key}.json'
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        
        params = {
            'latitude':   lat,
            'longitude':  lon,
            'start_date': start_date,
            'end_date':   end_date,
            'hourly':     _VARIABLES,
            'timezone':   'UTC'
        }

        for attempt in range(5):
            try:
                resp = requests.get(_ARCHIVE_URL, params=params, timeout=60)
                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    logger.warning('API rate-limited (429), waiting %ds before retry %d/5',
                                   wait, attempt + 1)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                hourly = _parse_response(data)
                with open(cache_file, 'w') as f:
                    json.dump(hourly, f)
                time.sleep(self.request_delay_s)
                return hourly
            except requests.HTTPError:
                raise
            except Exception as exc:
                logger.warning('Weather API attempt %d/5 failed: %s', attempt + 1, exc)
                time.sleep(2**attempt)
        
        logger.error('Weather fetch failed for (%.2f, %.2f) %s-%s after 5 attemps',
                     lat, lon, start_date, end_date)
        return {}

# ========== Helper functions ==========

_MISSING = (np.nan, np.nan, np.nan, np.nan, -1, 'unknown')

def _parse_response(data: dict) -> dict:
    """Convert Open-Meteo response to { ISO_hour_str: raw_values_dict }"""
    h = data.get('hourly', {})
    times = h.get('time', [])
    keys  = ['temperature_2m', 'precipitation', 'wind_speed_10m',
             'wind_direction_10m', 'weather_code']
    out = {}
    for i, t in enumerate(times):
        out[t] = { k: (h[k][i] if k in h and h[k] else None) for k in keys }
    return out

def _format_weather(w: dict | None) -> tuple:
    if not w:
        return _MISSING
    code = int(w.get('weather_code') or -1)
    return (
        float(w['temperature_2m'])     if w.get('temperature_2m')     is not None else np.nan,
        float(w['precipitation'])      if w.get('precipitation')      is not None else np.nan,
        float(w['wind_speed_10m'])     if w.get('wind_speed_10m')     is not None else np.nan,
        float(w['wind_direction_10m']) if w.get('wind_direction_10m') is not None else np.nan,
        code,
        WMO_DESCRIPTIONS.get(code, 'unknown')
    )

# ========== Weather enricher pipeline using ERA5 ==========

class ERA5WeatherEnricher:
    def __init__(self, nc_path: Path | str,
                 city_coords: dict[str, tuple[float, float]] | None = None):
        """
        Params:
            nc_path: path to a single .nc file or dir containing multiple .nc files
            city_coords: optional override for city-center coordinates
        """
        p = Path(nc_path)
        if p.is_dir():
            nc_files = sorted(p.glob('*.nc'))
            if not nc_files:
                raise FileNotFoundError(f'No .nc files found in {p}')
            logger.info('ERA5: opening %d files from %s', len(nc_files), p)
            self.ds = xr.open_mfdataset(nc_files, combine='by_coords')
        else:
            self.ds = xr.open_dataset(str(p))

        if 'valid_time' in self.ds.coords and 'time' not in self.ds.coords:
            self.ds = self.ds.rename({'valid_time': 'time'})

        self.city_coords = city_coords or CITY_WEATHER_COORDS
        logger.info(
            'ERA5 dataset ready: %d time steps, lat [%.2f, %.2f], lon [%.2f, %.2f]',
            len(self.ds.time),
            float(self.ds.latitude.min()), float(self.ds.latitude.max()),
            float(self.ds.longitude.min()), float(self.ds.longitude.max())
        )

    def enrich(self, df:pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['_dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df['_date'] = df['_dt'].dt.date.astype(str)
        df['_hour'] = df['_dt'].dt.hour

        lookup = self._build_lookup(df)
        df['_key'] = list(zip(df['city'], df['_date'], df['_hour']))

        result_rows = [lookup.get(k, _MISSING) for k in df['_key']]
        cols = ['temperature_c', 'precipitation_mm', 'wind_speed_kmh',
                'wind_direction_deg', 'weather_code', 'weather_description']
        for i, col in enumerate(cols):
            df[col] = [r[i] for r in result_rows]

        df.drop(columns=['_dt', '_date', '_hour', '_key'], inplace=True)
        return df
    
    def _build_lookip(self, df: pd.DataFrame) -> dict:
        lookup: dict = {}

        era5_lat_min = float(self.ds.latitude.min())
        era5_lat_max = float(self.ds.latitude.max())
        era5_lon_min = float(self.ds.longitude.min())
        era5_lon_max = float(self.ds.longitude.max())

        for city, grp in df.groupby('city', sort=False):
            city_str = str(city).lower()
            if city_str in self.city_coords:
                lat, lon = self.city_coords[city_str]
            else:
                lat = float(grp['lat'].median())
                lon = float(grp['lon'].median())
                logger.warning('City %s not in city_coords, using median (%.3f, %.3f)',
                               city_str, lat, lon)

            if not (era5_lat_min <= lat <= era5_lat_max and era5_lon_min <= lon <= era5_lon_max):
                logger.warning(
                    'City %s has coordinate (%.3f, %.3f) is outside ERA5 file coverage'
                    '[lat %.2f-%.2f, lon %.2f-%.2f]',
                    city_str, lat, lon, era5_lat_min, era5_lat_max, era5_lon_min, era5_lon_max
                )
            
            for (date_str, hour), _ in grp.groupby(['_date', '_hour']):
                key = (city_str, date_str, int(hour))
                if key in lookup:
                    continue
                ts = f'{date_str}T{int(hour):02d}:00'
                lookup[key] = self._lookup_era5(lat, lon, ts)
            
        return lookup
    
    def _lookup_era5(self, lat: float, lon: float, iso_hour: str) -> tuple:
        try:
            row = (
                self.ds
                    .sel(latitude=lat, longitude=lon, method='nearest')
                    .sel(time=iso_hour)
            )
            temp_c = float(row['t2m'].values) - 273.15 # Kelvin to Celcius
            precip_mm = float(row['tp'].values) * 1000.0 # m to mm
            u10 = float(row['u10'].values)
            v10 = float(row['v10'].values)
            speed_kmh = (u10 ** 2 + v10 ** 2) ** 0.5 * 3.6
            wind_dir = (270.0 - float(np.degrees(np.arctan2(v10, u10)))) % 360.0
            desc = _era5_weather_description(temp_c, precip_mm)
            return (temp_c, precip_mm, speed_kmh, wind_dir, -1, desc)
        except Exception as exc:
            logger.debug('ERA5 lookup failed for (%s, %.3f, %.3f): %s', iso_hour, lat, lon, exc)
            return _MISSING
        
    def close(self) -> None:
        self.ds.close()

# ========== Helper functions ==========

def _era5_weather_description(temp_c: float, precip_mm: float) -> str:
    """Simple weather description derived from temperature and precipitation"""
    if np.isnan(precip_mm) or np.isnan(temp_c):
        return 'unknown'
    if precip_mm >= 0.5 and temp_c < 2.0:
        return 'snow'
    if precip_mm >= 4.0:
        return 'heavy rain'
    if precip_mm >= 0.3:
        return 'moderate rain'
    if precip_mm >= 0.05:
        return 'light rain'
    return 'dry'

def _find_era5_nc(city: str, year: int, month: int) -> Path | None:
    """Return path to ERA5 .nc for for this city/month/year, None if absent"""
    era5_dir = _CITY_ERA5_DIR.get(city, '')
    if not era5_dir:
        return None
    p = Path(era5_dir) / f'{year}-{month:02d}.nc'
    return p if p.exists() else None

def _load_era5_month(nc_path: Path, lat: float, lon: float) -> dict:
    """Read all hourly value for one (lat, lon) from a .nc file"""
    ds = xr.open_dataset(str(nc_path), engine='netcdf4')
    try:
        if 'valid_time' in ds.coords and 'time' not in ds.coords:
            ds = ds.rename({'valid_time': 'time'})
        
        point = ds.sel(latitude=lat, longitude=lon, method='nearest')

        times = pd.to_datetime(point.time.values)
        if times.tz is not None:
            times = times.tz_localize(None)

        t2m = point['t2m'].values - 273.15 # Kelvin to Celcius
        tp = point['tp'].values * 1000.0 # m to mm
        u10 = point['u10'].values
        v10 = point['v10'].values
        spd = (u10 ** 2 + v10 ** 2) ** 0.5 * 3.6
        wdir = (270.0 - np.degrees(np.arctan2(v10, u10))) % 360.0

        result = {}
        for i, ts in enumerate(times):
            iso_hour = f'{ts.date()}T{ts.hour:02d}:00'
            temp_c = float(t2m[i])
            precip_mm = float(tp[i])
            result[iso_hour] = (
                temp_c,
                precip_mm,
                float(spd[i]),
                float(wdir[i]),
                -1,
                _era5_weather_description(temp_c, precip_mm)
            )
        return result
    finally:
        ds.close()