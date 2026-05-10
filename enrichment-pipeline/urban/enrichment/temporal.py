"""
Temporal feature enrichment

Timestamp handling:
    - All timestamps in canonical .parquet files are UTC Unix seconds
    - Temporal features are derived using local time of each city
        to better reflect local context

Columns added:
    hour_of_day             int     0-23 (local time)
    day_of_week             int     0=Monday, ..., 6=Sunday (local time)
    is_weekend              bool
    month                   int     1-12 (local time)
    season                  str     spring | summer | autumn | winter
    time_of_day_category    str     early_morning | am_rush | morning | midday |
                                    afternoon | pm_rush | evening | night
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ========== IANA timezone mapping ==========
CITY_TIMEZONES: dict[str, str] = {
    'porto':            'Europe/Lisbon',
    'athens':           'Europe/Athens',
    'beijing':          'Asia/Shanghai',
    'new_york':         'America/New_York',
    'san_francisco':    'America/Los_Angeles',
    'rome':             'Europe/Rome',
    'brussels':         'Europe/Brussels',
    'hanoi':            'Asia/Bangkok'
}

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal columns to a trajectory DataFrame
    Falls back to UTC for any city not found in CITY_TIMEZONES,
        logs a warning
    """
    if 'city' not in df.columns:
        # No city column found, use UTC
        logger.warning('No "city" column found; temporal features derived in UTC')
        return _enrich_partition(df, 'UTC')

    parts = []

def _enrich_partition(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Compute temporal features for a single-timezone partition"""
    dt = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(tz)

    df['hour_of_day']          = dt.dt.hour.astype(np.int8)
    df['day_of_week']          = dt.dt.dayofweek.astype(np.int8)
    df['is_weekend']           = df['day_of_week'] >= 5
    df['month']                = dt.dt.month.astype(np.int8)
    df['season']               = dt['month'].map(_month_to_season)
    df['time_of_day_category'] = df['hour_of_day'].map(_hour_to_category)
    return df

def _month_to_season(m: int) -> str:
    if m in (3, 4, 5): return 'spring'
    if m in (6, 7, 8): return 'summer'
    if m in (9, 10, 11): return 'autumn'
    return 'winter'

def _hour_to_category(h: int) -> str:
    if h < 5: return 'night'
    if h < 7: return 'early_morning'
    if h < 9: return 'am_rush'
    if h < 12: return 'morning'
    if h < 14: return 'midday'
    if h < 17: return 'afternoon'
    if h < 19: return 'pm_rush'
    if h < 22: return 'evening'
    return 'night'