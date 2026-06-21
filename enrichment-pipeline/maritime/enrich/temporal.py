"""
Temporal feature enrichment

Timestamp handling:
    - UTC timestamps
    - Local timezone is less meaningful for maritime context

Columns added:
    hour_of_day             int     0-23 (UTC)
    day_of_week             int     0=Monday, ..., 6=Sunday (UTC)
    is_weekend              bool
    month                   int     1-12
    season                  str     spring | summer | autumn | winter
    time_of_day_category    str     early_morning | am_rush | morning | midday |
                                    afternoon | pm_rush | evening | night
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ISO 3166 -> IANA timezone
# todo: need expansion?
_ISO_TO_TZ: dict[str, str] = {
    'DNK': 'Europe/Copenhagen',
    'SWE': 'Europe/Stockholm',
    'NOR': 'Europe/Oslo',
    'DEU': 'Europe/Berlin',
    'POL': 'Europe/Warsaw',
    'FIN': 'Europe/Helsinki',
    'EST': 'Europe/Tallinn',
    'LVA': 'Europe/Riga',
    'LTU': 'Europe/Vilnius',
    'NLD': 'Europe/Amsterdam',
    'BEL': 'Europe/Brussels',
    'GBR': 'Europe/London',
    'IRL': 'Europe/Dublin',
    'FRA': 'Europe/Paris',
    'ESP': 'Europe/Madrid',
    'PRT': 'Europe/Lisbon',
    'RUS': 'Europe/Moscow',
    'USA': 'America/New_York',
    'CAN': 'America/Halifax',
    'high_seas': 'UTC',
}

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal columns to a trajectory DataFrame"""
    if 'eez_country_iso' not in df.columns:
        raise ValueError('temporal.enrich() requires "eez_country_iso"; spatial enrichment needed')

    df = df.copy()
    n = len(df)

    # Map EEZ ISO code to IANA timezone
    tz_series = df['eez_country_iso'].map(_ISO_TO_TZ).fillna('UTC')
    unknown = set(df['eez_country_iso'].unique()) - set(_ISO_TO_TZ)
    if unknown:
        logger.warning('No timezone mapping available for EEZ codes %s; using UTC', unknown)

    df['_pos'] = np.arange(n)
    df['_tz'] = tz_series
    
    hour_arr = np.empty(n, dtype=np.int8)
    dow_arr = np.empty(n, dtype=np.int8)
    mon_arr = np.empty(n, dtype=np.int8)

    for tz_name, grp in df.groupby('_tz', sort=False):
        dt = pd.to_datetime(grp['ts_unix'], unit='s', utc=True).dt.tz_convert(tz_name)
        pos = grp['_pos'].values
        hour_arr[pos] = dt.dt.hour.values
        dow_arr[pos] = dt.dt.dayofweek.values
        mon_arr[pos] = dt.dt.month.values
    
    df = df.drop(columns=['_pos', '_tz'])

    df['hour_of_day']          = hour_arr
    df['day_of_week']          = dow_arr
    df['is_weekend']           = dow_arr >= 5
    df['month']                = mon_arr
    df['season']               = pd.Series(mon_arr, dtype=object).map(_month_to_season).values
    df['time_of_day_category'] = pd.Series(hour_arr, dtype=object).map(_hour_to_category).values

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