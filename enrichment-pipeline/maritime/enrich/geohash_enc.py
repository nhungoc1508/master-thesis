"""
Geohash spatial tokenization

Columns added:
    geohash_5   str     5-character geohash (~4.89km x 4.89km)
    geohash_7   str     7-character geohash (~153m x 153m)
"""
from __future__ import annotations

import pandas as pd
import pygeohash as pgh

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['geohash_5'] = df.apply(lambda r: pgh.encode(r['lat'], r['lon'], precision=5), axis=1)
    df['geohash_7'] = df.apply(lambda r: pgh.encode(r['lat'], r['lon'], precision=7), axis=1)
    return df