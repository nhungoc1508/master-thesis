"""
Road network enrichment using OSM (osmnx)

For each GPS point, find the nearest road edge and extracts:
    road_type       str     OSM highway tag (motorway, primary, residential, etc.)
                            https://wiki.openstreetmap.org/wiki/Key:highway
    road_name       str     OSM name tag (or empty string)
    speed_limit_kmh float   OSM maxspeed (or NaN)
                            https://wiki.openstreetmap.org/wiki/Key:maxspeed
    road_lanes      int     number of lanes (or 0)
    road_oneway     bool    True if one-way street

Implementation details:
    - OSM road networks are downloaded one per city and cached as GraphML files
    - Nearest edge lookup: Shapely's STRtree
    - Coordinates are deduplicated at 4 decimal places before lookup
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree

from .osm_utils import graph_from_place_or_bbox

logger = logging.getLogger(__name__)

_DEDUP_PRECISION = 4 # 0.0001° ≈ 11.1 m, see https://www.movable-type.co.uk/scripts/latlong.html

# ========== Road network enrichment pipeline ==========

class RoadNetworkEnricher:
    def __init__(self, cache_dir: Path, city_queries: dict[str, str],
                 city_bboxes: dict[str, tuple[float, float, float, float]] | None = None):
        """
        Params:
            cache_dir:      directory storying GraphML cache files
            city_queries:   mapping from city name to osmnx place query string
            city_bboxes:    fallback bbox (W, S, E, N) bbox per city
        """
        self.cache_dir = Path(cache_dir)
        self.city_queries = city_queries
        self.city_bboxes = city_bboxes or {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tree: dict[str, tuple[STRtree, gpd.GeoDataFrame]] = {}

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        results = []
        for city, city_df in df.groupby('city', sort=False):
            enriched = self._enrich_city(city_df.copy(), str(city))
            results.append(enriched)
        return pd.concat(results, ignore_index=True)

    def _enrich_city(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        try:
            tree, edges = self._load_tree(city)
        except Exception as exc:
            logger.warning('Road network unavailable for %s: %s, filling with unknown', city, exc)
            return _fill_unknown(df)
        
        # Deduplication
        df['_lat_r'] = df['lat'].round(_DEDUP_PRECISION)
        df['_lon_r'] = df['lon'].round(_DEDUP_PRECISION)

        unique_pts = df.drop_duplicates(['_lat_r', '_lon_r'])[['_lat_r', '_lon_r']].copy()
        geoms = [Point(r['_lon_r'], r['_lat_r']) for _, r in unique_pts.iterrows()]

        # Find index of nearest edges to geoms from STRtree
        nearest_idx = tree.nearest(geoms)
        nearest_edges = edges.iloc[nearest_idx].reset_index(drop=True)

        unique_pts['road_type'] = nearest_edges['highway'].apply(_normalize_tag).values
        unique_pts['road_name'] = nearest_edges['name'].apply(_safe_str).values
        unique_pts['speed_limit_kmh'] = nearest_edges['maxspeed'].apply(_parse_speed).values
        unique_pts['road_lanes'] = nearest_edges['lanes'].apply(_parse_int).values
        unique_pts['road_oneway'] = nearest_edges['oneway'].astype(bool).values

        df = df.merge(unique_pts, on=['_lat_r', '_lon_r'], how='left')
        df.drop(columns=['_lat_r', '_lon_r'], inplace=True)
        return df
    
    def _load_tree(self, city: str) -> tuple[STRtree, gpd.GeoDataFrame]:
        if city in self._trees:
            return self._trees[city]

        cache_file = self.cache_dir / f'{city}.graphml'
        if cache_file.exists():
            logger.info('Loading road network from cache for %s', city)
            G = ox.load_graphml(cache_file)
        else:
            query = self.city_queries.get(city, city)
            logger.info('Downloading road network for %s (%s)', city, query)
            G = graph_from_place_or_bbox(query, city, self.city_bboxes, network_type='drive')
            ox.save_graphml(G, cache_file)
            logger.info('Saved road network to path: %s', cache_file)
        
        _, edges = ox.graph_to_gdfs(G)
        edges = edges.reset_index()

        # Build STRtree on edge geometries
        tree = STRtree(edges.geometry.values)
        self._trees[city] = (tree, edges)
        return tree, edges
    
# ========== Helper functions ==========

def _fill_unknown(df: pd.DataFrame) -> pd.DataFrame:
    df['road_type'] = 'unknown'
    df['road_name'] = ''
    df['speed_limit_kmh'] = np.nan
    df['road_lanes'] = 0
    df['road_oneway'] = False
    return df

def _normalize_tag(val) -> str:
    if isinstance(val, (list, np.ndarray)):
        val = val[0] if len(val) > 0 else None
    if val is None:
        return 'unclassified'
    try:
        return str(val) if pd.notna(val) else 'unclassified'
    except (TypeError, ValueError):
        return str(val)
    
def _safe_str(val) -> str:
    if isinstance(val, (list, np.ndarray)):
        val = val[0] if len(val) > 0 else None
    if val is None:
        return ''
    try:
        return str(val) if pd.notna(val) else ''
    except (TypeError, ValueError):
        return str(val)
    
def _parse_speed(val) -> float:
    if val is None:
        return np.nan
    if isinstance(val, (list, np.ndarray)):
        val = val[0] if len(val) > 0 else None
    try:
        if pd.isna(val):
            return np.nan
    except (TypeError, ValueError):
        pass
    s = str(val).split(';')[0].strip()
    m = re.search(r'(\d+)', s)
    return float(m.group(1)) if m else np.nan

def _parse_int(val) -> int:
    if val is None:
        return 0
    if isinstance(val, (list, np.ndarray)):
        val = val[0] if len(val) > 0 else None
    try:
        if pd.isna(val):
            return 0
    except (TypeError, ValueError):
        pass
    s = str(val).split(';')[0].strip()
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else 0