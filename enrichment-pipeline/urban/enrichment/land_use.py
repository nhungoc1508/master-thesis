"""
Land use/functional zone enrichment using OSM (osmnx)

Columns added:
    land_use    str     residential | commercial | industrial | park |
                        forest | school | hospital | unknown

Implementation details:
    - https://wiki.openstreetmap.org/wiki/Land_use
    - For each GPS point, find the enclosing land use polygon and
        record its functional type
    - OSM polygon data (including landuse and amenity area features)
        is downloaded one per city & cached as a GeoPackage (gpkg)
    - Lookup uses a spatial join (point-in-polygon) using GeoPandas
    - Coordinates are deduplicated at 3 decimal places before lookup
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

from .osm_utils import features_from_place_or_bbox

logger = logging.getLogger(__name__)

_DEDUP_PRECISION = 3

_LANDUSE_TAGS = {
    "landuse": [
        "residential", "commercial", "industrial", "retail",
        "recreation_ground", "forest", "farmland", "grass",
        "cemetery", "institutional", "military", "religious",
        "allotments", "construction",
    ],
    "leisure": ["park", "nature_reserve", "garden", "golf_course"],
    "amenity": ["school", "hospital", "university", "college"],
}

# ========== Land use enrichment pipeline ==========

class LandUseEnricher:
    def __init__(self, cache_dir: Path, city_queries: dict[str, str],
                 city_bboxes: dict[str, tuple[float, float, float, float]] | None = None):
        """
        Params:
            cache_dir:      directory storying .gpkg cache files
            city_queries:   mapping from city name to osmnx place query string
            city_bboxes:    fallback bbox (W, S, E, N) bbox per city
        """
        self.cache_dir = Path(cache_dir)
        self.city_queries = city_queries
        self.city_bboxes = city_bboxes or {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._city_lu: dict = {}

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for city, city_df in df.groupby('city', sort=False):
            enriched = self._enrich_city(city_df.copy(), str(city))
            results.append(enriched)
        return pd.concat(results, ignore_index=True)
        
    def _enrich_city(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        try:
            lu_gdf = self._load_land_use(city)
        except Exception as exc:
            logger.warning('Land use data unavailable for %s: %s, filling with unknown', city, exc)
            df['land_use'] = 'unknown'
            return df
        
        df['_lat_r'] = df['lat'].round(_DEDUP_PRECISION)
        df['_lon_r'] = df['lon'].round(_DEDUP_PRECISION)
        unique_pts = (df.drop_duplicates(['_lat_r', '_lon_r'])
                      [['_lat_r', '_lon_r']].copy().reset_index(drop=True))

        pts_gdf = gpd.GeoDataFrame(
            unique_pts,
            geometry=[Point(r['_lon_r'], r['_lat_r']) for _, r in unique_pts.iterrows()],
            crs='EPSG:4326'
        )

        joined = gpd.sjoin(pts_gdf, lu_gdf[['geometry', 'lu_type']],
                           how='left', predicate='within')
        # If a point falls within multiple polygons, keep the most specific one
        joined = joined.sort_values('lu_type').drop_duplicates(
            subset=['_lat_r', '_lon_r'], keep='first'
        )
        unique_pts['land_use'] = joined['lu_type'].fillna('unknown').values

        df = df.merge(unique_pts, on=['_lat_r', '_lon_r'], how='left')
        df['land_use'] = df['land_use'].fillna('unknown')
        df.drop(columns=['_lat_r', '_lon_r'], inplace=True)
        return df
    
    def _load_land_use(self, city: str) -> gpd.GeoDataFrame:
        if city in self._city_lu:
            return self._city_lu[city]
        
        cache_file = self.cache_dir / f'{city}_landuse.gpkg'
        if cache_file.exists():
            logger.info('Loading land use from cache for %s', city)
            lu_gdf = gpd.read_file(cache_file)
        else:
            lu_gdf = self._download_land_use(city)
            lu_gdf.to_file(cache_file, driver='GPKG')
            logger.info('Saved land use to path: %s', cache_file)
        
        self._city_lu[city] = lu_gdf
        return lu_gdf
    
    def _download_land_use(self, city: str) -> gpd.GeoDataFrame:
        query = self.city_queries.get(city, city)
        logger.info('Downloading land use polygons for %s (%s)', city, query)
        feats = features_from_place_or_bbox(query, city, self.city_bboxes, tags=_LANDUSE_TAGS)
        polys = feats[feats.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
        polys = polys.to_crs('EPSG:4326')
        polys['lu_type'] = polys.apply(_classify_lu, axis=1)
        result = polys[['geometry', 'lu_type']].reset_index(drop=True)
        logger.info('\t%d land use polygons for %s', len(result), city)
        return result

# ========== Helper functions ==========

_TAG_PRIORITY = ['landuse', 'leisure', 'amenity']

def _classify_lu(row) -> str:
    for tag in _TAG_PRIORITY:
        v = row.get(tag)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return str(v)
    return 'other'