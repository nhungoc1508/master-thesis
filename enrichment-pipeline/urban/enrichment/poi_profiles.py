"""
Multi-radius POI profile enrichment using OSM (osmnx)

For each GPS point, count nearby POIs within 3 radii and records
    the total count + distribution across functional categories

Columns added:
    poi_count_50m           int     total POIs within 50m
    poi_count_200m          int     total POIs within 200m
    poi_count_500m          int     total POIs within 500m
    poi_categories_50m      str     JSON dict: {category: count}
    poi_categories_200m     str     JSON dict: {category: count}
    poi_categories_500m     str     JSON dict: {category: count}

Implementation details:
    - OSM POIs are downloaded one time per city and cached as GraphML files
    - Radius search: Shapely's STRtree
    - Coordinates are deduplicated at 3 decimal places before lookup
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree

from .osm_utils import features_from_place_or_bbox

logger = logging.getLogger(__name__)

_DEDUP_PRECISION = 3

# Mapping from OSM tag to category
# (osm_col, value/True, category)
_TAG_RULES: list[tuple[str, str, object, str]] = [
    ("amenity", {"restaurant","cafe","bar","fast_food","food_court","pub","ice_cream",
                 "bakery","biergarten"}, "food_drink"),
    ("amenity", {"school","university","college","library","kindergarten"}, "education"),
    ("amenity", {"hospital","clinic","pharmacy","doctors","dentist","veterinary"}, "healthcare"),
    ("amenity", {"bus_station","taxi","car_rental","bicycle_rental","fuel","parking"}, "transport"),
    ("amenity", {"hotel"}, "accommodation"),
    ("amenity", {"bank","atm"}, "office_finance"),
    ("amenity", {"place_of_worship"}, "religion"),
    ("amenity", {"town_hall","police","fire_station","post_office","courthouse","embassy"}, "government"),
    ("amenity", {"theatre","cinema","arts_centre","nightclub"}, "culture"),
    ("shop",    True,  "retail"),
    ("leisure", {"park","sports_centre","playground","gym","swimming_pool",
                 "stadium","fitness_centre"}, "recreation"),
    ("tourism", {"hotel","hostel","motel","guest_house","chalet"}, "accommodation"),
    ("tourism", {"museum","gallery","attraction","artwork","viewpoint"}, "culture"),
    ("office",  True,  "office_finance"),
    ("railway", {"station","halt","tram_stop","subway_entrance"}, "transport"),
    ("public_transport", {"station","stop_position"}, "transport"),
]

_RADII = [50, 200, 500]

# ========== POI enrichment pipeline ==========
class POIProfileEnricher:
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
        self._city_data: dict = {}

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for city, city_df in df.groupby('city', sort=False):
            enriched = self._enrich_city(city_df.copy(), str(city))
            results.append(enriched)
        return pd.concat(results, ignore_index=True)
    
    def _enrich_city(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        try:
            pois_proj, tree, crs = self._load_pois(city)
        except Exception as exc:
            logger.warning('POI data unavailable for %s: %s, filling with zeros', city, exc)
            return _fill_zero(df)
        
        df['_lat_r'] = df['lat'].round(_DEDUP_PRECISION)
        df['_lon_r'] = df['lon'].round(_DEDUP_PRECISION)

        unique_pts = (df.drop_duplicates(['_lat_r', '_lon_r'])
                      [['_lat_r', '_lon)r']].copy().reset_index(drop=True))
        
        # Project (unique) points to the same metric CRS as POIS
        pts_gdf = gpd.GeoDataFrame(
            unique_pts,
            geometry=[Point(r['_lon_r'], r['_lat_r']) for _, r in unique_pts.iterrows()],
            crs='EPSG:4326'
        ).to_crs(crs)

        # Build profile for 3 radii
        for radius in _RADII:
            counts, cat_jsons = [], []
            for pt_geom in pts_gdf.geometry:
                buffer = pt_geom.buffer(radius)
                nearby_idx = tree.query(buffer)
                nearby = pois_proj.iloc[nearby_idx]
                counts.append(len(nearby))
                if len(nearby):
                    cat_counts = nearby['category'].value_counts().to_dict()
                else:
                    cat_counts = {}
                cat_jsons.append(json.dumps(cat_counts, separators=(',', ':')))
            
            unique_pts[f'poi_count_{radius}m'] = counts
            unique_pts[f'poi_categories_{radius}m'] = cat_jsons
        
        df = df.merge(unique_pts, on=['_lat_r', '_lon_r'], how='left')
        df.drop(columns=['_lat_r', '_lon_r'], inplace=True)
        return df

    def _load_pois(self, city: str):
        if city in self._city_data:
            return self._city_data[city]
        
        cache_file = self.cache_dif / f'{city}_pois.gpkg'
        if cache_file.exists():
            logger.info('Loading POIs from cache for %s', city)
            pois = gpd.read_file(cache_file)
        else:
            pois = self._download_pois(city)
            if not pois.empty:
                pois[['geometry', 'category']].to_file(cache_file, driver='GPKG')
            logger.info('Saved POIs to path: %s', cache_file)
        
        if pois.empty:
            raise ValueError('No POI features found for %s', city)
        
        crs = pois.estimate_utm_crs()
        pois_proj = pois[['geometry', 'category']].to_crs(crs)
        tree = STRtree(pois_proj.geometry.values)

        self._city_data[city] = (pois_proj, tree, crs)
        return pois_proj, tree, crs
    
    def _download_pois(self, city: str) -> gpd.GeoDataFrame:
        query = self.city_queries.get(city, city)
        all_tags = _build_osmnx_tags()
        logger.info('Downloading POIs for %s (%s)', city, query)
        feats = features_from_place_or_bbox(query, city, self.city_bboxes, tags=all_tags)
        feats = feats.to_crs('EPSG:4326')

        # OSM POIs can be Point or (Multi)Polygon
        # Reduce areas to their centroids to use all POIs as point features
        is_point = feats.geometry.geom_type == 'Point'
        is_area = feats.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
        pts = feats[is_point].copy()
        area = feats[is_area].copy()
        if not area.empty:
            area = area.copy()
            area['geometry'] = area.geometry.centroid
            pts = gpd.GeoDataFrame(
                pd.concat([pts, area], ignore_index=True),
                crs='EPSG:4326'
            )
        
        pts['category'] = pts.apply(_classify_poi, axis=1)
        result = pts[['geometry', 'category']]
        logger.info('\t%d POI points for %s', len(result), city)
        logger.info('\t= %d nodes + %d area centroids', is_point.sum(), is_area.sum())

        return result

# ========== Helper functions ==========

def _build_osmnx_tags() -> dict:
    tags: dict = {}
    for osm_col, values, _ in _TAG_RULES:
        if osm_col not in tags:
            tags[osm_col] = True
    return tags

def _classify_poi(row) -> str:
    for osm_col, values, category in _TAG_RULES:
        v = row.get(osm_col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if values is True:
            return category
        if str(v) in values:
            return category
    return 'other'

def _fill_zero(df: pd.DataFrame) -> pd.DataFrame:
    for r in _RADII:
        df[f'poi_count_{r}m'] = 0
        df[f'poi_categories_{r}m'] = '{}'
    return df