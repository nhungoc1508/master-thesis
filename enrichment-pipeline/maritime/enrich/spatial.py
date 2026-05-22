"""
Spatial feature enrichment

Wraps Archimedes modules whenever possible

Columns added:
    nearest_port_nm         float       distance to nearest port in nautical miles (using World Port Index)
    nearest_port_name       str         name of nearest port (WPI)
    port_proximity_label    str         berthed | port_approach | coastal | offshort | open_anchorage
    in_port_zone            bool        true if point falls inside an OSM harbor polygon
    in_mpa                  bool        true if point is inside a Marine Protected Area polygon
    mpa_name                str         name of enclosing MPA or empty
    in_tss                  bool        true if point is inside a Traffic Separation Scheme zone
    tss_name                str         name of the TSS zone or empty
    sea_area_name           str         IHO sea area name
    eez_country_iso         str         ISO 3166 country code of the enclosing EEZ
    in_territorial_sea      bool        true if within 12 nautical miles of any baseline (EEZ territorial sea)

Port proximity hierarchy:
    1. OSM harbor polygon -> in_port_zone=True; berthed if nearest_port_nm < berthed_nm
    2. WPI radius buffer -> port_approach if nearest_port_nm < approach_nm
    3. coastal if nearest_port_nm < coastal_nm; otherwise offshore
    4. STOP outside harbor polygon and beyond approach_nm -> open_anchorage
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

logger = logging.getLogger(__name__)

class SpatialEnricher:
    def __init__(self, context: dict, port_cfg: dict, context_cfg: dict):
        """
        Params:
            context: loaded dict from utils.archimedes.load_context()
            port_cfg: dict from cfg['enrich']['port_proximity']
            context_cfg: dict from cfg['enrich']['context']
        """
        self.ctx = context
        self.berthed_nm = float(port_cfg.get('berthed_nm', 0.5))
        self.approach_nm = float(port_cfg.get('approach_nm', 5.0))
        self.coastal_nm = float(port_cfg.get('coastal_nm', 20.0))

        self._harbor_gdf = self._load_shp(context_cfg.get('harbour_polygons_shp'))
        self._eez_gdf = self._load_gpkg(context_cfg.get('eez_gpkg'))
        self._sea_areas_gdf = self._load_shp(context_cfg.get('sea_areas_shp'))

        import spatial_context as sc
        self._sc = sc

    def _load_shp(self, path) -> gpd.GeoDataFrame | None:
        if not path or not Path(path).exists():
            logger.warning('.shp is not found: %s', path)
            return None
        try:
            gdf = gpd.read_file(path)
            gdf.sindex
            logger.info('.shp loaded: %s', path)
            return gdf
        except Exception as exc:
            logger.warning('Could not load .shp %s: %s', path, exc)
            return None
        
    def _load_gpkg(self, path, layer=None) -> gpd.GeoDataFrame | None:
        if not path or not Path(path).exists():
            logger.warning('.gpkg is not found: %s', path)
            return None
        try:
            if layer:
                gdf = gpd.read_file(path, layer=layer)
            else:
                gdf = gpd.read_file(path)
            gdf.sindex
            logger.info('.gpkg loaded: %s', path)
            return gdf
        except Exception as exc:
            logger.warning('Could not load .gpkg %s: %s', path, exc)
            return None
        
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)

        pts_gdf = gpd.GeoDataFrame(
            {'_pos': np.arange(n)},
            geometry=gpd.points_from_xy(df['lon'].values, df['lat'].values),
            crs='EPSG:4326'
        )

        nearest_port_nm = np.full(n, np.nan)
        nearest_port_name = np.full(n, '', dtype=object)
        in_port_zone = np.zeros(n, dtype=bool)
        in_mpa = np.zeros(n, dtype=bool)
        mpa_name = np.full(n, '', dtype=object)
        in_tss = np.zeros(n, dtype=bool)
        tss_name = np.full(n, '', dtype=object)
        sea_area_name = np.full(n, '', dtype=object)
        eez_iso = np.full(n, 'high_seas', dtype=object)
        in_terr_sea = np.zeros(n, dtype=bool)

        def _sjoin_within(poly_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
            result = gpd.sjoin(pts_gdf, poly_gdf, how='left', predicate='within')
            return result.drop_duplicates(subset='_pos', keep='first')
        
        # ========== Nearest port ==========
        logger.info('\tSpatial enriching: nearest port')
        ports = self.ctx.get('ports')
        if ports is not None and isinstance(ports, gpd.GeoDataFrame):
            logger.info('\t\tUsing gdf for ports')
            try:
                joined = gpd.sjoin_nearest(
                    pts_gdf, ports, how='left', distance_col='_dist_deg'
                ).drop_duplicates(subset='_pos', keep='first')
                nm_arr = joined['_dist_deg'].values * 60.0 # 1 deg lat ~ 60 nm
                nearest_port_nm[:] = np.where(np.isfinite(nm_arr), np.round(nm_arr, 2), np.nan)
                name_col = next(
                    (c for c in ['NAME', 'name', 'Port_Name', 'port_name'] if c in joined.columns),
                    None
                )
                if name_col:
                    nearest_port_name[:] = joined[name_col].fillna('').values
            except Exception as exc:
                logger.warning('\t\tNearest port vectorization failed, skipping: %s', exc)
        elif ports is not None:
            logger.warning('\t\tPorts context is not a gdf, falling back to per-point')
            for i, (_, row) in enumerate(df.iterrows()):
                try:
                    point = Point(row['lon'], row['lat'])
                    name, dist_deg = self._sc.find_nearest_port(ports, point)
                    nearest_port_nm[i] = round(dist_deg * 60.0, 2)
                    nearest_port_name[i] = name or ''
                except Exception as exc:
                    logger.warning('\t\tFailed to get nearest port (per-point): %s', exc)
                    pass

        # ========== Harbor polygon containment (in_port_zone) ==========
        logger.info('\tSpatial enriching: in port zone')
        if self._harbor_gdf is not None:
            try:
                joined = _sjoin_within(self._harbor_gdf[['geometry']])
                hit_pos = joined.dropna(subset=['index_right'])['_pos'].values
                in_port_zone[hit_pos.astype(int)] = True
            except Exception as exc:
                logger.warning('\t\tHarbor sjoin failed: %s', exc)
        else:
            logger.warning('\t\tHarbor gdf not available')

        # ========== Port proximity label ==========
        logger.info('\tSpatial enriching: port proximity')
        anno = df.get('annotation', pd.Series('', index=df.index))
        is_stop = (anno.str.contains('STOP_START', na=False) |
                   anno.str.contains('STOP_END', na=False)).values
        nm = np.where(np.isnan(nearest_port_nm), 9999.0, nearest_port_nm)

        port_prox_label = np.full(n, 'offshort', dtype=object)
        port_prox_label[is_stop & ~in_port_zone & (nm >= self.approach_nm)] = 'open_anchorage'
        port_prox_label[nm < self.coastal_nm] = 'coastal'
        port_prox_label[nm < self.approach_nm] = 'port_approach'
        port_prox_label[in_port_zone & (nm < self.berthed_nm)] = 'berthed'

        # ========== MPA check ==========
        logger.info('\tSpatial enriching: MPA check')
        protected = self.ctx.get('protected_areas')
        if protected is not None and isinstance(protected, gpd.GeoDataFrame):
            try:
                cols = ['geometry'] + [c for c in ['name_int'] if c in protected.columns]
                joined = _sjoin_within(protected[cols])
                hits = joined.dropna(subset=['index_right'])
                pos = hits['_pos'].values.astype(int)
                in_mpa[pos] = True
                if 'name_int' in hits.columns:
                    mpa_name[pos] = hits['name_int'].fillna('').values
            except Exception as exc:
                logger.warning('\t\tMPA sjoin failed: %s', exc)
        else:
            logger.warning('\t\tProtected area gdf not available')

        # ========== TSS check ==========
        logger.info('\tSpatial enriching: TSS check')
        zones = self.ctx.get('separation_zones')
        if zones is not None and isinstance(zones, gpd.GeoDataFrame):
            try:
                cols = ['geometry'] + [c for c in ['name_int'] if c in zones.columns]
                joined = _sjoin_within(zones[cols])
                hits = joined.dropna(subset=['index_right'])
                pos = hits['_pos'].values.astype(int)
                in_tss[pos] = True
                if 'name_int' in hits.columns:
                    tss_name[pos] = hits['name_int'].fillna('').values
            except Exception as exc:
                logger.warning('\t\tTSS sjoin failed: %s', exc)
        else:
            logger.warning('\t\tTSS gdf not available')

        # ========== Sea area name ==========
        logger.info('\tSpatial enriching: sea area name')
        if self._sea_areas_gdf is not None:
            try:
                name_col = next(
                    (c for c in ['NAME', 'name', 'name_int'] if c in self._sea_areas_gdf.columns),
                    None
                )
                sea_cols = ['geometry'] + ([name_col] if name_col else [])
                joined = _sjoin_within(self._sea_areas_gdf[sea_cols])
                hits = joined.dropna(subset=['index_right'])
                if name_col:
                    pos = hits['_pos'].values.astype(int)
                    sea_area_name[pos] = hits[name_col].fillna('').values
            except Exception as exc:
                logger.warning('\t\tSea area sjoin failed: %s', exc)
        else:
            logger.warning('\t\tSea area gdf not available')
        
        # ========== EEZ lookup ==========
        logger.info('\tSpatial enriching: EEZ lookup')
        if self._eez_gdf is not None:
            try:
                eez_extra = [c for c in ['ISO_SOV1', 'SOVEREIGN1', 'POL_TYPE']
                             if c in self._eez_gdf.columns]
                joined = _sjoin_within(self._eez_gdf[['geometry'] + eez_extra])
                hits = joined.dropna(subset=['index_right'])
                pos = hits['_pos'].values.astype(int)
                if 'ISO_SOV1' in hits.columns:
                    iso_vals = hits['ISO_SOV1'].fillna('').str[:3].values
                elif 'SOVEREIGN1' in hits.columns:
                    iso_vals = hits['SOVEREIGN1'].fillna('').str[:3].values
                else:
                    iso_vals = np.full(len(hits), '', dtype=object)
                eez_iso[pos] = np.where(iso_vals == '', 'high_seas', iso_vals)
                if 'POL_TYPE' in hits.columns:
                    pol = hits['POL_TYPE'].fillna('')
                    in_terr_sea[pos] = (pol.str.contains('12NM', na=False) |
                                        pol.str.contains('Territorial', na=False)).values
            except Exception as exc:
                logger.warning('\t\tEEZ sjoin failed: %s', exc)
        else:
            logger.warning('\t\tEEZ gdf is not available')

        # ========== Assign results ==========
        df['nearest_port_nm'] = nearest_port_nm
        df['nearest_port_name'] = nearest_port_name
        df['port_proximity_label'] = port_prox_label
        df['in_port_zone'] = in_port_zone
        df['in_mpa'] = in_mpa
        df['mpa_name'] = mpa_name
        df['in_tss'] = in_tss
        df['tss_name'] = tss_name
        df['sea_area_name'] = sea_area_name
        df['eez_country_iso'] = eez_iso
        df['in_territorial_sea'] = in_terr_sea

        mask = df['sea_area_name'].str.strip().astype(bool)
        df.loc[mask, 'city'] = df.loc[mask, 'sea_area_name'].str.lower().str.replace(' ', '_')

        return df