"""
Helpers for using Archimedes' Python modules.
Original repository: https://github.com/M3-Archimedes/AIS-semantic-trajectories
"""
from __future__ import annotations
import sys
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

def setup_path(python_src: str | Path) -> None:
    """Prepend the src directory to sys.path"""
    src = str(Path(python_src).resolve())
    if src not in sys.path:
        sys.path.insert(0, src)

def load_context(cfg: dict) -> dict:
    """Load all geospatial context data"""
    import spatial_context
    import netcdf_context

    logger = logging.getLogger(__name__)
    context: dict = {}

    def _try_load(key: str, loader, *args, **kwargs):
        try:
            context[key] = loader(*args, **kwargs)
            logger.info('Loaded context: %s', key)
        except Exception as exc:
            context[key] = None
            logger.warning('Could not load context "%s": %s', key, exc)

    if cfg.get('ports_csv'):
        _try_load('ports', spatial_context.read_csv_dataset,
                  cfg['ports_csv'], col_x='LON', col_y='LAT',
                  crs='epsg:4326', col_name='NAME', sep=',')
        
    if cfg.get('placemarks_shp'):
        _try_load('placemarks', spatial_context.read_osm_shp_dataset,
                  cfg['placemarks_shp'])
        
    if cfg.get('capes_shp'):
        _try_load('capes', spatial_context.read_osm_shp_dataset,
                  cfg['capes_shp'])
        
    if cfg.get('protected_areas_shp'):
        _try_load('protected_areas', spatial_context.read_osm_shp_dataset,
                  cfg['protected_areas_shp'])
        
    if cfg.get('separation_zones_shp'):
        _try_load('separation_zones', spatial_context.read_osm_shp_dataset,
                  cfg['separation_zones_shp'])

    if cfg.get('gebco_nc_dir'):
        _try_load('bytho_netcdf', netcdf_context.read_netcdf_data,
                  cfg['gebco_nc_dir'])
        
    context['meteo_netcdf'] = None

    return context