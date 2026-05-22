"""
Bathymetry feature enrichment

Wraps Archimedes' netcdf_context.get_depth_at_location() and calls per-point

Column added:
    water_depth_m   float   sea depth in meters, None/NaN if outside the GEBCO coverage grid
                            Stored as positive depth values
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

class BathymetryEnricher:
    def __init__(self, bytho_dataset):
        """
        Params:
            batho_dataset: xarray Dataset loaded by netcdf_context.read_netcdf_data()
                           Expected variable: 'elevation' (negative = below sea level)
        """
        self._ds = bytho_dataset

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._ds is None:
            logger.warning('No bathymetry dataset loaded; water_depth_m set to NaN')
            df = df.copy()
            df['water_depth_m'] = np.nan
            return df
        
        df = df.copy()
        lats = xr.DataArray(df['lat'].values, dims='points')
        lons = xr.DataArray(df['lon'].values, dims='points')

        elevs = self._ds['elevation'].sel(lat=lats, lon=lons, method='nearest').values
        # Negative elevation = below sea level -> store as positive depth
        depths = np.where(elevs < 0, np.around(np.abs(elevs), 1), np.nan)

        df['water_depth_m'] = depths
        return df