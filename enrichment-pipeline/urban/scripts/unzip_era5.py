#!/usr/bin/env python3
"""
Unzip ERA5 .zip files one-by-one, rename each extracted .nc file to
YYYY-mm.nc based on time, then remove original .zip file.

Usage:
    python scripts/unzip_era5.py --dir path/to/zips
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def _read_year_month(nc_path: Path) -> tuple[int, int]:
    """Return (year, month) from the first time value found in netCDF4 file"""
    ds = xr.open_dataset(nc_path, engine='netcdf4')
    try:
        time_coord = next(
            (c for c in ('valid_time', 'time') if c in ds),
            None
        )
        if time_coord is None:
            raise ValueError(f'No time coordinate found. Variables: {list(ds)}')
        time_vals = ds[time_coord].values
        if len(time_vals) == 0:
            raise ValueError(f'{time_coord!r} coordinate is empty')
        ts = pd.Timestamp(time_vals[0])
        return ts.year, ts.month
    finally:
        ds.close()

def _process_zip(zip_path: Path, out_dir: Path) -> Path | None:
    """
    Extract .zip into a temp dir, read (year, month), rename .nc to YYYY-mm.nc
    in out_dir, delete the zip
    Return the final .nc path or None on errpr
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        with zipfile.ZipFile(zip_path) as zf:
            nc_names = [n for n in zf.namelist() if n.endswith('.nc')]
            if not nc_names:
                logger.warning('No .nc files in %s, skipping', zip_path.name)
                return None
            if len(nc_names) > 1:
                logger.warning('%s contains %d .nc files, using first: %s',
                               zip_path.name, len(nc_names), nc_names[0])
            zf.extract(nc_names[0], path=tmp_path)
            extracted = tmp_path / nc_names[0]

        year, month = _read_year_month(extracted)
        dest = out_dir / f'{year}-{month:02d}.nc'

        if dest.exists():
            logger.warning('Destination %s already exists, overwriting', dest.name)
        shutil.move(str(extracted), dest)
    
    zip_path.unlink()
    logger.info('\t%s -> %s (zip removed)', zip_path.name, dest.name)
    return dest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()

    target = Path(args.dir).resolve()
    if not target.is_dir():
        sys.exit(f'Not a directory: {target}')

    zips = sorted(target.glob('*.zip'))
    if not zips:
        sys.exit(f'No .zip files found in {target}')
    
    logger.info('Found %d .zip files in %s', len(zips), target)

    ok, failed = 0, 0
    for zp in zips:
        try:
            result = _process_zip(zp, target)
            if result:
                ok += 1
            else:
                failed += 1
        except Exception as exc:
            logger.error('\tFAILED %s: %s', zp.name, exc)
            failed += 1
    
    logger.info('Done: %d succeeded, %d failed', ok, failed)

if __name__ == '__main__':
    main()