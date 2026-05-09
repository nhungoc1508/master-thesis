"""
Canonical schema and base cleaner for the urban trajectory enrichment pipeline.

Canonical format (one row per GPS point):
    trajectory_id   str     unique identifier across all sources
    point_idx       int     0-based position within the trajectory
    lat             f64     WGS-84 latitude
    lon             f64     WGS-84 longitude
    timestamp       i64     Unix epoch seconds (UTC)
    city            str     lowercase city name
    source          str     dataset name
    transport_mode  str     taxi | delivery | car | motorcycle | bus | unknown

A canonical .parquet file is produced for each dataset.
A lightweight manifest CSV is also written:
    {stem}.manifest.csv     one row per trajectory with bbox stats and
                            start/end timestamps. Used by run_clean.py
                            (grid sampling) and run_split.py (chronological
                            split) so neither script needs to load the full
                            heavy .parquet just for metadata.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.geo import haversine_vectorized, bbox_diagonal_m

logger = logging.getLogger(__name__)

# ================= CANONICAL SCHEMA =================

CANONICAL_SCHEMA = pa.schema([
    pa.field('trajectory_id',  pa.string(),  nullable=False),
    pa.field('point_idx',      pa.int32(),   nullable=False),
    pa.field('lat',            pa.float64(), nullable=False),
    pa.field('lon',            pa.float64(), nullable=False),
    pa.field('timestamp',      pa.int64(),   nullable=False),
    pa.field('city',           pa.string(),  nullable=True),
    pa.field('source',         pa.string(),  nullable=True),
    pa.field('transport_mode', pa.string(),  nullable=True),
])

# ================= MANIFEST SCHEMA =================

MANIFEST_COLUMNS = [
    'trajectory_id', 'n_points',
    'lat_mid', 'lon_mid',
    'start_ts', 'end_ts'
]

# ================= QUALITY FILTER CONFIGURATIONS =================

@dataclass
class QualityConfig:
    min_points:         int   = 10
    min_duration_s:     float = 60.0
    max_speed_kmh:      float = 180.0
    min_displacement_m: float = 100.0

# ================= BASE CLEANER CLASS =================

class BaseCleaner(ABC):
    """
    Abstract base class for all cleaners.
    Use streaming as much as possible to avoid heavy RAM usage.
    Implemented here:
        streaming I/O
        manifest handling
    To be implemented in subclasses:
        `iter_raw`: yield raw trajectory dicts
    """

    source:         str = 'unknown'
    city:           str = 'unknown'
    transport_mode: str = 'unknown'

    # How many GPS points to buffer before flushing a parquet row-group
    WRITE_BATCH_PTS: int = 50_000

    def __init__(self, config: QualityConfig | None = None):
        self.cfg = config or QualityConfig()
        self.log = logging.getLogger(self.__class__.__name__)

    # ---------- Subclass interface ----------

    @abstractmethod
    def iter_raw(self, data_path: Path) -> Iterator[dict]:
        """
        Yield raw trajectory dicts with mandatory fields:
            lats:          list[float]
            lons:          list[float]
            timestamps:    list[int]
            trajectory_id: str
        Optional fields: city, source, transport_mode
        """

    # ---------- Public entry point ----------

    def run(self, data_path: Path, output_path: Path) -> int:
        """
        Clean data_path -> canonical parquet + manifest CSV
        In streaming mode
        Returns total GPS point count written to disk
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = output_path.with_suffix('').with_suffix('.manifest.csv')

        writer: pq.ParquetWriter | None = None
        batch_pts: list[dict] = []

        # Track manifest as parallel lists
        m_ids:      list[str]   = []
        m_n_pts:    list[int]   = []
        m_lat_mid:  list[float] = []
        m_lon_mid:  list[float] = []
        m_start_ts: list[int]   = []
        m_end_ts:   list[int]   = []

        total = kept = n_points = 0

        for raw in self.iter_raw(data_path):
            total += 1
            pts = self._validate_and_filter(raw)
            if pts is None:
                continue

            kept += 1
            n_pts = len(pts)
            n_points += n_pts

            # Add manifest entry (one per trajectory)
            lats = [p['lat'] for p in pts]
            lons = [p['lon'] for p in pts]
            m_ids.append(pts[0]['trajectory_id'])
            m_n_pts.append(n_pts)
            m_lat_mid.append((min(lats) + max(lats)) / 2)
            m_lon_mid.append((min(lons) + max(lons)) / 2)
            m_start_ts.append(pts[0]['timestamp'])
            m_end_ts.append(pts[-1]['timestamp'])

            # Streaming parquet write
            batch_pts.extend(pts)
            if len(batch_pts) >= self.WRITE_BATCH_PTS:
                writer = _flush(batch_pts, writer, output_path, CANONICAL_SCHEMA)
                batch_pts = []

        if batch_pts:
            writer = _flush(batch_pts, writer, output_path, CANONICAL_SCHEMA)

        if writer:
            writer.close()

        self.log.info(
            '%s: %d/%d trajectories kept = %d points',
            self.source, kept, total, n_points,
        )

        # Write manifest
        pd.DataFrame({
            'trajectory_id': m_ids,
            'n_points':      m_n_pts,
            'lat_mid':       m_lat_mid,
            'lon_mid':       m_lon_mid,
            'start_ts':      m_start_ts,
            'end_ts':        m_end_ts,
        }).to_csv(manifest_path, index=False)
        self.log.info('Manifest written: %s (%d rows)', manifest_path, kept)

        return n_points

    # ---------- Quality filter ----------

    def _validate_and_filter(self, raw: dict) -> list[dict] | None:
        lats = np.asarray(raw['lats'], dtype=np.float64)
        lons = np.asarray(raw['lons'], dtype=np.float64)
        ts = np.asarray(raw['timestamps'], dtype=np.int64)
        n = len(lats)

        # Filter: min points
        if n != len(lons) or n != len(ts) or n < self.cfg.min_points:
            return None
        
        order = np.argsort(ts, stable=True)
        lats, lons, ts = lats[order], lons[order], ts[order]

        # Filter: min duration
        if float(ts[-1] - ts[0]) < self.cfg.min_duration_s:
            return None
        
        # Filter: max speed
        if n > 1:
            dt = np.diff(ts).astype(float)
            dt = np.where(dt <= 0, 1e-3, dt)
            dist = haversine_vectorized(lats[:-1], lons[:-1], lats[1:], lons[1:])
            if (dist / dt * 3.6).max() > self.cfg.max_speed_kmh:
                return None
            
        # Filter: min displacement
        if bbox_diagonal_m(lats, lons) < self.cfg.min_displacement_m:
            return None
        
        tid = raw['trajectory_id']
        city = raw.get('city', self.city)
        src = raw.get('source', self.source)
        mode = raw.get('transport_mode', self.transport_mode)

        return [
            {
                'trajectofy_id':  tid,
                'point_idx':      int(i),
                'lat':            float(lats[i]),
                'lon':            float(lons[i]),
                'timestamp':      int(ts[i]),
                'city':           city,
                'source':         src,
                'transport_mode': mode
            }
            for i in range(n)
        ]
    
# ================= HELPERS =================

def _flush(
        batch:       list[dict],
        writer:      pq.ParquetWriter | None,
        output_path: Path,
        schema:      pa.Schema,
) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(batch, schema=schema)
    if writer is None:
        writer = pq.ParquetWriter(output_path, schema)
        writer.write_table(table)
        return writer