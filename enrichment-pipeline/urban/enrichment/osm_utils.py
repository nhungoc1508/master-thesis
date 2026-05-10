"""
Helpers for osmnx downloads

Bbox format: (west, south, east, north) = (left, bottom, right, top)
"""
from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import networkx as nx
import osmnx as ox

logger = logging.getLogger(__name__)

def graph_from_place_or_bbox(
        query: str, city: str,
        city_bboxes: dict[str, tuple[float, float, float, float]],
        network_type: str = 'drive'
) -> nx.MultiDiGraph:
    bbox = city_bboxes.get(city)
    if bbox is not None:
        logger.debug('\tUsing bbox for %s (skipping Nominatim place query)', city)
        return ox.graph_from_bbox(bbox, network_type=network_type, simplify=True)
    return ox.graph_from_place(query, network_type=network_type, simplify=True)