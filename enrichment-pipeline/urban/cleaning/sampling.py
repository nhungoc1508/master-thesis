"""
Grid-based trajectory sampling.

Algorithm:
----------
1. For each trajectory, compute the midpoint of the bounding box: (lat_mid, lon_mid)
2. Assign each trajectory to a grid cell on an RxC grid over the dataset bounding bpx
3. Compute per-cell quota: floor(n / n_occupied_cells), distribute remainder to the densest cells
4. From each cell, randomly sample min(cell_size, quota) trajectories
5. If the total sampled < n (sparse cells), top up by re-sampling from the densest cells
"""
from __future__ import annotations
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def grid_sample_ids(
        manifest: pd.DataFrame,
        n: int,
        grid_rows: int = 20,
        grid_cols: int = 20,
        seed: int = 42,
) -> set[str]:
    """
    Return a set of trajectory_ids selected by grid-based sampling

    Params:
        manifest    : manifest DataFrame with columns: trajectory_id, lat_mid, lon_mid
        n           : number of trajectories to sample
        grid_rows   : number of grid rows (latitude axis)
        grid_cols   : number of grid columns (longitude axis)
        seed        : random seed for reproducibility
    """
    n_total = len(manifest)
    if n_total <= n:
        logger.info(
            'Dataset as %d trajectories, <= target %d, no sampling needed',
            n_total, n
        )
        return set(manifest['trajectory_id'].tolist())

    logger.info(
        'Grid sampling: %d -> %d trajectories (%dx%d grid, seed=%d)',
        n_total, n, grid_rows, grid_cols, seed,
    )

    agg = manifest[['trajectory_id', 'lat_mid', 'lon_mid']].copy()

    # ---------- Assign grid cells ----------
    lat_min, lat_max = agg['lat_mid'].min(), agg['lat_mid'].max()
    lon_min, lon_max = agg['lon_mid'].min(), agg['lat_mid'].max()

    eps = 1e-9
    agg['row'] = np.floor(
        (agg['lat_mid'] - lat_min) / (lat_max - lat_min + eps) * grid_rows
    ).astype(int).clip(0, grid_rows - 1)

    agg['col'] = np.floor(
        (agg['lon_mid'] - lon_min) / (lon_max - lon_min + eps) * grid_cols
    ).astype(int).clip(0, grid_cols - 1)

    agg['cell'] = agg['row'] * grid_cols + agg['col']

    # ---------- Per-cell quota ----------
    cell_counts = agg['cell'].value_counts().sort_index()
    occupied_cells = cell_counts.index.tolist()
    n_cells = len(occupied_cells)

    base_quota = n // n_cells
    remainder = n % n_cells

    top_cells = cell_counts.nlargest(remainder).index
    quotas = {
        cell: base_quota + (1 if cell in top_cells else 0)
        for cell in occupied_cells
    }

    # ---------- Sample per-cell ----------
    rng = np.random.default_rng(seed)
    sampled_ids: list[str] = []

    for cell, quota in quotas.items():
        cell_trajs = agg.loc[agg['cell'] == cell, 'trajectory_id'].values
        take = min(len(cell_trajs), quota)
        chosen = rng.choice(cell_trajs, size=take, replace=False)
        sampled_ids.extend(chosen)

    # ---------- Fill shortfall using dense cells ----------
    shortfall = n - len(sampled_ids)
    if shortfall > 0:
        already_sampled = set(sampled_ids)
        candidates = agg[~agg['trajectory_id'].isin(already_sampled)]
        candidates = candidates.merge(
            cell_counts.rename('cell_size'), left_on='cell', right_index=True
        ).sort_values('cell_size', ascending=False)
        extra = candidates['trajectory_id'].values[:shortfall]
        sampled_ids.extend(extra)
        logger.debug('Filled %d shortfall trajectories from dense cells', len(extra))
    
    logger.info('\tOccupied grid cells: %d / %d', n_cells, grid_rows * grid_cols)
    logger.info('\tTrajectories sampled: %d (target %d)', len(sampled_ids), n)

    return set(sampled_ids)