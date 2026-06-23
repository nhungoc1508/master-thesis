"""
Transforming frozen benchmark to make data compatible with t2vec's expected inputs

Frozen benchmark:
    - Trajectories are stored as per-trajectory normalized offsets + denorm parameters
    - t2vec uses absolute lon/lat
    - Steps:
        - Denormalize benchmark data
        - Build per-dataset SpatialRegion (region bbox from each dataset's extent)
        - Tokenize to hot cell vocab sequence
        - Produce (src, trg) pairs for denoising objective: src = downsampled & distorted;
            trg = clean using original code
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2])) # benchmark/ -> get benchmark_dataset.py, metrics.py

from bench_dataset import BenchmarkDataset
from region import SpatialRegion, downsampling, distort

def item_to_abs(item):
    """One frozen item -> (traj_len, 2) absolute [lon, lat] array"""
    traj_len = int(item['traj_len'])
    coords = item['coords'][:traj_len].numpy()
    denorm = item['denorm']
    lat = coords[:, 0] * denorm['bbox_half'] + denorm['lat0']
    lon = coords[:, 1] * denorm['bbox_half'] + denorm['lon0']
    return np.stack([lon, lat], axis=1)

def load_abs_trajs(units, task):
    """
    Return (abs_trajs, items) for a list of frozen unit dirs

    Args:
        abs_trajs: list of (n, 2) absolute [lon, lat] arrays
        items: raw frozen items (carry pos_mask, denorm, coords, domain, dataset)
    """
    ds = BenchmarkDataset(units, task=task, with_sem=False)
    abs_trajs, items = [], []
    for i in range(len(ds)):
        item = ds[i]
        abs_trajs.append(item_to_abs(item))
        items.append(item)
    return abs_trajs, items

def build_region(abs_trajs, name, cellsize_m, minfreq, k, pad_deg=0.01, maxvocab_size=50000):
    """Build & populate a SpatialRegion from the dataset's own coordinate extent"""
    all_pts = np.concatenate(abs_trajs, axis=0)
    min_lon, min_lat = all_pts.min(axis=0)
    max_lon, max_lat = all_pts.max(axis=0)
    region = SpatialRegion(
        name=name,
        minlon=float(min_lon) - pad_deg, minlat=float(min_lat) - pad_deg,
        maxlon=float(max_lon) + pad_deg, maxlat=float(max_lat) + pad_deg,
        xstep=cellsize_m, ystep=cellsize_m,
        minfreq=minfreq, maxvocab_size=maxvocab_size, k=k
    )
    n_out = region.make_vocab(abs_trajs)
    return region, n_out

def tokenize_pairs(region, abs_trajs, rng, drop_rate=0.3, distort_rate=0.0):
    """
    Create (src, trg) token-id sequences for denoising objective

    trg = trip2seq(clean)
    src = trip2seq(downsample & distort)
    """
    src_list, trg_list = [], []
    for trip in abs_trajs:
        trg = region.trip2seq(trip)
        noisy = downsampling(trip, drop_rate, rng)
        if distort_rate > 0:
            noisy = distort(noisy, distort_rate, rng)
        src = region.trip2seq(noisy)
        if len(trg) < 2 or len(src) < 2:
            continue
        src_list.append(np.array(src, dtype=np.int64))
        trg_list.append(np.array(trg, dtype=np.int64))
    return src_list, trg_list