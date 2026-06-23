"""
Transforming frozen benchmark to make data compatible with TrajCL's expected inputs

Process:
    - Denormalize frozen offsets -> absolute lon/lat -> Mercator
    - Build per-dataset CellSpace
    - Build cell-adjacent graph
    - Run TrajCL's train_node2vec -> get cell embeddings
"""
from __future__ import annotations

import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent / 'vendor'))
sys.path.insert(0, str(_HERE.parents[2]))

from bench_dataset import BenchmarkDataset
from utils.cellspace import CellSpace

def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = np.sin(north)
    return semimajoraxis * east, 3189068.5 * np.log((1 + t) / (1 - t))

def item_to_mercator(item) -> np.ndarray:
    """Frozen item -> (traj_len, 2) Mercator [x, y]"""
    traj_len = int(item['traj_len'])
    coords = item['coords'][:traj_len].numpy()
    denorm = item['denorm']
    lat = coords[:, 0] * denorm['bbox_half'] + denorm['lat0']
    lon = coords[:, 1] * denorm['bbox_half'] + denorm['lon0']
    x, y = lonlat2meters(lon, lat)
    return np.stack([x, y], axis=1)

def load_mercator_trajs(units, task):
    ds = BenchmarkDataset(units, task=task, with_sem=False)
    trajs, items = [], []
    for i in range(len(ds)):
        item = ds[i]
        trajs.append(item_to_mercator(item))
        items.append(item)
    return trajs, items

def load_mercator_by_dataset(units, task):
    ds = BenchmarkDataset(units, task=task, with_sem=False)
    groups = defaultdict(lambda: {'trajs': [], 'items': []})
    for i in range(len(ds)):
        item = ds[i]
        g = groups[item['dataset']]
        g['trajs'].append(item_to_mercator(item))
        g['items'].append(item)
    return dict(groups)

def build_cellspace(merc_trajs, cell_size, buffer=500.0) -> CellSpace:
    """CellSpace over the dataset's Mercator extent (+ buffer)"""
    allpts = np.concatenate(merc_trajs, axis=0)
    x_min, y_min = allpts.min(axis=0) - buffer
    x_max, y_max = allpts.max(axis=0) + buffer
    return CellSpace(cell_size, cell_size, float(x_min), float(y_min), float(x_max), float(y_max))

def build_edge_index(cellspace: CellSpace) -> torch.Tensor:
    """Cell-adjacency graph (bidirectional) -> edge_index [2, E] for node2vec"""
    _, pair_ids = cellspace.all_neighbour_cell_pairs_permutated_optmized()
    e = np.array(pair_ids, dtype=np.int64).T
    e = np.concatenate([e, e[::-1]], axis=1)
    return torch.from_numpy(e)

def build_cell_embeddings(cellspace, cfg, device, tag='frozen') -> torch.Tensor:
    """Run TrajCL's train_node2vec on the cell graph, return embs tensor (num_cells, cell_embedding_dim).

    node2vec embeddings depend only on the cell-adjacency grid (pure geometry), not on which cells
    are visited -> safe to build the grid over train+test extent with no train/test leakage.
    """
    from config import Config
    from model.node2vec_ import train_node2vec
    safe = re.sub(r'\W+', '_', str(tag))            # dataset name -> filesystem-safe checkpoint tag
    ckpt_dir = _HERE.parent / '_node2vec_tmp'
    ckpt_dir.mkdir(exist_ok=True)
    Config.device = device
    Config.cell_size = float(cfg.cell_size)
    Config.cell_embedding_dim = int(cfg.emb_dim)
    Config.node2vec_epochs = int(getattr(cfg, 'node2vec_epochs', 20)) # [perf-config]
    Config.node2vec_batch_size = int(getattr(cfg, 'node2vec_batch', 256)) # [perf-config]
    Config.node2vec_workers = int(getattr(cfg, 'node2vec_workers', 8)) # [perf-config]
    Config.checkpoint_dir = str(ckpt_dir)
    Config.dataset_prefix = safe
    Config.dataset_embs_file = str(ckpt_dir / f'{safe}_embs.pkl')
    edge_index = build_edge_index(cellspace).to(device)
    train_node2vec(edge_index)
    with open(Config.dataset_embs_file, 'rb') as fh:
        embs = pickle.load(fh)
    return embs.detach().to('cpu')