"""
Masking strategies for trajectory MAE pre-training

All functions take a sequence length n and return a boolean mask array
of shape (n,) where True = masked (to be reconstructed), False = visible
"""
from __future__ import annotations

import numpy as np

_MODES = ['spatial_random', 'block', 'key_point', 'kinematic_group', 'semantic_group', 'last_n']
_PROBS = {
    'urban': [0.25, 0.20, 0.15, 0.15, 0.05, 0.20],
    'maritime': [0.25, 0.20, 0.15, 0.15, 0.05, 0.20]
}

def random_mask(n: int, r: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    mask = np.zeros(n, dtype=bool)
    k = max(1, round(r * n))
    idxs = rng.choice(n, k, replace=False)
    mask[idxs] = True
    return mask

def last_n_mask(n: int, max_n: int = 10, frac: float = 0.15) -> np.ndarray:
    k = min(max_n, max(1, round(frac * n)))
    mask = np.zeros(n, dtype=bool)
    mask[n - k:] = True
    return mask

def block_mask(n: int, min_frac: float = 0.20, max_frac: float = 0.30,
               rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    mask = np.zeros(n, dtype=bool)
    frac = rng.uniform(min_frac, max_frac)
    block_size = max(1, round(frac * n))
    start = int(rng.integers(0, max(1, n - block_size + 1)))
    mask[start:start + block_size] = True
    return mask

def key_point_mask(coords: np.ndarray, frac: float = 0.15, epsilon: float = 0.001) -> np.ndarray:
    """Mask key structural points identified by Ramer-Douglas-Peucker"""
    n = len(coords)
    if n < 3:
        return np.zeros(n, dtype=bool)
    deviations = _point_deviations(coords)
    k = max(1, round(frac * n))
    threshold = np.partition(deviations, -k)[-k]
    mask = deviations >= threshold
    return mask

def _point_deviations(coords: np.ndarray) -> np.ndarray:
    """Perpendicular deviation of each interior point from the line start->end"""
    n = len(coords)
    devs = np.zeros(n)
    if n < 3:
        return devs
    start, end = coords[0], coords[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        devs[1:-1] = np.linalg.norm(coords[1:-1] - start, axis=1)
        return devs
    line_unit = line_vec / line_len
    for i in range(1, n - 1):
        v = coords[i] - start
        proj = np.dot(v, line_unit)
        nearest = start + proj * line_unit
        devs[i] = np.linalg.norm(coords[i] - nearest)
    return devs

def sample_mode(domain: str = 'urban', rng: np.random.Generator | None = None) -> str:
    """Sample one masking mode for a batch using domain-specific probabilities"""
    rng = rng or np.random.default_rng()
    probs = _PROBS[domain]
    return str(rng.choice(_MODES, p=probs))

def make_pos_mask(mode: str, n: int, coords: np.ndarray | None = None,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Build a (n,) bool position mask for one trajectory given a batch mode"""
    rng = rng or np.random.default_rng()
    if mode == 'block':
        return block_mask(n, rng=rng)
    if mode == 'key_point':
        if coords is not None and n >= 3:
            return key_point_mask(coords, frac=0.15)
        return random_mask(n, r=0.15, rng=rng)
    if mode == 'last_n':
        return last_n_mask(n, max_n=10, frac=0.15)
    return random_mask(n, r=0.5, rng=rng)