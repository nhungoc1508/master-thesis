"""
Python port of t2vec's spatial grid/vocabulary construction

Ported from the original Julia preprocessing:
    - SpatialRegionTools.jl -> grid, vocab, V/D
    - utils.jl -> Web-Mercator, corruption
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

UNK = 3
VOCAB_START = 4

# ========== Web Mercator (utils.jl) ==========

def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = np.sin(north)
    return semimajoraxis * east, 3189068.5 * np.log((1 + t) / (1 - t))

def meters2lonlat(x, y):
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = np.exp(y / 3189068.5)
    lat = np.arcsin((t - 1) / (t + 1)) / 0.017453292519943295
    return lon, lat

# ========== SpatialRegion (SpatialRegionTools.jl) ==========

class SpatialRegion:
    """
    SpatialRegion struct + makeVocab! + saveKNearestVocabs + gps2vocab + trip2seq
    """
    def __init__(self, name, minlon, minlat, maxlon, maxlat,
                 xstep, ystep, minfreq, maxvocab_size=50000, k=10,
                 vocab_start=VOCAB_START):
        self.name = name
        self.minlon, self.minlat = minlon, minlat
        self.maxlon, self.maxlat = maxlon, maxlat
        self.minx, self.miny = lonlat2meters(minlon, minlat)
        self.maxx, self.maxy = lonlat2meters(maxlon, maxlat)
        self.xstep, self.ystep = xstep, ystep
        self.numx = int(math.ceil(round(self.maxx - self.minx, 6) / xstep))
        self.numy = int(math.ceil(round(self.maxy - self.miny, 6) / ystep))
        self.minfreq = minfreq
        self.maxvocab_size = maxvocab_size
        self.k = k
        self.vocab_start = vocab_start
        self.vocab_size = vocab_start
        self.hotcell = []
        self.hotcell2vocab = {}
        self.vocab2hotcell = {}
        self._kdtree = None
        self.built = False

    # ----- Coord <-> cell -----

    def coord2cell(self, x, y):
        xoffset = int(math.floor(round(x - self.minx, 6) / self.xstep))
        yoffset = int(math.floor(round(y - self.miny, 6) / self.ystep))
        return yoffset * self.numx + xoffset
    
    def cell2coord(self, cell):
        yoffset = cell // self.numx
        xoffset = cell % self.numx
        y = self.miny + (yoffset + 0.5) * self.ystep
        x = self.minx + (xoffset + 0.5) * self.xstep
        return x, y
    
    def gps2cell(self, lon, lat):
        x, y = lonlat2meters(lon, lat)
        return self.coord2cell(x, y)
    
    def cell2gps(self, cell):
        x, y = self.cell2coord(cell)
        return meters2lonlat(x, y)
    
    def inregion(self, lon, lat):
        return (self.minlon <= lon < self.maxlon and
                self.minlat <= lat < self.maxlat)
    
    # ----- Vocab construction (makeVocab!) -----

    def make_vocab(self, trips):
        cellcount = Counter()
        num_out_region = 0
        for trip in trips:
            for lon, lat in trip:
                if not self.inregion(lon, lat):
                    num_out_region += 1
                else:
                    cellcount[self.gps2cell(lon, lat)] += 1

        max_num_hotcells = min(self.maxvocab_size, len(cellcount))
        topcells = cellcount.most_common(max_num_hotcells)
        self.hotcell = [c for c, cnt in topcells if cnt >= self.minfreq]

        self.hotcell2vocab = {cell: i + self.vocab_start
                              for i, cell in enumerate(self.hotcell)}
        self.vocab2hotcell = {v: c for c, v in self.hotcell2vocab.items()}
        self.vocab_size = self.vocab_start + len(self.hotcell)

        coords = np.array([self.cell2coord(c) for c in self.hotcell], dtype=np.float64)
        self._kdtree = cKDTree(coords)
        self.built = True
        return num_out_region
    
    def knearest_hotcells(self, cell, k):
        assert self.built, 'Build vocab first'
        x, y = self.cell2coord(cell)
        dists, idxs = self._kdtree.query([x, y], k=k)
        idxs = np.atleast_1d(idxs)
        dists = np.atleast_1d(dists)
        return [self.hotcell[i] for i in idxs], dists
    
    def nearest_hotcell(self, cell):
        cells, _ = self.knearest_hotcells(cell, 1)
        return cells[0]

    def cell2vocab(self, cell):
        assert self.built, 'Build vocab first'
        if cell in self.hotcell2vocab:
            return self.hotcell2vocab[cell]
        return self.hotcell2vocab[self.nearest_hotcell(cell)]
    
    def gps2vocab(self, lon, lat):
        if not self.inregion(lon, lat):
            return UNK
        return self.cell2vocab(self.gps2cell(lon, lat))

    def trip2seq(self, trip):
        seq = []
        for lon, lat in trip:
            v = self.gps2vocab(lon, lat)
            if not seq or seq[-1] != v:
                seq.append(v)
        return seq
    
    # ----- k-nearest vocab matrices for spatial-aware loss -----

    def knearest_vocabs(self):
        assert self.built, 'Build vocab first'
        V = np.zeros((self.vocab_size, self.k), dtype=np.int64)
        D = np.zeros((self.vocab_size, self.k), dtype=np.float64)
        for v in range(self.vocab_start):
            V[v, :] = v
            D[v, :] = 0.0
        for v in range(self.vocab_start, self.vocab_size):
            cell = self.vocab2hotcell[v]
            kcells, dists = self.knearest_hotcells(cell, self.k)
            V[v, :] = [self.hotcell2vocab[c] for c in kcells]
            D[v, :len(dists)] = dists
        return V, D
    
# ========== Corruption = downsampling + distort (utils.jl) ==========

def downsampling(trip, rate, rng):
    n = len(trip)
    if n <= 2:
        return trip.copy()
    keep = [0]
    for i in range(1, n - 1):
        if rng.random() > rate:
            keep.append(i)
    keep.append(n - 1)
    return trip[keep].copy()

def distort(trip, rate, rng, radius=50.0):
    out = trip.copy()
    for i in range(len(out)):
        if rng.random() <= rate:
            x, y = lonlat2meters(out[i, 0], out[i, 1])
            xn, yn = 2 * rng.random() - 1, 2 * rng.random() - 1
            normz = math.hypot(xn, yn) or 1.0
            xn, yn = xn * radius / normz, yn * radius / normz
            lon, lat = meters2lonlat(x + xn, y + yn)
            out[i] = [lon, lat]
    return out