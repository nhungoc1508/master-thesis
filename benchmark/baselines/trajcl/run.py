"""
DOMAIN: URBAN
TASK: PREDICTION
TrajCL baseline on frozen benchmark

Process:
- SSL-pretrain encoder with TrajCL's pipeline
- Attach a prediction head over encoder's own trajectory embedding
- Forecast last_n masked points
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / 'vendor'))
sys.path.insert(0, str(_HERE.parents[2]))

import metrics
import adapt
from config import Config
from bench_dataset import find_units

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
_DOMAIN ={0: 'urban', 1: 'maritime'}

class PredictionHead(nn.Module):
    """
    Forecast a masked point's normalized (d_lat, d_lon) from trajectory embedding
    + temporal features
    """
    def __init__(self, hidden, tau_dim=4, mlp=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden + tau_dim, mlp),
            nn.GELU(),
            nn.Linear(mlp, 2)
        )

    def forward(self, emb, tau):
        return self.net(torch.cat([emb, tau], dim=-1))
    
def set_config(args, device):
    """Populate TrajCL's global config for own data"""
    Config.device = device
    Config.cell_size = float(args.cell_size)
    Config.cell_embedding_dim = args.emb_dim
    Config.seq_embedding_dim = args.emb_dim
    Config.moco_proj_dim = args.emb_dim // 2
    Config.moco_nqueue = args.moco_nqueue
    Config.trans_hidden_dim = args.trans_hidden_dim
    Config.trans_attention_head = args.heads
    Config.trans_attention_layer = args.layers
    Config.trajcl_local_mask_sidelen = Config.cell_size * 11
    Config.trajcl_aug1, Config.trajcl_aug2 = 'mask', 'subset'

def pretrain(model, groups, spaces, cfg, device):
    """Shared MoCo encoder over pooled trajectories, each batch is built within one dataset so it
    uses that dataset's own cellspace + node2vec embeddings"""
    from model.trajcl import collate_and_augment
    from utils.traj import get_aug_fn
    aug1, aug2 = get_aug_fn(Config.trajcl_aug1), get_aug_fn(Config.trajcl_aug2)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    model.train()
    for ep in range(1, cfg.pretrain_epochs + 1):
        batches = []
        for dsname, g in groups.items():
            trajs = g['trajs']; n = len(trajs); idx = np.random.permutation(n)
            for b in range(0, n, cfg.batch):
                sel = idx[b:b + cfg.batch]
                if len(sel) < 2:
                    continue
                batches.append((dsname, [trajs[i] for i in sel]))
        np.random.shuffle(batches)
        tot = nb = 0.0
        for dsname, trajs in batches:
            cellspace, embs = spaces[adapt.region_key(dsname)]
            batch = collate_and_augment(trajs, cellspace, embs, aug1, aug2)
            logits, targets = model(*batch)
            loss = model.loss(logits, targets)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        logger.info('[TrajCL pretrain] epoch %3d/%d | moco_loss=%.4f', ep, cfg.pretrain_epochs, tot / max(nb, 1))

def _prefix_records(items):
    """Per trajectory: visible-prefix Mercator points + masked tau/targets"""
    recs = []
    for item in items:
        traj_len = int(item['traj_len'])
        pos = item['pos_mask'][:traj_len].numpy(); pad = item['pad_mask'][:traj_len].numpy()
        masked = pos & ~pad; visible = (~pos) & ~pad
        if masked.sum() == 0 or visible.sum() < 2:
            continue
        recs.append({'prefix': adapt.item_to_mercator(item)[visible],
                     'tau': item['tau'][:traj_len][masked].float(),
                     'target': item['target_coords'][:traj_len][masked].float(),
                     'masked': masked, 'tlen': traj_len, 'item': item})
    return recs

def _embed(model, cellspace, embs, prefixes, device, no_grad):
    from model.trajcl import collate_for_test
    emb_cell, emb_p, lens = collate_for_test(prefixes, cellspace, embs)
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        z = model.interpret(emb_cell, emb_p, lens)
    return z

def train_probe(model, head, recs, spaces, cfg, device, finetune):
    params = list(head.parameters()) + (list(model.parameters()) if finetune else [])
    opt = torch.optim.Adam(params, lr=cfg.probe_lr)
    model.train(finetune); head.train()
    by_ds = defaultdict(list)
    for r in recs:
        by_ds[r['item']['dataset']].append(r)
    for ep in range(1, cfg.probe_epochs + 1):
        batches = []
        for dsname, rs in by_ds.items():
            order = np.random.permutation(len(rs))
            for b in range(0, len(rs), cfg.batch):
                batches.append((dsname, [rs[i] for i in order[b:b + cfg.batch]]))
        np.random.shuffle(batches)
        tot = nb = 0.0
        for dsname, batch in batches:
            cellspace, embs = spaces[adapt.region_key(dsname)]
            z = _embed(model, cellspace, embs, [r['prefix'] for r in batch], device, no_grad=not finetune)
            loss = 0.0
            for j, r in enumerate(batch):
                M = r['target'].shape[0]
                pred = head(z[j:j+1].expand(M, -1), r['tau'].to(device))
                loss = loss + nn.functional.mse_loss(pred, r['target'].to(device))
            loss = loss / len(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        logger.info('[TrajCL probe%s] epoch %3d/%d | mse=%.6f',
                    ' (ft)' if finetune else '', ep, cfg.probe_epochs, tot / max(nb, 1))

@torch.no_grad()
def evaluate(model, head, recs, spaces, device):
    model.eval(); head.eval()
    overall, by_dom, by_ds = [], defaultdict(list), defaultdict(list)
    groups = defaultdict(list)
    for r in recs:
        groups[r['item']['dataset']].append(r)
    for dsname, rs in groups.items():
        cellspace, embs = spaces[adapt.region_key(dsname)]
        z = _embed(model, cellspace, embs, [r['prefix'] for r in rs], device, no_grad=True)
        for j, r in enumerate(rs):
            it = r['item']; tlen = r['tlen']; M = r['target'].shape[0]
            pred_m = head(z[j:j+1].expand(M, -1), r['tau'].to(device)).cpu().numpy()
            pred_full = np.zeros((tlen, 2), np.float32); pred_full[r['masked']] = pred_m
            target = it['target_coords'][:tlen].numpy()
            err = metrics.recovery_error_m(pred_full, target, r['masked'], it['denorm'])
            overall.append(err)
            by_dom[_DOMAIN[int(it['domain_id'])]].append(err)
            by_ds[it['dataset']].append(err)
    return {'overall': metrics.aggregate(overall),
            'by_domain': {k: metrics.aggregate(v) for k, v in sorted(by_dom.items())},
            'by_dataset': {k: metrics.aggregate(v) for k, v in sorted(by_ds.items())}}

def _build_region_spaces(region_pts, args, device):
    """region -> (cellspace, embs). Logs size BEFORE node2vec and bails past --max-cells so a
    giant lattice (excluded/outlier dataset) fails loud instead of OOM-killing the process."""
    spaces = {}
    for rk, pts in sorted(region_pts.items()):
        cellspace = adapt.build_cellspace(pts, args.cell_size, extent_pct=args.extent_pct)
        n_cells = cellspace.x_size * cellspace.y_size
        logger.info('region=%s | cellspace=%s | n_cells=%d', rk, cellspace, n_cells)
        if n_cells > args.max_cells:
            raise RuntimeError(
                f"region '{rk}' grid has {n_cells:,} cells (> --max-cells {args.max_cells:,}); "
                f"likely an excluded/outlier dataset or inflated extent — exclude it or raise --cell-size.")
        embs = adapt.build_cell_embeddings(cellspace, args, device, tag=rk).to(device)
        spaces[rk] = (cellspace, embs)
        logger.info('  region=%s embs=%s', rk, tuple(embs.shape))
    return spaces

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir')                  # not required in eval-only (--ckpt) mode
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--domains', nargs='*', default=None)
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--mode', default='frozen', choices=['frozen', 'finetune'])
    ap.add_argument('--cell-size', type=float, default=100.0)
    ap.add_argument('--max-cells', type=int, default=5_000_000)  # bail before a giant lattice OOM-kills node2vec
    ap.add_argument('--extent-pct', type=float, default=0.5)  # clip cellspace to [pct,100-pct] percentiles (GPS-outlier robust); 0=raw min/max
    ap.add_argument('--emb-dim', type=int, default=128)
    ap.add_argument('--trans-hidden-dim', type=int, default=512)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--moco-nqueue', type=int, default=256)
    ap.add_argument('--node2vec-epochs', type=int, default=20) # node2vec cell-embedding pretraining
    ap.add_argument('--node2vec-batch', type=int, default=256) # loss mem ~ batch*walks*walk_len*num_neg*dim; 256 fits, 32 is overhead-bound
    ap.add_argument('--node2vec-workers', type=int, default=8)
    ap.add_argument('--node2vec-num-neg', type=int, default=10) # repo 10; 5 ~halves cost (speed lever)
    ap.add_argument('--node2vec-walks', type=int, default=10) # walks_per_node; repo 10
    ap.add_argument('--node2vec-walk-length', type=int, default=50)  # repo 50
    ap.add_argument('--pretrain-epochs', type=int, default=15)
    ap.add_argument('--probe-epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--probe-lr', type=float, default=1e-3)
    ap.add_argument('--device', default=None)
    ap.add_argument('--save-ckpt', default=None)    # after training, persist encoder+head+arch here
    ap.add_argument('--ckpt', default=None)         # load weights, skip training (eval-only)
    args = ap.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info('Device: %s | mode=%s', device, args.mode)

    test_units = find_units(args.test_dir, domains=args.domains, datasets=args.datasets)
    if not test_units:
        raise FileNotFoundError('No frozen test units found (check dir/filters).')
    test_groups = adapt.load_mercator_by_dataset(test_units, task='prediction')

    from model.trajcl import TrajCL
    cfg = type('C', (), dict(pretrain_epochs=args.pretrain_epochs, probe_epochs=args.probe_epochs,
                             batch=args.batch, lr=args.lr, probe_lr=args.probe_lr))

    if args.ckpt:
        # ----- eval-only (region/domain transfer): rebuild grids from TEST extent, reuse trained encoder -----
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        a = ck['arch']
        for k, v in a.items():
            setattr(args, k, v)
        set_config(args, device)
        region_pts = defaultdict(list) # one grid per region (split suffix stripped)
        for dsname, g in test_groups.items():
            region_pts[adapt.region_key(dsname)] += g['trajs']
        spaces = _build_region_spaces(region_pts, args, device)
        model = TrajCL().to(device); model.load_state_dict(ck['model'])
        head = PredictionHead(Config.seq_embedding_dim).to(device); head.load_state_dict(ck['head'])
        logger.info('Loaded checkpoint %s', args.ckpt)
        test_items = [it for g in test_groups.values() for it in g['items']]
        res = evaluate(model, head, _prefix_records(test_items), spaces, device)
    else:
        set_config(args, device)
        train_units = find_units(args.train_dir, domains=args.domains, datasets=args.datasets) if args.train_dir else None
        if not train_units:
            raise FileNotFoundError('No frozen train units found (pass --train-dir, or --ckpt for eval-only).')

        # ----- Per-REGION Mercator trajs + CellSpace + node2vec cell embeddings -----
        # Key on region (split suffix stripped) so a city's _train/_test share ONE cellspace+node2vec:
        # halves node2vec runs and ensures pretrain & eval use the SAME embedding space
        train_groups = adapt.load_mercator_by_dataset(train_units, task='prediction')
        region_pts = defaultdict(list) # region -> all (train+test) Mercator trajs
        for groups in (train_groups, test_groups):
            for dsname, g in groups.items():
                region_pts[adapt.region_key(dsname)] += g['trajs']
        spaces = _build_region_spaces(region_pts, args, device)

        # ----- Model (reused) + pretrain (reused MoCo objective) -----
        model = TrajCL().to(device)
        logger.info('--- Pre-training (TrajCL MoCo, reused) ---')
        pretrain(model, train_groups, spaces, cfg, device)

        # ----- Prediction probe + eval -----
        train_items = [it for g in train_groups.values() for it in g['items']]
        test_items = [it for g in test_groups.values() for it in g['items']]
        head = PredictionHead(Config.seq_embedding_dim).to(device)
        logger.info('--- Prediction probe (%s) ---', args.mode)
        train_probe(model, head, _prefix_records(train_items), spaces, cfg, device,
                    finetune=(args.mode == 'finetune'))
        if args.save_ckpt:
            Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'head': head.state_dict(),
                        'arch': dict(cell_size=args.cell_size, emb_dim=args.emb_dim,
                                     trans_hidden_dim=args.trans_hidden_dim, heads=args.heads,
                                     layers=args.layers, moco_nqueue=args.moco_nqueue)}, args.save_ckpt)
            logger.info('Saved checkpoint -> %s', args.save_ckpt)
        res = evaluate(model, head, _prefix_records(test_items), spaces, device)
    o = res['overall']
    print(f"\n=== TrajCL • prediction · {args.mode} ===")
    print(f"  OVERALL mae={o.get('mae_m', float('nan')):.1f}m median={o.get('median_m', float('nan')):.1f}m "
          f"p90={o.get('p90_m', float('nan')):.1f}m (n_traj={o.get('n_trajectories',0)})")
    for d, m in res['by_domain'].items():
        print(f"  [{d}] mae={m['mae_m']:.1f}m median={m['median_m']:.1f}m n_traj={m['n_trajectories']}")

if __name__ == '__main__':
    main()