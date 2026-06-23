"""
DOMAIN: URBAN
TASK: PREDICTION
t2vec baseline on frozen benchmark

Process:
- SSL-pretrain encoder with t2vec's denoising spatial-aware-KL objective
- Attach a prediction head over encoder's own trajectory embedding
- Forecast last_n masked points
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parent / 'vendor'))

import constants
from models import EncoderDecoder
from train import genLoss, KLDIVloss, dist2weight, KLDIVcriterion
from data_utils import pad_arrays_pair, pad_arrays_keep_invp

import metrics
import data as bridge

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
    
def _wrap_trg(tokens):
    """t2vec convention: src raw, trg = BOS + tokens + EOS."""
    return np.array([constants.BOS, *tokens.tolist(), constants.EOS], dtype=np.int64)

def pretrain(m0, m1, pairs, V, D, cfg, device):
    """t2vec denoising pre-training"""
    src_list, trg_list = pairs
    criterion = KLDIVcriterion(cfg.vocab_size).to(device)
    lossF = lambda o, t: KLDIVloss(o, t, criterion, V, D)          # repo loss
    args = SimpleNamespace(cuda=(device.type == 'cuda'), generator_batch=cfg.generator_batch)
    opt0 = torch.optim.Adam(m0.parameters(), lr=cfg.lr)
    opt1 = torch.optim.Adam(m1.parameters(), lr=cfg.lr)
    n = len(src_list)
    idx = np.arange(n)
    for ep in range(1, cfg.pretrain_epochs + 1):
        np.random.shuffle(idx)
        tot = nb = 0.0
        for b in range(0, n, cfg.batch):
            sel = idx[b:b + cfg.batch]
            src = [src_list[i] for i in sel]
            trg = [_wrap_trg(trg_list[i]) for i in sel]
            gendata = pad_arrays_pair(src, trg, keep_invp=False)
            opt0.zero_grad(); opt1.zero_grad()
            loss = genLoss(gendata, m0, m1, lossF, args)
            loss.backward()
            nn.utils.clip_grad_norm_(m0.parameters(), cfg.max_grad_norm)
            nn.utils.clip_grad_norm_(m1.parameters(), cfg.max_grad_norm)
            opt0.step(); opt1.step()
            tot += loss.item(); nb += 1
        logger.info('[t2vec pretrain] epoch %3d/%d | genloss=%.4f', ep, cfg.pretrain_epochs, tot / max(nb, 1))

def _embed_prefixes(m0, token_seqs, device, no_grad):
    """Encode a list of token-id arrays -> (B, hidden) trajectory embeddings"""
    src, lengths, invp = pad_arrays_keep_invp(token_seqs)
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        hn, _ = m0.encoder(src.to(device), lengths.to(device))
        h = m0.encoder_hn2decoder_h0(hn)[-1] # last layer
    return h[invp.to(device)]

def _prefix_tokens_and_targets(region, items):
    """For each trajectory: tokenized visible prefix (~pos_mask) + masked
    positions' normalized targets and tau"""
    recs = []
    for it in items:
        tlen = int(it['traj_len'])
        pos = it['pos_mask'][:tlen].numpy()
        pad = it['pad_mask'][:tlen].numpy()
        masked = pos & ~pad
        visible = (~pos) & ~pad
        if masked.sum() == 0 or visible.sum() < 2:
            continue
        abs_traj = bridge.item_to_abs(it)
        toks = region.trip2seq(abs_traj[visible])
        if len(toks) < 2:
            continue
        recs.append({
            'tokens': np.array(toks, dtype=np.int64),
            'tau': it['tau'][:tlen][masked].float(),
            'target': it['target_coords'][:tlen][masked].float(),
            'masked': masked, 'tlen': tlen, 'item': it,
        })
    return recs

def train_probe(m0, head, recs, cfg, device, finetune):
    params = list(head.parameters()) + (list(m0.parameters()) if finetune else [])
    opt = torch.optim.Adam(params, lr=cfg.probe_lr)
    m0.train(finetune); head.train()
    n = len(recs)
    for ep in range(1, cfg.probe_epochs + 1):
        order = np.random.permutation(n)
        tot = nb = 0.0
        for b in range(0, n, cfg.batch):
            batch = [recs[i] for i in order[b:b + cfg.batch]]
            emb = _embed_prefixes(m0, [r['tokens'] for r in batch], device, no_grad=not finetune)
            loss = 0.0
            for j, r in enumerate(batch):
                M = r['target'].shape[0]
                pred = head(emb[j:j+1].expand(M, -1), r['tau'].to(device))
                loss = loss + nn.functional.mse_loss(pred, r['target'].to(device))
            loss = loss / len(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        logger.info('[t2vec probe%s] epoch %3d/%d | mse=%.6f',
                    ' (ft)' if finetune else '', ep, cfg.probe_epochs, tot / max(nb, 1))

@torch.no_grad()
def evaluate(m0, head, recs, device):
    m0.eval(); head.eval()
    overall, by_dom, by_ds = [], defaultdict(list), defaultdict(list)
    emb = _embed_prefixes(m0, [r['tokens'] for r in recs], device, no_grad=True)
    for j, r in enumerate(recs):
        it = r['item']; tlen = r['tlen']
        M = r['target'].shape[0]
        pred_m = head(emb[j:j+1].expand(M, -1), r['tau'].to(device)).cpu().numpy()  # (M,2)
        pred_full = np.zeros((tlen, 2), np.float32)
        pred_full[r['masked']] = pred_m
        target = it['target_coords'][:tlen].numpy()
        err = metrics.recovery_error_m(pred_full, target, r['masked'], it['denorm'])
        overall.append(err)
        by_dom[_DOMAIN[int(it['domain_id'])]].append(err)
        by_ds[it['dataset']].append(err)
    return {'overall': metrics.aggregate(overall),
            'by_domain': {k: metrics.aggregate(v) for k, v in sorted(by_dom.items())},
            'by_dataset': {k: metrics.aggregate(v) for k, v in sorted(by_ds.items())}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir')                  # not required in eval-only (--ckpt) mode
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--domains', nargs='*', default=None)
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--mode', default='frozen', choices=['frozen', 'finetune'])
    ap.add_argument('--cellsize-m', type=float, default=100.0)
    ap.add_argument('--minfreq', type=int, default=20)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--embedding-size', type=int, default=256)
    ap.add_argument('--hidden-size', type=int, default=256)
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--pretrain-epochs', type=int, default=15)
    ap.add_argument('--probe-epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--probe-lr', type=float, default=1e-3)
    ap.add_argument('--device', default=None)
    ap.add_argument('--save-ckpt', default=None)    # after training, persist encoder+head+region here
    ap.add_argument('--ckpt', default=None)         # load weights+region, skip training (eval-only)
    args = ap.parse_args()

    from bench_dataset import find_units
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info('Device: %s | mode=%s', device, args.mode)

    test_units = find_units(args.test_dir, domains=args.domains, datasets=args.datasets)
    if not test_units:
        raise FileNotFoundError('No frozen test units found (check dir/filters).')

    if args.ckpt:
        # ----- eval-only (region/domain transfer): reuse the SAVED region/vocab -----
        # NOTE: the encoder embedding table is tied to the training region's cell vocabulary.
        # Region transfer = held-out cities tokenize to UNK (weak but valid zero-shot).
        # Domain transfer (e.g. urban->maritime) tokenizes to near-all-UNK -> not meaningful for t2vec.
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        a = ck['arch']; region = ck['region']
        m0 = EncoderDecoder(a['vocab_size'], a['embedding_size'], a['hidden_size'],
                            a['num_layers'], a['dropout'], bidirectional=True).to(device)
        m0.load_state_dict(ck['m0'])
        head = PredictionHead(a['hidden_size']).to(device)
        head.load_state_dict(ck['head'])
        logger.info('Loaded checkpoint %s (vocab_size=%d)', args.ckpt, a['vocab_size'])
        _, test_items = bridge.load_abs_trajs(test_units, task='prediction')
        res = evaluate(m0, head, _prefix_tokens_and_targets(region, test_items), device)
    else:
        train_units = find_units(args.train_dir, domains=args.domains, datasets=args.datasets) if args.train_dir else None
        if not train_units:
            raise FileNotFoundError('No frozen train units found (pass --train-dir, or --ckpt for eval-only).')

        # ----- Build region + tokenized denoising pairs from train (data bridge) -----
        abs_trajs, _ = bridge.load_abs_trajs(train_units, task='prediction')
        region, n_out = bridge.build_region(abs_trajs, 'train', args.cellsize_m, args.minfreq, args.k)
        logger.info('region: vocab_size=%d hotcells=%d out_of_region=%d',
                    region.vocab_size, len(region.hotcell), n_out)
        rng = np.random.default_rng(0)
        pairs = bridge.tokenize_pairs(region, abs_trajs, rng, drop_rate=0.3)
        logger.info('denoising pairs: %d', len(pairs[0]))

        # ----- Repo model + V/D, then repo pre-training -----
        cfg = SimpleNamespace(vocab_size=region.vocab_size, generator_batch=256,
                              pretrain_epochs=args.pretrain_epochs, probe_epochs=args.probe_epochs,
                              batch=args.batch, lr=args.lr, probe_lr=args.probe_lr, max_grad_norm=5.0)
        m0 = EncoderDecoder(region.vocab_size, args.embedding_size, args.hidden_size,
                            args.num_layers, args.dropout, bidirectional=True).to(device)
        m1 = nn.Sequential(nn.Linear(args.hidden_size, region.vocab_size), nn.LogSoftmax(dim=1)).to(device)
        V, D = region.knearest_vocabs()
        V = torch.LongTensor(V).to(device)
        D = dist2weight(torch.FloatTensor(D)).to(device)
        logger.info('--- Pre-training (t2vec denoising, repo objective) ---')
        pretrain(m0, m1, pairs, V, D, cfg, device)

        # ----- Prediction probe + eval -----
        _, train_items = bridge.load_abs_trajs(train_units, task='prediction')
        _, test_items = bridge.load_abs_trajs(test_units, task='prediction')
        head = PredictionHead(args.hidden_size).to(device)
        logger.info('--- Prediction probe (%s) ---', args.mode)
        train_probe(m0, head, _prefix_tokens_and_targets(region, train_items), cfg, device,
                    finetune=(args.mode == 'finetune'))
        if args.save_ckpt:
            torch.save({'m0': m0.state_dict(), 'head': head.state_dict(), 'region': region,
                        'arch': dict(vocab_size=region.vocab_size, embedding_size=args.embedding_size,
                                     hidden_size=args.hidden_size, num_layers=args.num_layers,
                                     dropout=args.dropout)}, args.save_ckpt)
            logger.info('Saved checkpoint -> %s', args.save_ckpt)
        res = evaluate(m0, head, _prefix_tokens_and_targets(region, test_items), device)
    o = res['overall']
    print(f"\n=== t2vec • prediction · {args.mode} ===")
    print(f"  OVERALL mae={o.get('mae_m', float('nan')):.1f}m median={o.get('median_m', float('nan')):.1f}m "
          f"p90={o.get('p90_m', float('nan')):.1f}m (n_traj={o.get('n_trajectories',0)})")
    for d, m in res['by_domain'].items():
        print(f"  [{d}] mae={m['mae_m']:.1f}m median={m['median_m']:.1f}m n_traj={m['n_trajectories']}")

if __name__ == '__main__':
    main()