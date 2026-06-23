"""
DOMAIN: MARITIME
TASK: RECOVERY

Vanilla bi-directional LSTM over the masked input sequence, predicting the masked
points' coordinates from both-direction context
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # baselines/ on path before importing common

import common

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class BiLSTMImputer(nn.Module):
    def __init__(self, d_model=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.feat = common.InputFeaturizer(d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(2 * d_model, 2)

    def forward(self, coords, tau, kin, domain_id, hide_mask):
        e = self.feat(coords, tau, kin, domain_id, hide_mask)
        h, _ = self.lstm(e)
        return self.head(h)

def train(model, units, task, device, epochs, bs, nw, lr):
    loader = common.make_loader(units, task, bs, nw, True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for ep in range(1, epochs + 1):
        tot = nb = 0.0
        for b in loader:
            pred = model(b['coords'].to(device), b['tau'].to(device), b['kinematics'].to(device),
                         b['domain_id'].to(device), b['pos_mask'].to(device))
            mask = (b['pos_mask'] & ~b['pad_mask']).to(device)
            loss = common.masked_mse(pred, b['target_coords'].to(device), mask)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        logger.info('[BiLSTM %s] epoch %3d/%d | mse=%.6f', task, ep, epochs, tot / max(nb, 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir')                  # not required in eval-only (--ckpt) mode
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--task', default='recovery', choices=['recovery', 'prediction'])
    ap.add_argument('--domains', nargs='*', default=None)
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', default=None)
    ap.add_argument('--save-ckpt', default=None)    # after training, persist weights here
    ap.add_argument('--ckpt', default=None)         # load weights + skip training (eval-only)
    args = ap.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    test_units = common.find_units(args.test_dir, domains=args.domains, datasets=args.datasets)
    if not test_units:
        raise FileNotFoundError('No frozen test units found (check dir/filters).')
    logger.info('Device: %s | task=%s', device, args.task)

    if args.ckpt: # ----- eval-only (region/domain transfer) -----
        ck = common.load_ckpt(args.ckpt, device); a = ck['arch']
        model = BiLSTMImputer(a['d_model'], a['layers']).to(device)
        model.load_state_dict(ck['model'])
        logger.info('Loaded checkpoint %s (trained task=%s)', args.ckpt, a.get('task'))
    else: # ----- train -----
        train_units = common.find_units(args.train_dir, domains=args.domains, datasets=args.datasets) if args.train_dir else None
        if not train_units:
            raise FileNotFoundError('No frozen train units found (pass --train-dir, or --ckpt for eval-only).')
        model = BiLSTMImputer(args.d_model, args.layers).to(device)
        logger.info('--- Training (BiLSTM masked imputation) ---')
        train(model, train_units, args.task, device, args.epochs, args.batch_size, args.num_workers, args.lr)
        if args.save_ckpt:
            common.save_ckpt(args.save_ckpt, {'model': model.state_dict(),
                                              'arch': dict(d_model=args.d_model, layers=args.layers,
                                                           task=args.task)})
            logger.info('Saved checkpoint -> %s', args.save_ckpt)

    model.eval()
    def predict_batch(b):
        return model(b['coords'].to(device), b['tau'].to(device), b['kinematics'].to(device),
                     b['domain_id'].to(device), b['pos_mask'].to(device)).cpu().numpy()
    res = common.evaluate(predict_batch, test_units, args.task, device, args.batch_size, args.num_workers)
    common.print_block(f'bilstm · {args.task}', res)

if __name__ == '__main__':
    main()