"""
Seq2seq (encoder-decoder RNN) baseline on frozen benchmark

One model serving 2 evaluation quadrants:
    --rnn gru --task recovery -> (recovery, urban)
    --rnn lstm --task prediction -> (prediction, maritime)

Encoder: RNN over masked input sequence
Decoder: autoregressively generates masked points' coordinates
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # baselines/ on path before importing common

import common

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class Seq2Seq(nn.Module):
    def __init__(self, rnn='gru', d_model=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.rnn_type = rnn
        self.feat = common.InputFeaturizer(d_model)
        RNN = nn.GRU if rnn == 'gru' else nn.LSTM
        drop = dropout if n_layers > 1 else 0.0
        self.encoder = RNN(d_model, d_model, num_layers=n_layers, batch_first=True, dropout=drop)
        self.decoder = RNN(d_model, d_model, num_layers=n_layers, batch_first=True, dropout=drop)
        self.coord_in = nn.Linear(2, d_model)
        self.tau_in = nn.Linear(4, d_model)
        self.head = nn.Linear(d_model, 2)

    def encode(self, b, device):
        e = self.feat(b['coords'].to(device), b['tau'].to(device), b['kinematics'].to(device),
                      b['domain_id'].to(device), b['pos_mask'].to(device))
        lengths = b['traj_len'].clamp(min=1).cpu()
        packed = pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        _, hN = self.encoder(packed)
        return hN

    def decode(self, hN, tau_m, tgt_m, teacher_force):
        B, Mmax, _ = tau_m.shape
        h = hN
        prev = torch.zeros(B, 2, device=tau_m.device)
        preds = []
        for k in range(Mmax):
            x = (self.coord_in(prev) + self.tau_in(tau_m[:, k])).unsqueeze(1)
            out, h = self.decoder(x, h)
            p = self.head(out.squeeze(1))
            preds.append(p)
            prev = tgt_m[:, k] if teacher_force else p
        return torch.stack(preds, 1)
    
def train(model, units, task, device, epochs, bs, nw, lr):
    loader = common.make_loader(units, task, bs, nw, True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for ep in range(1, epochs + 1):
        tot = nb = 0.0
        for b in loader:
            hN = model.encode(b, device)
            tau_m, tgt_m, valid, _, _ = common.gather_masked(b, device)
            preds = model.decode(hN, tau_m, tgt_m, teacher_force=True)
            loss = common.masked_mse(preds, tgt_m, valid)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        logger.info('[seq2seq %s %s] epoch %3d/%d | mse=%.6f',
                    model.rnn_type, task, ep, epochs, tot / max(nb, 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir', required=True)
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--rnn', default='gru', choices=['gru', 'lstm'])
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
    args = ap.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    train_units = common.find_units(args.train_dir, domains=args.domains, datasets=args.datasets)
    test_units = common.find_units(args.test_dir, domains=args.domains, datasets=args.datasets)
    if not train_units or not test_units:
        raise FileNotFoundError('No frozen units found (check dirs/filters).')
    logger.info('Device: %s | rnn=%s | task=%s', device, args.rnn, args.task)

    model = Seq2Seq(args.rnn, args.d_model, args.layers).to(device)
    logger.info('--- Training (Seq2Seq, teacher-forced) ---')
    train(model, train_units, args.task, device, args.epochs, args.batch_size, args.num_workers, args.lr)

    model.eval()
    @torch.no_grad()
    def predict_batch(b):
        hN = model.encode(b, device)
        tau_m, tgt_m, valid, idx_list, _ = common.gather_masked(b, device)
        preds = model.decode(hN, tau_m, tgt_m, teacher_force=False).cpu().numpy()
        B, L = b['pos_mask'].shape
        pred_full = np.zeros((B, L, 2), np.float32)
        for i in range(B):
            n = len(idx_list[i])
            if n:
                pred_full[i, idx_list[i].cpu().numpy()] = preds[i, :n]
        return pred_full
    res = common.evaluate(predict_batch, test_units, args.task, device, args.batch_size, args.num_workers)
    common.print_block(f'seq2seq({args.rnn}) · {args.task}', res)

if __name__ == '__main__':
    main()