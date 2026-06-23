"""
Run all baselines over frozen benchmark splits.

Quadrant map:
    - Prediction & urban: t2vec, TrajCL
    - Prediction & maritime: LSTM seq2seq, Kalman
    - Recovery & urban: Transformer, GRU seq2seq
    - Recovery & maritime: Kalman, BiLSTM

Usage:
    # Tier 1 (train once, save weights, eval in-distribution test):
    python run_all.py --train-dir frozen/train --test-dir frozen/test \
        --ckpt-dir ckpts --device cuda --out tier1.json

    # Region transfer (no retrain; eval saved models on held-out-region split):
    python run_all.py --eval-only --ckpt-dir ckpts --test-dir frozen/heldout_regions \
        --device cuda --out region_transfer.json

    # Domain transfer (no retrain; e.g. urban models on maritime data):
    python run_all.py --eval-only --ckpt-dir ckpts --test-dir frozen/test \
        --eval-domains maritime --include transformer gru_seq2seq trajcl \
        --device cuda --out urban_to_maritime.json
    # ...and the reverse for maritime models on urban data:
    #   --eval-domains urban --include lstm_seq2seq bilstm
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

MATRIX = [
    dict(name='t2vec', quad='pred-urban', script='t2vec/run.py', domain='urban', task='prediction', train=True, device=True, epochs_flag=None, extra=[]),
    dict(name='trajcl', quad='pred-urban', script='trajcl/run.py', domain='urban', task='prediction', train=True, device=True, epochs_flag=None, extra=[]),
    dict(name='lstm_seq2seq', quad='pred-maritime', script='seq2seq/run.py', domain='maritime', task='prediction', train=True, device=True, epochs_flag='--epochs', extra=['--rnn', 'lstm', '--task', 'prediction']),
    dict(name='kalman', quad='pred-maritime', script='kalman/run.py', domain='maritime', task='prediction', train=False, device=False, epochs_flag=None, extra=['--task', 'prediction']),
    dict(name='transformer', quad='rec-urban', script='transformer/run.py', domain='urban', task='recovery', train=True, device=True, epochs_flag='--epochs', extra=['--task', 'recovery']),
    dict(name='gru_seq2seq', quad='rec-urban', script='seq2seq/run.py', domain='urban', task='recovery', train=True, device=True, epochs_flag='--epochs', extra=['--rnn', 'gru', '--task', 'recovery']),
    dict(name='linear', quad='rec-urban', script='linear/run.py', domain='urban', task='recovery', train=False, device=False, epochs_flag=None, extra=['--task', 'recovery']),
    dict(name='kalman', quad='rec-maritime', script='kalman/run.py', domain='maritime', task='recovery', train=False, device=False, epochs_flag=None, extra=['--task', 'recovery']),
    dict(name='bilstm', quad='rec-maritime', script='bilstm/run.py', domain='maritime', task='recovery', train=True, device=True, epochs_flag='--epochs', extra=['--task', 'recovery']),
    dict(name='linear', quad='rec-maritime', script='linear/run.py', domain='maritime', task='recovery', train=False, device=False, epochs_flag=None, extra=['--task', 'recovery']),
]

_OVERALL = re.compile(r'OVERALL\s+mae=([\d.]+)m\s+median=([\d.]+)m\s+p90=([\d.]+)m\s+\(n_traj=(\d+)\)')
_BYDOM = re.compile(r'\[(\w+)\s*\]\s+mae=([\d.]+)m\s+median=([\d.]+)m\s+n_traj=(\d+)')

def parse_stdout(text):
    m = _OVERALL.search(text)
    if not m:
        return None
    overall = {'mae_m': float(m[1]), 'median_m': float(m[2]), 'p90_m': float(m[3]), 'n_traj': int(m[4])}
    by_dom = {d: {'mae_m': float(a), 'median_m': float(b), 'n_traj': int(n)}
              for d, a, b, n in _BYDOM.findall(text)}
    return {'overall': overall, 'by_domain': by_dom}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir')
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--device', default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--include', nargs='*', default=None)
    ap.add_argument('--skip', nargs='*')
    ap.add_argument('--ckpt-dir', default=None) # train mode: --save-ckpt <dir>/<name>.pt ; eval-only: --ckpt
    ap.add_argument('--eval-only', action='store_true') # skip training, load checkpoints (region/domain transfer)
    ap.add_argument('--eval-domains', default=None) # override per-baseline domain at eval (domain transfer)
    ap.add_argument('--out', default='baseline_results.json')
    args = ap.parse_args()

    if args.eval_only and not args.ckpt_dir:
        ap.error('--eval-only requires --ckpt-dir (where the trained checkpoints live)')
    if not args.eval_only and not args.train_dir:
        ap.error('training mode requires --train-dir (or pass --eval-only for transfer eval)')

    results = {}
    for e in MATRIX:
        quad, name = e['quad'], e['name']
        slot = results.setdefault(quad, {})
        if args.include is not None:
            if name not in args.include:
                continue
        elif name in (args.skip or []):
            slot[name] = {'status': 'skipped'}
            continue
        domain = args.eval_domains if (args.eval_only and args.eval_domains) else e['domain']
        cross_domain = args.eval_only and args.eval_domains is not None and args.eval_domains != e['domain']
        if cross_domain and name == 't2vec':        # t2vec vocab is region-bound -> cross-domain = all-UNK
            slot[name] = {'status': 'skipped (t2vec cross-domain: vocab UNK)'}
            print(f">>> [{quad}] {name}: skipped (t2vec cross-domain not meaningful)", flush=True)
            continue
        cmd = [PY, '-u', str(HERE / e['script'])]   # -u so child stdout streams unbuffered
        if e['train']:
            if args.eval_only:
                cmd += ['--ckpt', str(Path(args.ckpt_dir) / f'{name}.pt')]
            else:
                cmd += ['--train-dir', args.train_dir]
                if args.ckpt_dir:
                    cmd += ['--save-ckpt', str(Path(args.ckpt_dir) / f'{name}.pt')]
        cmd += ['--test-dir', args.test_dir, '--domains', domain] + e['extra']
        if e['device'] and args.device:
            cmd += ['--device', args.device]
        if not args.eval_only and e['epochs_flag'] and args.epochs is not None:
            cmd += [e['epochs_flag'], str(args.epochs)]
        print(f">>> [{quad}] {name}: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        captured = []
        for line in proc.stdout:                 # tee: stream live + keep text for parsing
            print(line, end='', flush=True)
            captured.append(line)
        proc.wait()
        out = ''.join(captured)
        res = parse_stdout(out)
        if res is None:
            slot[name] = {'status': 'FAILED', 'returncode': proc.returncode,
                          'stderr_tail': out[-800:]}
            print(f"    FAILED (exit {proc.returncode}) — see stderr_tail in JSON", flush=True)
        else:
            slot[name] = res
            o = res['overall']
            print(f"    median={o['median_m']:.1f}m  mae={o['mae_m']:.1f}m  (n_traj={o['n_traj']})", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))

    print("\n================  BASELINE RESULTS  ================")
    for quad, entries in results.items():
        print(f"\n[{quad}]")
        for name, r in entries.items():
            if 'overall' in r:
                o = r['overall']
                dom = "  ".join(f"{d}:med={v['median_m']:.0f}m" for d, v in r['by_domain'].items())
                print(f"  {name:14s} median={o['median_m']:7.1f}m  mae={o['mae_m']:8.1f}m   [{dom}]")
            else:
                print(f"  {name:14s} {r.get('status')}")
    print(f"\nWrote consolidated results -> {args.out}")

if __name__ == '__main__':
    main()