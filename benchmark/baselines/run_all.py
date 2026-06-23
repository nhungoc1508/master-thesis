"""
Run all baselines over frozen benchmark splits.

Quadrant map:
    - Prediction & urban: t2vec, TrajCL
    - Prediction & maritime: LSTM seq2seq, Kalman
    - Recovery & urban: Transformer, GRU seq2seq
    - Recovery & maritime: Kalman, BiLSTM

Usage:
    python run_all.py \
        --train-dir benchmark/frozen/train \
        --test-dir benchmark/frozen/test \
        --device cuda --out baseline_results.json
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
    ap.add_argument('--train-dir', required=True)
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--device', default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--include', nargs='*', default=None)
    ap.add_argument('--skip', nargs='*')
    ap.add_argument('--out', default='baseline_results.json')
    args = ap.parse_args()

    results = {}
    for e in MATRIX:
        quad, name = e['quad'], e['name']
        slot = results.setdefault(quad, {})
        if args.include and name not in args.include:
            continue
        if name in (args.skip or []):
            slot[name] = {'status': 'skipped'}
            continue
        cmd = [PY, str(HERE / e['script'])]
        if e['train']:
            cmd += ['--train-dir', args.train_dir]
        cmd += ['--test-dir', args.test_dir, '--domains', e['domain']] + e['extra']
        if e['device'] and args.device:
            cmd += ['--device', args.device]
        if e['epochs_flag'] and args.epochs is not None:
            cmd += [e['epochs_flag'], str(args.epochs)]
        print(f">>> [{quad}] {name}: {' '.join(cmd)}", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        res = parse_stdout(p.stdout)
        if res is None:
            slot[name] = {'status': 'FAILED', 'returncode': p.returncode,
                          'stderr_tail': p.stderr[-800:]}
            print(f"    FAILED (exit {p.returncode}) — see stderr_tail in JSON", flush=True)
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