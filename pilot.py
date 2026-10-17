# -*- coding: utf-8 -*-
# Pilot experiment: tune lr per loss function on the validation set, then compare test accuracy.
#
#   python pilot.py                       # train every loss x lr, then summarize
#   python pilot.py --summarize_only      # only summarize finished runs
import argparse
import os
import subprocess
import sys
import pandas as pd

parser = argparse.ArgumentParser(description='Pilot lr sweep')
parser.add_argument('--dataset', default='CIFAR100')
parser.add_argument('--opt_alg', default='ADAM')
parser.add_argument('--losses', nargs='+', default=['CROSSENTROPY', 'LWSCE'])
parser.add_argument('--lrs', nargs='+', default=['1e-3', '3e-4', '1e-4'])
parser.add_argument('--runs', type=int, default=3)
# training.py trains range(1, epoch), i.e. epoch - 1 epochs
parser.add_argument('--epoch', type=int, default=31)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--summarize_only', action='store_true')
args = parser.parse_args()

current_folder = os.path.dirname(os.path.abspath(__file__))


def tag(lr):
    return 'pilot_a{}_lr{}'.format(args.alpha, lr)


def summary_path(loss, lr):
    run_name = '{}_{}_{}_{}'.format(args.dataset, args.opt_alg, loss, tag(lr))
    return os.path.join(current_folder, 'result', '{}_result'.format(run_name), 'summary.csv')


def finished(loss, lr):
    path = summary_path(loss, lr)
    return os.path.exists(path) and len(pd.read_csv(path)) >= args.runs


def train():
    for loss in args.losses:
        for lr in args.lrs:
            # skip finished combinations so an interrupted sweep can be resumed
            if finished(loss, lr):
                print('skip {} lr={}, already finished'.format(loss, lr))
                continue
            cmd = [sys.executable, os.path.join(current_folder, 'training.py'),
                   '--dataset', args.dataset, '--opt_alg', args.opt_alg, '--lossfunction', loss,
                   '--lr', lr, '--epoch', str(args.epoch), '--runs', str(args.runs),
                   '--alpha', str(args.alpha), '--tag', tag(lr)]
            print('run', ' '.join(cmd))
            subprocess.run(cmd, check=True, cwd=current_folder)


def summarize():
    rows = []
    selected = {}
    for loss in args.losses:
        best = None
        for lr in args.lrs:
            if not os.path.exists(summary_path(loss, lr)):
                print('missing {} lr={}'.format(loss, lr))
                continue
            df = pd.read_csv(summary_path(loss, lr)).head(args.runs)
            row = [loss, lr, len(df), round(df['best_val_acc'].mean(), 2),
                   round(df['test_acc'].mean(), 2), round(df['test_acc'].std(), 2), '']
            rows.append(row)
            # lr is chosen by validation accuracy only
            if best is None or row[3] > best[3]:
                best = row
                selected[loss] = df
        if best is not None:
            best[6] = '*'
    table = pd.DataFrame(rows, columns=['loss', 'lr', 'runs', 'val_acc', 'test_acc', 'test_std', 'selected'])
    print()
    print(table.to_string(index=False))

    # runs share the same seed sequence and data split, so they can be compared pairwise
    base = args.losses[0]
    if base in selected:
        for loss in args.losses[1:]:
            if loss not in selected:
                continue
            n = min(len(selected[base]), len(selected[loss]))
            diff = selected[loss]['test_acc'].values[:n] - selected[base]['test_acc'].values[:n]
            print('{} - {} test acc per run: {}, mean {:+.2f}'.format(
                loss, base, ', '.join('{:+.2f}'.format(d) for d in diff), diff.mean()))

    out = os.path.join(current_folder, 'result', 'pilot_{}_{}_a{}.csv'.format(args.dataset, args.opt_alg, args.alpha))
    table.to_csv(out, index=False)
    print('saved', out)


if not args.summarize_only:
    train()
summarize()
