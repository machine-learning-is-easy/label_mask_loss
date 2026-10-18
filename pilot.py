# Pilot sweep: train every loss x lr (x alpha for the masked losses) for a few runs,
# pick the configuration of each loss on the validation set, then compare test accuracy.
#
#   python pilot.py                       # train every configuration, then summarize
#   python pilot.py --summarize_only      # only summarize finished runs
#
# Combinations missing from the grid are skipped by the summary, so a second call with
# e.g. --losses LWSCE --lrs 1e-3 --alphas 0.1 0.5 can extend the sweep around a chosen lr.
import argparse
import os
import subprocess
import sys
import pandas as pd

ALPHA_LOSSES = ['LWSCE', 'ALWSCE', 'LWSCE+LS']

parser = argparse.ArgumentParser(description='Pilot sweep')
parser.add_argument('--name', default='pilot2', help='prefix of the result names, separates sweeps')
parser.add_argument('--dataset', default='CIFAR100')
parser.add_argument('--model', default='resnet18', choices=['default', 'resnet18', 'resnet50'])
parser.add_argument('--opt_alg', default='ADAMW')
parser.add_argument('--losses', nargs='+', default=['CROSSENTROPY', 'LWSCE'])
parser.add_argument('--lrs', nargs='+', default=['1e-3', '3e-4'])
parser.add_argument('--alphas', nargs='+', default=['0.2'], help='only used by ' + ', '.join(ALPHA_LOSSES))
parser.add_argument('--runs', type=int, default=3)
# training.py trains range(1, epoch), i.e. epoch - 1 epochs
parser.add_argument('--epoch', type=int, default=31)
parser.add_argument('--augment', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--weight_decay', default='5e-2')
parser.add_argument('--scheduler', default='cosine', choices=['none', 'cosine'])
parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--summarize_only', action='store_true')
args = parser.parse_args()

current_folder = os.path.dirname(os.path.abspath(__file__))


def configs():
    for loss in args.losses:
        for lr in args.lrs:
            for alpha in (args.alphas if loss in ALPHA_LOSSES else [None]):
                yield loss, lr, alpha


def tag(lr, alpha):
    return '{}_lr{}'.format(args.name, lr) + ('' if alpha is None else '_a{}'.format(alpha))


def result_dir(loss, lr, alpha):
    # same naming as training.py
    run_name = '{}_{}_{}_{}'.format(args.dataset, args.opt_alg, loss, tag(lr, alpha))
    if args.model != 'default':
        run_name = '{}_{}'.format(args.model, run_name)
    return os.path.join(current_folder, 'result', '{}_result'.format(run_name))


def summary_path(loss, lr, alpha):
    return os.path.join(result_dir(loss, lr, alpha), 'summary.csv')


def finished(loss, lr, alpha):
    path = summary_path(loss, lr, alpha)
    return os.path.exists(path) and len(pd.read_csv(path)) >= args.runs


def train():
    for loss, lr, alpha in configs():
        # skip finished combinations so an interrupted sweep can be resumed
        if finished(loss, lr, alpha):
            print('skip {} lr={} alpha={}, already finished'.format(loss, lr, alpha))
            continue
        cmd = [sys.executable, os.path.join(current_folder, 'training.py'),
               '--dataset', args.dataset, '--model', args.model, '--opt_alg', args.opt_alg,
               '--lossfunction', loss, '--lr', lr, '--epoch', str(args.epoch), '--runs', str(args.runs),
               '--weight_decay', args.weight_decay, '--scheduler', args.scheduler,
               '--num_workers', str(args.num_workers), '--tag', tag(lr, alpha)]
        if alpha is not None:
            cmd += ['--alpha', alpha]
        if args.augment:
            cmd.append('--augment')
        if args.amp:
            cmd.append('--amp')
        print('run', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=current_folder)


def summarize():
    rows = []
    for loss, lr, alpha in configs():
        if not os.path.exists(summary_path(loss, lr, alpha)):
            print('missing {} lr={} alpha={}'.format(loss, lr, alpha))
            continue
        df = pd.read_csv(summary_path(loss, lr, alpha)).head(args.runs)
        rows.append({'loss': loss, 'lr': lr, 'alpha': alpha if alpha is not None else '', 'runs': len(df),
                     'val_acc': round(df['best_val_acc'].mean(), 2), 'test_acc': round(df['test_acc'].mean(), 2),
                     'test_std': round(df['test_acc'].std(), 2), 'best_epoch': round(df['best_epoch'].mean(), 1),
                     'result_dir': os.path.relpath(result_dir(loss, lr, alpha), current_folder)})
    table = pd.DataFrame(rows, columns=['loss', 'lr', 'alpha', 'runs', 'val_acc', 'test_acc', 'test_std',
                                        'best_epoch', 'result_dir'])
    # the configuration of each loss is chosen by validation accuracy only
    table['selected'] = table.index.isin(table.groupby('loss')['val_acc'].idxmax())
    print()
    print(table.drop(columns='result_dir').to_string(index=False))

    # runs share the same seeds and data split, so they can be compared pairwise
    selected = {r['loss']: pd.read_csv(os.path.join(current_folder, r['result_dir'], 'summary.csv')).head(args.runs)
                for _, r in table[table['selected']].iterrows()}
    base = args.losses[0]
    if base in selected:
        for loss in args.losses[1:]:
            if loss not in selected:
                continue
            n = min(len(selected[base]), len(selected[loss]))
            diff = selected[loss]['test_acc'].values[:n] - selected[base]['test_acc'].values[:n]
            print('{} - {} test acc per run: {}, mean {:+.2f}'.format(
                loss, base, ', '.join('{:+.2f}'.format(d) for d in diff), diff.mean()))

    out = os.path.join(current_folder, 'result', '{}_{}_{}_{}_sweep.csv'.format(
        args.name, args.dataset, args.model, args.opt_alg))
    table.to_csv(out, index=False)
    print('saved', out)


if not args.summarize_only:
    train()
summarize()
