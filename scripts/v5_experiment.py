"""Command-line entry point for the isolated, six-class V5 experiment."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

from v5_dataset import ARCHIVE_NAME, NAMES, prepare, read_archive, verify_dataset
from v5_trials import TRIALS, run_trial, stage_report, trial_recipe


def campaign(dataset, runs, device, batch):
    runs = Path(runs).resolve()
    verify_dataset(dataset)

    def one(trial, seed):
        report = stage_report(dataset, runs)
        if dict(trial=trial, seed=seed) in report['completed_runs']:
            print(f'Already verified complete: {trial}, seed {seed}', flush=True)
            return
        out = runs / f'{trial}_seed{seed}'
        failed = out / 'run.json'
        resume = failed.is_file() and json.loads(failed.read_text()).get('status') == 'failed'
        print(f'Running {trial}, seed {seed}. All metrics use validation only.', flush=True)
        run_trial(dataset, out, trial, seed, device, batch, resume=resume)

    for trial in TRIALS:
        one(trial, 57)
    report = stage_report(dataset, runs)
    for item in report.get('required_confirmation_runs', []):
        one(item['trial'], item['seed'])
    return stage_report(dataset, runs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit', help='Read/verify a full source archive without extracting or training')
    audit.add_argument('--archive', type=Path, default=Path.home() / 'Downloads' / ARCHIVE_NAME)
    prep = sub.add_parser('prepare', help='Remove bottle and freeze a supported group-aware split')
    prep.add_argument('--archive', type=Path, required=True)
    prep.add_argument('--exclusions', type=Path, required=True,
                      help='28-image ambiguity exclusion CSV (image_path column) or JSON path list')
    prep.add_argument('--output', type=Path, required=True)
    sub.add_parser('plan', help='Print trial recipes without loading models or training')
    check = sub.add_parser('verify', help='Verify a previously prepared six-class dataset')
    check.add_argument('--dataset', type=Path, required=True)
    status = sub.add_parser('status', help='Report trial progress; specialists stay blocked pending analysis')
    status.add_argument('--dataset', type=Path, required=True)
    status.add_argument('--runs', type=Path, required=True)
    train = sub.add_parser('train', help='Run one unified trial, using only train/val')
    train.add_argument('--trial', choices=TRIALS, required=True)
    train.add_argument('--seed', type=int, choices=(57, 58, 59), default=57)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--smoke', action='store_true', help='Two epochs; NEVER counts as a completed trial')
    train.add_argument('--resume', action='store_true', help='Resume verified interrupted training or recover evaluation')
    all_trials = sub.add_parser('campaign', help='Run baseline, U1-U6, then required seed confirmations')
    all_trials.add_argument('--runs', type=Path, required=True)
    for cmd in (train, all_trials):
        cmd.add_argument('--dataset', type=Path, required=True)
        cmd.add_argument('--device', default='mps', help='mps, cpu, or CUDA device such as 0')
        cmd.add_argument('--batch', type=int, help='Override per-recipe batch only when required by memory')
    args = parser.parse_args(argv)
    try:
        if args.command == 'audit':
            _, _, result = read_archive(args.archive)
        elif args.command == 'prepare':
            result = prepare(args.archive, args.output, args.exclusions)
        elif args.command == 'plan':
            result = dict(names=NAMES, order=list(TRIALS), specialists_allowed=False,
                          recipes={k: trial_recipe(k, 57) for k in TRIALS[:-1]},
                          U6='Combine supported earlier changes, or extend baseline to 80 epochs.',
                          confirmation_seeds=[58, 59], annotation_status='provisional')
        elif args.command == 'verify':
            receipt = verify_dataset(args.dataset)
            result = dict(names=receipt['names'], split_counts=receipt['split_counts'],
                          class_counts=receipt['class_counts'], annotation_status=receipt['annotation_status'])
        elif args.command == 'status':
            result = stage_report(args.dataset, args.runs)
        elif args.command == 'campaign':
            result = campaign(args.dataset, args.runs, args.device, args.batch)
        else:
            result = run_trial(args.dataset, args.output, args.trial, args.seed,
                               args.device, args.batch, args.smoke, args.resume)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f'V5 experiment stopped: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
