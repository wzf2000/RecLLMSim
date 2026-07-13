import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROFILE_DIR))

from log_util import ExpType, get_log_file

DEFAULT_ITEMS = [
    'Personality',
    'Daily Interests and Hobbies',
    'Travel Habits',
    'Dining Preferences',
    'Spending Habits',
]


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(',') if part.strip()]


def completed(log_file: str, items: list[str]) -> bool:
    path = Path(log_file)
    if not path.exists():
        return False
    with path.open() as file:
        completed_items = {line.split('\t', 1)[0] for line in file if '\t' in line}
    return set(items).issubset(completed_items)


def run(command: list[str], log_file: str, items: list[str], force: bool, dry_run: bool) -> None:
    if not force and completed(log_file, items):
        print(f'Skip completed run: {log_file}', flush=True)
        return
    print(' '.join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=PROFILE_DIR, env={**os.environ, 'TOKENIZERS_PARALLELISM': 'false'}, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=['ml', 'lm'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--seeds', type=parse_csv, default=parse_csv('13,21,42,87,100'))
    parser.add_argument('--items', type=parse_csv, default=DEFAULT_ITEMS)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--data_version', type=int, default=2)
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--topk', type=int, default=6)
    parser.add_argument('--epochs', type=float, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--include_agent_only', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    entrypoint = PROFILE_DIR / ('predict_ml.py' if args.family == 'ml' else 'predict_lm.py')
    common = [
        args.python,
        str(entrypoint),
        '-m',
        args.model,
        '-d',
        str(args.data_version),
        '--only_all',
        '--items',
        ','.join(args.items),
    ]
    if args.family == 'lm':
        common.extend([
            '--epochs',
            str(args.epochs),
            '--batch_size',
            str(args.batch_size),
            '--max_length',
            str(args.max_length),
        ])

    for seed_text in args.seeds:
        seed = int(seed_text)
        human_log = get_log_file(args.model, ExpType.HUMAN, seed=seed)
        run(common + ['-t', 'human', '--seed', str(seed)], human_log, args.items, args.force, args.dry_run)

        if args.include_agent_only:
            agent_only_log = get_log_file(args.model, ExpType.SIM2HUMAN3, seed=seed)
            run(common + ['-t', 'sim2human3', '--seed', str(seed)], agent_only_log, args.items, args.force, args.dry_run)

        augmentation_metadata = {
            'hc': 'hot',
            'ratio': args.ratio,
            'seed': seed,
            'topk': args.topk,
        }
        augmentation_log = get_log_file(args.model, ExpType.SIM4HUMAN5, **augmentation_metadata)
        run(
            common + [
                '-t',
                'sim4human5',
                '-r',
                str(args.ratio),
                '-hc',
                'hot',
                '--topk',
                str(args.topk),
                '--seed',
                str(seed),
            ],
            augmentation_log,
            args.items,
            args.force,
            args.dry_run,
        )


if __name__ == '__main__':
    main()
