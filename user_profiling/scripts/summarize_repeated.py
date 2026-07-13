import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

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
METRIC_COLUMNS = {
    'f1_micro': 'f1_micro',
    'f1_macro': 'f1_macro',
    'hit_rate_3': 'hit@3',
    'recall_3': 'recall@3',
    'map_macro': 'map_macro',
}


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(',') if part.strip()]


def read_metric(path: str, metric: str) -> dict[str, float]:
    with open(path, newline='') as file:
        return {row['prediction']: float(row[metric]) for row in csv.DictReader(file, delimiter='\t')}


def paired_tests(baseline: np.ndarray, augmented: np.ndarray) -> tuple[float, float]:
    t_pvalue = float(ttest_rel(augmented, baseline).pvalue)
    differences = augmented - baseline
    if np.allclose(differences, 0):
        return t_pvalue, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        wilcoxon_pvalue = float(wilcoxon(differences, method='approx').pvalue)
    return t_pvalue, wilcoxon_pvalue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--seeds', type=parse_csv, default=parse_csv('13,21,42,87,100'))
    parser.add_argument('--items', type=parse_csv, default=DEFAULT_ITEMS)
    parser.add_argument('--metric', choices=METRIC_COLUMNS, default='recall_3')
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--topk', type=int, default=6)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    metric_column = METRIC_COLUMNS[args.metric]
    baseline_by_seed = []
    augmented_by_seed = []
    for seed_text in args.seeds:
        seed = int(seed_text)
        baseline_by_seed.append(read_metric(get_log_file(args.model, ExpType.HUMAN, seed=seed), metric_column))
        augmented_by_seed.append(read_metric(get_log_file(
            args.model,
            ExpType.SIM4HUMAN5,
            hc='hot',
            ratio=args.ratio,
            seed=seed,
            topk=args.topk,
        ), metric_column))

    rows = []
    for item in args.items + ['Macro average']:
        if item == 'Macro average':
            baseline = np.array([np.mean([run[name] for name in args.items]) for run in baseline_by_seed])
            augmented = np.array([np.mean([run[name] for name in args.items]) for run in augmented_by_seed])
        else:
            baseline = np.array([run[item] for run in baseline_by_seed])
            augmented = np.array([run[item] for run in augmented_by_seed])
        t_pvalue, wilcoxon_pvalue = paired_tests(baseline, augmented)
        relative_gain = np.divide(
            augmented - baseline,
            baseline,
            out=np.full_like(baseline, np.nan),
            where=baseline != 0,
        )
        rows.append({
            'item': item,
            'baseline_mean': np.mean(baseline),
            'baseline_std': np.std(baseline, ddof=1),
            'augmented_mean': np.mean(augmented),
            'augmented_std': np.std(augmented, ddof=1),
            'absolute_gain': np.mean(augmented - baseline),
            'relative_gain_percent': np.nanmean(relative_gain) * 100,
            'paired_t_pvalue': t_pvalue,
            'wilcoxon_pvalue': wilcoxon_pvalue,
        })

    fieldnames = list(rows[0])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)


if __name__ == '__main__':
    main()
