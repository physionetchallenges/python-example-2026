#!/usr/bin/env python3
"""
confusion_at_threshold.py — Team Narnia

Computes confusion matrices (TP/FP/FN/TN, sensitivity, specificity, PPV)
from a loso_probabilities.csv file at one or more chosen thresholds, using
calibrated_probability (not the uncalibrated model.predict() default that
loso_results.csv's TP/FP/FN/TN columns are built from).

Built to answer a specific question: does logreg's small-set reward
advantage over XGBoost come from fewer false positives (specificity) or
better true-positive capture, and does it hold at a SHARED threshold or
only at each model's own tuned optimum?

Usage:
    python confusion_at_threshold.py --probs loso_outputs_xgb/loso_probabilities.csv \\
        --probs loso_outputs_logreg_l1/loso_probabilities.csv \\
        --thresholds 0.10,0.12,0.15,0.20 \\
        --per-site

    # If you only have one file and want to sweep it:
    python confusion_at_threshold.py --probs loso_outputs_logreg_l1/loso_probabilities.csv \\
        --thresholds 0.05,0.08,0.10,0.12,0.15,0.20
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def confusion_at(labels, probs, threshold):
    preds = (probs > threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                sensitivity=round(sensitivity, 3), specificity=round(specificity, 3),
                ppv=round(ppv, 3), accuracy=round(accuracy, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--probs', action='append', required=True,
                         help='Path to a loso_probabilities.csv. Repeat for multiple runs to compare.')
    parser.add_argument('--thresholds', default='0.05,0.08,0.10,0.12,0.15,0.20')
    parser.add_argument('--per-site', action='store_true',
                         help='Also break out I0006 vs S0001 separately, not just pooled.')
    parser.add_argument('--prob-col', default='calibrated_probability',
                         help='Column to threshold. Use "raw_probability" to compare uncalibrated instead.')
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(',')]

    for path in args.probs:
        path = Path(path)
        df = pd.read_csv(path)
        label = f'{path.parent.name}/{path.name}'
        print(f'\n{"="*70}\n{label}  (n={len(df)}, {int(df.label.sum())} positive)\n{"="*70}')

        labels = df['label'].to_numpy()
        probs = df[args.prob_col].to_numpy()

        print(f'\n  POOLED (both folds combined):')
        for t in thresholds:
            c = confusion_at(labels, probs, t)
            print(f'    t={t:.2f}  TP={c["TP"]:>3} FP={c["FP"]:>3} FN={c["FN"]:>3} TN={c["TN"]:>3}  '
                  f'sens={c["sensitivity"]:.3f}  spec={c["specificity"]:.3f}  ppv={c["ppv"]:.3f}')

        if args.per_site:
            for site in sorted(df['SiteID'].unique()):
                sub = df[df['SiteID'] == site]
                print(f'\n  {site} only (n={len(sub)}, {int(sub.label.sum())} positive):')
                for t in thresholds:
                    c = confusion_at(sub['label'].to_numpy(), sub[args.prob_col].to_numpy(), t)
                    print(f'    t={t:.2f}  TP={c["TP"]:>3} FP={c["FP"]:>3} FN={c["FN"]:>3} TN={c["TN"]:>3}  '
                          f'sens={c["sensitivity"]:.3f}  spec={c["specificity"]:.3f}  ppv={c["ppv"]:.3f}')


if __name__ == '__main__':
    main()