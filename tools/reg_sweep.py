#!/usr/bin/env python

"""
reg_sweep.py
Team Narnia — PhysioNet Challenge 2026

Ridge / Lasso / ElasticNet regularization sweep for the logreg model family,
plus a per-fold coefficient-stability check (does the same feature keep a
similar sign/magnitude across LOSO folds, or is the additive signal
fold-specific?).

Built standalone rather than bolted onto loso_cv.py, because that file's
current CLI surface (hyperparameter flags, single-fold-only mode) wasn't
available to build against. This script:
  - Uses evaluate_model.py's ACTUAL compute_auroc_age / compute_reward /
    compute_prevalence functions directly (imported, not reimplemented) —
    metric fidelity to the official scorer is guaranteed by construction.
  - Writes loso_results.csv / loso_threshold_sweep.csv in the EXACT column
    schema confirmed against a real loso_cv.py output (2026-07-08, CatBoost
    run) — drops straight into ratchet_check.py with zero changes.
  - Implements its own LOSO fold splitting (2-site or 3-site via
    --rotate-i0002) rather than assuming loso_cv.py exposes this the same way.

IMPORTANT — things this script assumes that you should confirm before
trusting cross-comparisons against existing baselines:

1. REWARD PREVALENCE REFERENCE: CONFIRMED 2026-07-09 against two real runs
   (exact reproduction on every threshold tested) — the prevalence
   reference is the FULL labeled population across ALL sites, including
   whichever site wasn't held out in a given LOSO run. This script now
   passes the complete --features-cache population (age, y) as that
   reference automatically, since your cache should already include every
   site regardless of rotation scheme (I0002 is always at least in
   training). No separate prevalence file needed as long as your cache is
   built from the full dataset, not a pre-filtered subset.

2. SCALING: build_logreg_pipeline() (features/pipeline.py) adds a
   StandardScaler that was not confirmed to exist in whatever produced the
   0.6002 large-set baseline. See that file's docstring for why this
   matters for a C sweep specifically. This is the one remaining
   unverified assumption — resolving it needs the actual pipeline code
   that generated that baseline, not just its numeric output.

3. BINARY THRESHOLD for the per-fold sensitivity/specificity/ppv/accuracy
   columns is fixed at 0.5 on the CALIBRATED probability — matching
   evaluate_model.py's own binary_predictions convention. The
   reward-optimal threshold is chosen separately via the pooled threshold
   sweep (loso_threshold_sweep.csv), same two-stage structure as your
   existing Entry 3 THRESHOLD convention.

Usage:
    python reg_sweep.py --features-cache small_features.npz \\
        --penalties l2,l1,elasticnet --C 0.01,0.1,1,10 --l1-ratios 0.3,0.5,0.7 \\
        --output-dir outputs/reg_sweep_small

    python reg_sweep.py --features-cache large_features.npz --rotate-i0002 \\
        --penalties l2 --C 0.1,1 --output-dir outputs/reg_sweep_large_top2

Expected --features-cache format: a .npz file with arrays:
    X    — (n_patients, 48) float, PRE-AgeResidualizer feature vector
           (same 48-length hstack team_code.py/loso_cv.py produce)
    y    — (n_patients,) int/bool, Cognitive_Impairment label
    site — (n_patients,) str, SiteID (e.g. 'S0001', 'I0006', 'I0002')
    age  — (n_patients,) float, Age

This is NOT the same as any file already in your repo — you'll need to
build/cache this once from your existing extraction step. If you already
have loso_probabilities.csv-style per-patient data with features attached,
adapt build_dataset_npz() below instead of re-extracting from EDFs.
"""

import argparse
import ast
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import FEATURE_NAMES_48, FEATURE_NAMES_50
from features.pipeline import build_logreg_pipeline, extract_fitted_coefficients
from evaluate_model import compute_prevalence, compute_reward, compute_auroc as _compute_auroc

# ── Fold definitions ─────────────────────────────────────────────────────────
ALWAYS_TRAIN_SITE = 'I0002'  # only relevant when --rotate-i0002 is NOT passed
TWO_SITE_ROTATION = ['I0006', 'S0001']
THREE_SITE_ROTATION = ['I0006', 'S0001', 'I0002']

REQUIRED_LOSO_RESULTS_COLUMNS = [
    'fold', 'holdout_site', 'n_train', 'n_pos_train', 'n_neg_train',
    'n_test', 'n_pos_test', 'n_neg_test', 'age_cond_auroc', 'n_age_pairs',
    'std_auroc', 'accuracy', 'sensitivity', 'specificity', 'ppv',
    'TP', 'FP', 'FN', 'TN', 'top5_features',
]


def compute_auroc_age_with_pairs(labels, predictions, ages, gap=0):
    """
    Mirrors evaluate_model.py's compute_auroc_age EXACTLY (same loop, same
    tie-handling) but also returns the pair count (denom), matching
    loso_results.csv's n_age_pairs column. Do not let this drift from
    evaluate_model.py's own logic — it's a read-only duplicate purely to
    expose one more number for logging, never edit evaluate_model.py itself.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    ages = np.asarray(ages, dtype=float)

    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]

    numer = 0.0
    denom = 0
    for i in idx_pos:
        for j in idx_neg:
            if abs(ages[i] - ages[j]) <= gap:
                if predictions[i] > predictions[j]:
                    numer += 1
                elif predictions[i] == predictions[j]:
                    numer += 0.5
                denom += 1
    auroc = numer / denom if denom > 0 else float('nan')
    return auroc, denom


def load_feature_cache(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature cache not found: {path}")
    data = np.load(path, allow_pickle=True)
    for key in ('X', 'y', 'site', 'age'):
        if key not in data:
            raise ValueError(
                f"'{key}' missing from {path}. Found keys: {list(data.keys())}. "
                f"See this script's module docstring for the expected format.")
    X = np.asarray(data['X'], dtype=float)
    y = np.asarray(data['y']).astype(int)
    site = np.asarray(data['site']).astype(str)
    age = np.asarray(data['age'], dtype=float)
    if X.shape[1] != len(FEATURE_NAMES_48):
        raise ValueError(
            f"X has {X.shape[1]} columns, expected {len(FEATURE_NAMES_48)} "
            f"(the pre-AgeResidualizer 48-length vector). Check your cache "
            f"was built from the same extraction order as team_code.py.")
    n = len(y)
    if not (len(site) == len(age) == n == X.shape[0]):
        raise ValueError(
            f"Mismatched lengths: X={X.shape[0]}, y={n}, site={len(site)}, age={len(age)}")
    return X, y, site, age


def make_folds(site, rotate_i0002):
    """Yields (holdout_site, train_idx, test_idx). I0002 is excluded from
    the rotation (always in training) unless --rotate-i0002 is passed —
    same logic/justification as loso_cv.py's own KNOWN_MISLABELED_PATIENT_IDS-
    adjacent fold setup (learning_log.md, 2026-07-05: I0002 no longer
    justified as a permanent-training-only site at large scale)."""
    holdout_sites = THREE_SITE_ROTATION if rotate_i0002 else TWO_SITE_ROTATION
    for holdout in holdout_sites:
        test_idx = np.where(site == holdout)[0]
        train_idx = np.where(site != holdout)[0]
        if len(test_idx) == 0:
            raise ValueError(f"No patients found for holdout site '{holdout}' — check site labels.")
        yield holdout, train_idx, test_idx


def _binary_metrics(y_true, probs, threshold=0.5):
    pred = (probs > threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, accuracy=accuracy,
                sensitivity=sensitivity, specificity=specificity, ppv=ppv)


def run_one_config(X, y, age, site, rotate_i0002, penalty, C, l1_ratio,
                    calibrated=True, threshold=0.5, config_label=None):
    """
    Runs one LOSO cycle for one (penalty, C, l1_ratio) config.
    Returns (results_rows, coef_rows, pooled_df) where pooled_df has one row
    per patient across ALL test folds (site, age, label, probability) — used
    both for the reward threshold sweep and as this config's own prevalence
    reference (see module docstring, assumption #1).
    """
    results_rows = []
    coef_rows = []
    pooled_records = []

    for fold_i, (holdout, train_idx, test_idx) in enumerate(make_folds(site, rotate_i0002)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test, age_test = X[test_idx], y[test_idx], age[test_idx]

        pipeline = build_logreg_pipeline(
            y_train, penalty=penalty, C=C, l1_ratio=l1_ratio, calibrated=calibrated)
        pipeline.fit(X_train, y_train)

        probs = pipeline.predict_proba(X_test)[:, 1]

        age_cond_auroc, n_age_pairs = compute_auroc_age_with_pairs(y_test, probs, age_test, gap=2)
        std_auroc = _compute_auroc(y_test, probs)
        bm = _binary_metrics(y_test, probs, threshold=threshold)

        coefs = extract_fitted_coefficients(pipeline)
        top5_idx = np.argsort(-np.abs(coefs))[:5]
        top5 = [(FEATURE_NAMES_50[i], round(float(coefs[i]), 4)) for i in top5_idx]

        results_rows.append({
            'fold': f'holdout_{holdout}',
            'holdout_site': holdout,
            'n_train': len(train_idx),
            'n_pos_train': int(y_train.sum()),
            'n_neg_train': int((~y_train.astype(bool)).sum()),
            'n_test': len(test_idx),
            'n_pos_test': int(y_test.sum()),
            'n_neg_test': int((~y_test.astype(bool)).sum()),
            'age_cond_auroc': round(age_cond_auroc, 4),
            'n_age_pairs': n_age_pairs,
            'std_auroc': round(std_auroc, 4),
            'accuracy': round(bm['accuracy'], 4),
            'sensitivity': round(bm['sensitivity'], 4),
            'specificity': round(bm['specificity'], 4),
            'ppv': round(bm['ppv'], 4),
            'TP': bm['TP'], 'FP': bm['FP'], 'FN': bm['FN'], 'TN': bm['TN'],
            'top5_features': str(top5),
        })

        for feat_name, coef_val in zip(FEATURE_NAMES_50, coefs):
            coef_rows.append({'fold': f'holdout_{holdout}', 'feature': feat_name,
                               'coefficient': float(coef_val)})

        for i, p in zip(test_idx, probs):
            pooled_records.append({'site': site[i], 'age': age[i], 'label': y[i], 'probability': p})

    pooled_df = pd.DataFrame(pooled_records)
    return results_rows, coef_rows, pooled_df


def threshold_sweep(pooled_df, thresholds, prevalence_ages=None, prevalence_labels=None):
    """
    Pooled reward/AUROC threshold sweep, matching your existing
    loso_threshold_sweep.csv columns (threshold, reward, auroc).

    CONFIRMED 2026-07-09 against two real runs (2-site logreg + 3-site
    I0002-rotation logreg, same underlying data): the prevalence reference
    for compute_prevalence is the FULL available labeled population —
    ALL sites, INCLUDING whichever site wasn't held out / scored in a given
    run. Verified by exact reproduction on every tested threshold: scoring
    the 2-site (6281-patient) pool using only itself as the prevalence
    reference gave numbers 20-40% too high; scoring it using the full
    3-site (6600-patient) pool as the reference reproduced the real
    reported reward EXACTLY at every threshold tested. This makes sense as
    a design choice — local age-prevalence should reflect the true
    population base rate from every available label, not just whichever
    patients happen to be in a particular fold's test set.

    prevalence_ages/prevalence_labels: pass the FULL dataset's age/label
    arrays (not just the pooled test predictions) — i.e. every patient in
    your --features-cache, train and test folds combined, regardless of
    rotation scheme. If omitted, falls back to the pooled test set as its
    own reference (NOT recommended — this was the old, disproven default;
    kept only so the function still runs standalone without a full dataset
    handy, e.g. for quick synthetic smoke tests).

    The 'auroc' column is pooled age-conditioned AUROC over the combined
    scored population (gap=2) — confirmed exact match separately, see
    compute_auroc_age_with_pairs usage below.
    """
    labels = pooled_df['label'].to_numpy()
    ages = pooled_df['age'].to_numpy(dtype=float)
    probs = pooled_df['probability'].to_numpy()

    if prevalence_ages is not None and prevalence_labels is not None:
        ref_ages, ref_labels = prevalence_ages, prevalence_labels
    else:
        ref_ages, ref_labels = ages, labels

    age_to_prevalence = compute_prevalence(ages, ref_labels, ref_ages, gap=2)
    pooled_age_cond_auroc, _ = compute_auroc_age_with_pairs(labels, probs, ages, gap=2)

    rows = []
    for t in thresholds:
        preds = (probs > t).astype(int)
        reward = compute_reward(labels, preds, ages, age_to_prevalence)
        rows.append({'threshold': t, 'reward': round(reward, 4), 'auroc': round(pooled_age_cond_auroc, 4)})
    return pd.DataFrame(rows)


def config_dirname(penalty, C, l1_ratio):
    if penalty == 'elasticnet':
        return f'logreg_elasticnet_C{C}_l1r{l1_ratio}'
    return f'logreg_{penalty}_C{C}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features-cache', required=True)
    parser.add_argument('--rotate-i0002', action='store_true',
                         help='Use 3-site LOSO rotation (I0006, S0001, I0002) instead of the default 2-site.')
    parser.add_argument('--penalties', default='l2', help='Comma-separated: l1,l2,elasticnet')
    parser.add_argument('--C', default='1.0', help='Comma-separated C values, e.g. 0.01,0.1,1,10')
    parser.add_argument('--l1-ratios', default='0.5', help='Comma-separated, only used for elasticnet')
    parser.add_argument('--threshold', type=float, default=0.5,
                         help='Binary decision threshold for per-fold sens/spec/ppv/accuracy columns.')
    parser.add_argument('--reward-thresholds', default='0.05,0.08,0.10,0.12,0.15,0.20',
                         help='Thresholds swept for the pooled reward table.')
    parser.add_argument('--no-calibration', action='store_true',
                         help='Skip CalibratedClassifierCV wrapping (faster, for quick screening).')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-configs', type=int, default=40,
                         help='Safety cap on total sweep size — refuses to run a grid larger than this '
                              'without --force, so a typo in --C doesn\'t accidentally launch 200 fits.')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    X, y, site, age = load_feature_cache(args.features_cache)
    print(f"Loaded {len(y)} patients ({int(y.sum())} positive) from {args.features_cache}")
    print(f"Sites: {dict(zip(*np.unique(site, return_counts=True)))}")

    penalties = args.penalties.split(',')
    C_values = [float(c) for c in args.C.split(',')]
    l1_ratios = [float(r) for r in args.l1_ratios.split(',')] if 'elasticnet' in penalties else [None]

    configs = []
    for penalty in penalties:
        if penalty == 'elasticnet':
            for C, l1r in itertools.product(C_values, l1_ratios):
                configs.append((penalty, C, l1r))
        else:
            for C in C_values:
                configs.append((penalty, C, None))

    if len(configs) > args.max_configs and not args.force:
        print(f"ERROR: sweep grid has {len(configs)} configs, exceeding --max-configs={args.max_configs}. "
              f"Narrow your grid or pass --force if this is intentional.", file=sys.stderr)
        sys.exit(2)

    reward_thresholds = [float(t) for t in args.reward_thresholds.split(',')]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for penalty, C, l1_ratio in configs:
        label = config_dirname(penalty, C, l1_ratio)
        print(f"\n=== {label} ===")
        config_dir = output_dir / label
        config_dir.mkdir(parents=True, exist_ok=True)

        results_rows, coef_rows, pooled_df = run_one_config(
            X, y, age, site, args.rotate_i0002, penalty, C, l1_ratio,
            calibrated=not args.no_calibration, threshold=args.threshold)

        results_df = pd.DataFrame(results_rows)[REQUIRED_LOSO_RESULTS_COLUMNS]
        results_df.to_csv(config_dir / 'loso_results.csv', index=False)

        pd.DataFrame(coef_rows).to_csv(config_dir / 'coefficients.csv', index=False)

        sweep_df = threshold_sweep(pooled_df, reward_thresholds,
                                    prevalence_ages=age, prevalence_labels=y)
        sweep_df.to_csv(config_dir / 'loso_threshold_sweep.csv', index=False)

        mean_auroc = results_df['age_cond_auroc'].mean()
        best_reward_row = sweep_df.loc[sweep_df['reward'].idxmax()]
        print(f"  mean age_cond_auroc: {mean_auroc:.4f}")
        print(f"  best reward: {best_reward_row['reward']:.4f} @ t={best_reward_row['threshold']}")

        summary_rows.append({
            'config': label, 'penalty': penalty, 'C': C, 'l1_ratio': l1_ratio,
            'mean_age_cond_auroc': round(mean_auroc, 4),
            'best_reward': round(best_reward_row['reward'], 4),
            'best_reward_threshold': best_reward_row['threshold'],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values('mean_age_cond_auroc', ascending=False)
    summary_df.to_csv(output_dir / 'sweep_summary.csv', index=False)
    print(f"\n{'='*60}\nSweep summary (sorted by mean age_cond_auroc):\n{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nPer-config loso_results.csv files are ready for ratchet_check.py, e.g.:")
    print(f"  python check_submission_files.py --loso-results "
          f"{output_dir}/{summary_df.iloc[0]['config']}/loso_results.csv "
          f"--ratchet-baseline large_logreg")


if __name__ == '__main__':
    main()