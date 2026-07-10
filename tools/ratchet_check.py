#!/usr/bin/env python

"""
ratchet_check.py
Team Narnia — PhysioNet Challenge 2026

Answers one question before you submit: "is this LOSO result actually worse
than what we've already confirmed, or is it inside the noise band we already
characterized?" Motivated directly by the Entry 1-3 leaderboard sequence
(0.624 -> 0.616 -> 0.606), which LOOKED like a monotonic decline but was
confirmed via Hanley-McNeil SE to be ~0.12-0.40 sigma per step — i.e. noise,
not regression. This script automates that same check instead of re-deriving
it by hand every time.

One-directional ratchet: only ever fails on a REGRESSION beyond the sigma
threshold. An improvement of any size always passes — this is a gate against
backsliding, not a two-sided significance test.

Baselines live in ratchet_baselines.json, hand-curated (see that file's
_readme). This script never writes to it — promoting a new result to a
baseline is a deliberate decision, not something a script should do for you.

Usage (standalone):
    python ratchet_check.py --loso-results loso_results.csv --baseline small_entry3

    # If your loso_results.csv uses different column names than the
    # defaults below, point at them explicitly:
    python ratchet_check.py --loso-results loso_results.csv --baseline small_entry3 \\
        --auroc-col age_cond_auroc --n-pos-col n_pos_test --n-neg-col n_neg_test

Also importable — see check_ratchet() for use from check_submission_files.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path

_DEFAULT_BASELINES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ratchet_baselines.json')

import numpy as np
import pandas as pd

# Confirmed against a real loso_results.csv (2026-07-08, CatBoost run):
# columns are fold, holdout_site, n_train, n_pos_train, n_neg_train, n_test,
# n_pos_test, n_neg_test, age_cond_auroc, n_age_pairs, std_auroc, accuracy,
# sensitivity, specificity, ppv, TP, FP, FN, TN, top5_features. Candidates
# below are ordered with the confirmed real names first; older/alternate
# names kept as fallbacks in case a different loso_cv.py revision is used.
_AUROC_COL_CANDIDATES = ['age_cond_auroc', 'age_conditioned_auroc', 'auroc_age', 'auroc']
_N_POS_COL_CANDIDATES = ['n_pos_test', 'n_pos', 'num_pos', 'n_positive']
_N_NEG_COL_CANDIDATES = ['n_neg_test', 'n_neg', 'num_neg', 'n_negative']
_SITE_COL_CANDIDATES = ['holdout_site', 'fold', 'site']


def _resolve_column(df, candidates, explicit, quantity_name):
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"--{quantity_name}-col '{explicit}' not found in loso_results.csv. "
                f"Actual columns: {list(df.columns)}")
        return explicit
    for cand in candidates:
        if cand in df.columns:
            return cand
    raise ValueError(
        f"Could not find a column for '{quantity_name}' in loso_results.csv. "
        f"Tried: {candidates}. Actual columns: {list(df.columns)}. "
        f"Pass --{quantity_name}-col explicitly to point at the right one.")


def hanley_mcneil_se(auroc, n_pos, n_neg):
    """
    Standard error of an AUROC estimate (Hanley & McNeil, 1982).
    Same formula already used by hand in learning_log.md's 2026-07-01
    entry to confirm the Entry 1-3 leaderboard decline was noise.
    """
    if n_pos <= 1 or n_neg <= 1:
        raise ValueError(
            f"Need n_pos > 1 and n_neg > 1 for a Hanley-McNeil SE "
            f"(got n_pos={n_pos}, n_neg={n_neg}).")

    q1 = auroc / (2 - auroc)
    q2 = (2 * auroc ** 2) / (1 + auroc)

    variance = (
        auroc * (1 - auroc)
        + (n_pos - 1) * (q1 - auroc ** 2)
        + (n_neg - 1) * (q2 - auroc ** 2)
    ) / (n_pos * n_neg)

    return float(np.sqrt(max(variance, 0.0)))


def se_of_fold_mean(fold_aurocs_n_pos_n_neg):
    """
    SE of a SIMPLE (unweighted) mean of per-fold AUROCs — matching
    loso_cv.py's own "Mean age-conditioned AUROC" convention (confirmed
    2026-07-08 against a real loso_summary.txt: it's a plain average
    across folds, e.g. (0.5493+0.5393)/2 = 0.5443 for the 2-fold I0006/S0001
    LOSO setup — NOT weighted by each fold's test-set size).

    For n independent fold estimates, Var(mean) = (1/n^2) * sum(Var_i), so
    SE(mean) = sqrt(sum(SE_i^2)) / n. This is standard for an average of
    independent estimates and differs from a single Hanley-McNeil SE
    computed on pooled/summed n_pos+n_neg, which would implicitly (and
    wrongly here) assume one AUROC was computed over the full pooled
    sample rather than averaged from per-site folds.
    """
    n = len(fold_aurocs_n_pos_n_neg)
    if n == 0:
        raise ValueError("Need at least one fold to compute a mean SE.")
    variance_sum = sum(
        hanley_mcneil_se(auroc, n_pos, n_neg) ** 2
        for auroc, n_pos, n_neg in fold_aurocs_n_pos_n_neg
    )
    return float(np.sqrt(variance_sum) / n)


def compare_auroc(candidate_folds, baseline_folds, sigma_threshold=1.0):
    """
    Compares a candidate's mean age-conditioned AUROC against a baseline's,
    where "mean" means the same simple per-fold average loso_cv.py itself
    reports (see se_of_fold_mean). Each *_folds argument is a list of
    (auroc, n_pos, n_neg) tuples, one per LOSO holdout fold.

    One-directional ratchet: only fails on a regression beyond
    sigma_threshold pooled SEs. An improving candidate always passes.
    """
    candidate_auroc = float(np.mean([f[0] for f in candidate_folds]))
    baseline_auroc = float(np.mean([f[0] for f in baseline_folds]))

    se_candidate = se_of_fold_mean(candidate_folds)
    se_baseline = se_of_fold_mean(baseline_folds)
    se_diff = float(np.sqrt(se_candidate ** 2 + se_baseline ** 2))

    delta = candidate_auroc - baseline_auroc
    sigmas = delta / se_diff if se_diff > 0 else float('inf')

    if delta >= 0:
        verdict = 'PASS (improvement)'
        is_regression = False
    elif sigmas >= -sigma_threshold:
        verdict = 'PASS (within noise band)'
        is_regression = False
    else:
        verdict = 'FAIL (regression exceeds noise band)'
        is_regression = True

    return {
        'metric': 'age_cond_auroc',
        'candidate': candidate_auroc,
        'baseline': baseline_auroc,
        'delta': delta,
        'se_diff': se_diff,
        'sigmas': sigmas,
        'sigma_threshold': sigma_threshold,
        'verdict': verdict,
        'is_regression': is_regression,
    }


def compare_reward(candidate_reward, baseline, max_pct_drop=0.15):
    """
    Reward has no clean analytic SE — it's a prevalence-weighted score per
    patient (compute_reward in evaluate_model.py), not a simple binomial
    rate, so Hanley-McNeil doesn't apply. This is a blunt percentage-drop
    heuristic, explicitly informational rather than a rigorous statistical
    gate. A proper version would bootstrap over loso_probabilities.csv
    (resample patients with replacement, recompute reward each time) — not
    implemented here; flag if you want that added.
    """
    baseline_reward = baseline['reward']
    if baseline_reward == 0:
        pct_drop = float('inf') if candidate_reward < 0 else 0.0
    else:
        pct_drop = (baseline_reward - candidate_reward) / abs(baseline_reward)

    if candidate_reward >= baseline_reward:
        verdict = 'PASS (improvement)'
        is_regression = False
    elif pct_drop <= max_pct_drop:
        verdict = 'PASS (within tolerance, NOT statistically rigorous)'
        is_regression = False
    else:
        verdict = 'WARN (drop exceeds tolerance, NOT statistically rigorous)'
        is_regression = True

    return {
        'metric': 'reward',
        'candidate': candidate_reward,
        'baseline': baseline_reward,
        'pct_drop': pct_drop,
        'max_pct_drop': max_pct_drop,
        'verdict': verdict,
        'is_regression': is_regression,
        'note': 'Percentage-drop heuristic only — reward has no analytic SE. '
                'Not as rigorous as the AUROC check.',
    }


def load_baselines(path=_DEFAULT_BASELINES_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {path}. Run from the repo root, or "
            f"pass --baselines-file to point at it explicitly.")
    with open(path) as f:
        data = json.load(f)
    data.pop('_readme', None)
    return data


def summarize_loso_results(csv_path, auroc_col=None, n_pos_col=None, n_neg_col=None,
                            site_col=None):
    """
    Parses per-fold LOSO results into the (auroc, n_pos, n_neg) tuple list
    compare_auroc() needs. Does NOT collapse to a single pooled number here
    — the simple-mean-of-folds convention (matching loso_cv.py's own
    "Mean age-conditioned AUROC" line) is applied by the caller via
    compare_auroc(), so this function stays a pure, honest parse of the CSV.
    """
    df = pd.read_csv(csv_path)

    auroc_col = _resolve_column(df, _AUROC_COL_CANDIDATES, auroc_col, 'auroc')
    n_pos_col = _resolve_column(df, _N_POS_COL_CANDIDATES, n_pos_col, 'n-pos')
    n_neg_col = _resolve_column(df, _N_NEG_COL_CANDIDATES, n_neg_col, 'n-neg')
    try:
        site_col = _resolve_column(df, _SITE_COL_CANDIDATES, site_col, 'site')
        sites = df[site_col].astype(str).tolist()
    except ValueError:
        sites = [f'fold_{i}' for i in range(len(df))]  # site label is cosmetic only

    folds = list(zip(df[auroc_col].astype(float), df[n_pos_col].astype(int),
                      df[n_neg_col].astype(int)))
    return folds, sites


def _folds_from_baseline(baseline):
    """Builds the (auroc, n_pos, n_neg) tuple list from a baseline dict's
    'folds' list. Raises a clear error if the baseline predates this schema
    (i.e. still uses the old pooled n_pos/n_neg format)."""
    if 'folds' not in baseline:
        raise ValueError(
            "This baseline uses the old pooled-total schema (n_pos/n_neg at "
            "the top level) and can't be used with the corrected per-fold SE "
            "calculation. Update ratchet_baselines.json to include a 'folds' "
            "list: [{'site':..., 'age_cond_auroc':..., 'n_pos':..., 'n_neg':...}, ...]")
    return [(f['age_cond_auroc'], f['n_pos'], f['n_neg']) for f in baseline['folds']]


def check_ratchet(loso_results_csv, baseline_key, baselines_file=_DEFAULT_BASELINES_PATH,
                   candidate_reward=None, sigma_threshold=1.0, max_reward_pct_drop=0.15,
                   auroc_col=None, n_pos_col=None, n_neg_col=None, site_col=None):
    """
    Main entry point — also called from check_submission_files.py.
    Returns (passed: bool, results: list[dict], messages: list[str]).
    """
    messages = []
    baselines = load_baselines(baselines_file)

    if baseline_key not in baselines:
        raise KeyError(
            f"Baseline key '{baseline_key}' not found in {baselines_file}. "
            f"Available: {list(baselines.keys())}")

    baseline = baselines[baseline_key]
    baseline_folds = _folds_from_baseline(baseline)

    if not baseline.get('leaderboard_confirmed', False):
        messages.append(
            f"CAUTION: baseline '{baseline_key}' is LOSO-only — it has never been "
            f"submitted to the real leaderboard (leaderboard_confirmed=false). "
            f"A PASS here means 'not worse than your best LOSO estimate so far', "
            f"NOT 'confirmed to perform on the real leaderboard'. Your only two "
            f"known LOSO->leaderboard deltas (both at small scale) had DIFFERENT "
            f"signs of surprise (+0.125, then +0.076) — do not assume this offset "
            f"transfers to large scale, where you have zero leaderboard data points "
            f"so far."
        )

    candidate_folds, candidate_sites = summarize_loso_results(
        loso_results_csv, auroc_col, n_pos_col, n_neg_col, site_col)

    if len(candidate_folds) != len(baseline_folds):
        messages.append(
            f"candidate has {len(candidate_folds)} fold(s), baseline has "
            f"{len(baseline_folds)} — simple-mean comparison still works but isn't "
            f"apples-to-apples if these represent different LOSO rotations "
            f"(e.g. 2-site vs 3-site with I0002 included).")

    auroc_result = compare_auroc(candidate_folds, baseline_folds, sigma_threshold=sigma_threshold)
    auroc_result['candidate_sites'] = candidate_sites
    results = [auroc_result]

    if candidate_reward is not None:
        results.append(compare_reward(
            candidate_reward, baseline, max_pct_drop=max_reward_pct_drop))
    else:
        messages.append(
            "No --candidate-reward passed — reward ratchet check skipped. "
            "AUROC-only gate is NOT a full picture; pass reward explicitly "
            "once you have a pooled threshold sweep result.")

    passed = not any(r['is_regression'] for r in results)
    return passed, results, messages


def _print_report(baseline_key, results, messages):
    print(f"Ratchet check against baseline: {baseline_key}\n")

    cautions = [m for m in messages if m.startswith('CAUTION')]
    other_messages = [m for m in messages if not m.startswith('CAUTION')]
    for c in cautions:
        print(f"!! {c}\n")

    for r in results:
        print(f"  [{r['metric']}]")
        print(f"    candidate: {r['candidate']:.4f}   baseline: {r['baseline']:.4f}")
        if r['metric'] == 'age_cond_auroc':
            sites = r.get('candidate_sites')
            if sites:
                print(f"    candidate folds: {sites}")
            print(f"    delta: {r['delta']:+.4f}   SE_diff: {r['se_diff']:.4f}   "
                  f"sigmas: {r['sigmas']:+.2f} (threshold: {r['sigma_threshold']})")
        else:
            print(f"    pct_drop: {r['pct_drop']:.1%}   tolerance: {r['max_pct_drop']:.1%}")
            print(f"    note: {r['note']}")
        print(f"    verdict: {r['verdict']}\n")

    for m in other_messages:
        print(f"NOTE: {m}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--loso-results', required=True,
                         help='Path to loso_cv.py\'s per-fold results CSV.')
    parser.add_argument('--baseline', required=True,
                         help='Key into ratchet_baselines.json (e.g. small_entry3).')
    parser.add_argument('--baselines-file', default=_DEFAULT_BASELINES_PATH)
    parser.add_argument('--candidate-reward', type=float, default=None,
                         help='Pooled reward at your chosen threshold, if you have one '
                              '(from a separate threshold-sweep step). Optional.')
    parser.add_argument('--sigma-threshold', type=float, default=1.0,
                         help='AUROC regression must exceed this many pooled SEs to fail. '
                              'Default 1.0 (matches the "noise floor" framing already used '
                              'in learning_log.md).')
    parser.add_argument('--max-reward-pct-drop', type=float, default=0.15,
                         help='Reward regression tolerance as a fraction of baseline. '
                              'Default 0.15 (15%%). NOT statistically derived.')
    parser.add_argument('--auroc-col', default=None)
    parser.add_argument('--n-pos-col', default=None)
    parser.add_argument('--n-neg-col', default=None)
    parser.add_argument('--site-col', default=None,
                         help='Optional — used only for display, not the SE calculation.')
    args = parser.parse_args()

    try:
        passed, results, messages = check_ratchet(
            args.loso_results, args.baseline, args.baselines_file,
            candidate_reward=args.candidate_reward,
            sigma_threshold=args.sigma_threshold,
            max_reward_pct_drop=args.max_reward_pct_drop,
            auroc_col=args.auroc_col, n_pos_col=args.n_pos_col, n_neg_col=args.n_neg_col,
            site_col=args.site_col,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    _print_report(args.baseline, results, messages)
    print('PASS' if passed else 'FAIL')
    sys.exit(0 if passed else 1)