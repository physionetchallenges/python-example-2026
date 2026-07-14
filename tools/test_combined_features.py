#!/usr/bin/env python3
"""
test_combined_features.py — Team Narnia

Combined-feature ablation for NF1 (Markov transitions, 9 features) and
NF2 (arousal clustering, 2 features) — tested TOGETHER as an 11-feature
block against the shipped 48-feature baseline, NOT one-at-a-time.

This deliberately departs from the project's one-variable-per-test
discipline (learning_log.md, 2026-07-01 decision log). Justification,
per TEST_PLAN.md's dependency graph: RB1 (2026-07-13) ruled out the
regularization-budget explanation for the 3-for-3 CAISR null pattern
(mean_wake_bout_duration, rem_latency, no_arousal_entropy — all null at
both C=0.01 and C=1.0). With regularization budget eliminated, the
remaining live hypothesis for why real univariate EDA signal keeps
dying in the full pipeline is that single-feature ablation is
structurally blind to INTERACTION effects — a feature that only helps
conditional on another feature already being present. Testing NF1+NF2
combined is the direct test of that hypothesis. This is branch (a) of
the RB1-null decision point; branch (b) (ceiling + calibration/threshold
tuning) was not chosen this round.

REQUIRED collinearity checks (per TEST_PLAN.md's NF1/NF2 internal
dependencies) are computed and printed BEFORE the ablation runs, not
buried afterward — a null or positive ablation result is uninterpretable
without knowing whether the new features are redundant with what's
already shipped:
    - NF1's stationary distribution vs. existing Wake_pct/N1_pct/N2_pct/
      N3_pct/REM_pct (caisr_base.py)
    - NF2's CV/burst_index vs. existing Arousal_idx/Spontaneous_arousal_idx
      (caisr_base.py / caisr_enriched.py)

Same fixed config as test_spectral_ablation.py: logreg, l2, calibrated
(cv5), AgeResidualizer on. --C defaults to 0.01 (shipped value) since
RB1 already established C=1.0 doesn't help single-feature candidates —
no a priori reason a combined test needs different regularization, but
the flag is included for consistency/completeness, not because it's
expected to matter here.

Usage:
    python tools/test_combined_features.py \\
        --data /path/to/training_set \\
        --out  outputs/nf1_nf2_combined \\
        --rotate-i0002

Outputs:
    collinearity_report.txt   — correlations, printed AND written, so
                                 they aren't lost if the run is re-piped.
    ablation_results_combined.csv
    ablation_summary_combined.txt
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_code import (
    find_patients, load_demographics, load_signal_data,
    load_diagnoses, DEMOGRAPHICS_FILE,
    PHYSIOLOGICAL_DATA_SUBFOLDER,
    ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
    HEADERS,
)
from features.demographic          import extract_demographic_features
from features.caisr_base           import extract_caisr_base_features
from features.caisr_enriched       import (
    extract_caisr_enriched_features,
    extract_markov_transition_features, N_MARKOV_FEATURES,
    extract_arousal_clustering_features, N_AROUSAL_CLUSTERING_FEATURES,
)
from features.physiological_ratios import (extract_physiological_ratio_features,
                                            N_RATIO_FEATURES)
from features import N_CAISR_BASE_FEATURES, N_CAISR_ENRICHED_FEATURES
from features.pipeline import build_logreg_pipeline
from evaluate_model import compute_reward, compute_prevalence

_N_BASE     = N_CAISR_BASE_FEATURES
_N_ENRICHED = N_CAISR_ENRICHED_FEATURES
_N_RATIO    = N_RATIO_FEATURES
N_COMBINED  = N_MARKOV_FEATURES + N_AROUSAL_CLUSTERING_FEATURES  # 11

# Same denylist as every other large-set-comparable script.
KNOWN_MISLABELED_PATIENT_IDS = {'115257116'}

# Indices into the shipped 48-vector for the collinearity check. Per
# FEATURES.md: demo(10) + caisr_base(12) + caisr_enriched(11) + ratios(15).
# caisr_base occupies [10:22]; within it (FEATURES.md rows 59-70):
#   Wake_pct=idx13(local2), N1_pct=14(3), N2_pct=15(4), N3_pct=16(5),
#   REM_pct=17(6), Arousal_idx=idx11(local1) — CONFIRM against your
# actual caisr_base.py return order before trusting this block; written
# from FEATURES.md's documented order, not verified against the live
# extraction function's actual array order.
_IDX_AROUSAL_IDX = 10 + 1      # caisr_base[1] = Arousal_idx      ✓
# caisr_base order (verified against caisr_base.py return array, 2026-07-14):
#   [0] AHI_total  [1] Arousal_idx  [2] Limb_idx  [3] Wake%  [4] N1%
#   [5] N2%  [6] N3%  [7] REM%  [8] Sleep_eff  [9] Prob_W  [10] Prob_N3  [11] Prob_arous
# The original FEATURES.md comment (Wake_pct=local2) was wrong — caisr_base[2]
# is Limb_idx; Wake% starts at local index 3. Fixed here 2026-07-14.
_IDX_WAKE_PCT    = 10 + 3      # caisr_base[3] = Wake %
_IDX_N1_PCT      = 10 + 4      # caisr_base[4] = N1 %
_IDX_N2_PCT      = 10 + 5      # caisr_base[5] = N2 %
_IDX_N3_PCT      = 10 + 6      # caisr_base[6] = N3 %
_IDX_REM_PCT     = 10 + 7      # caisr_base[7] = REM %
_IDX_SPONT_AROUSAL = 10 + 12 + 9  # caisr_enriched[9] = Spontaneous_arousal_idx  ✓


def extract_all_features_combined(data_folder, verbose=True):
    """
    Extracts the existing 48-feature vector AND the combined NF1+NF2
    11-feature block in one pass. Same one-EDF-load-per-patient
    discipline as test_spectral_ablation.py's extraction loop.
    """
    demo_path    = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_list = find_patients(demo_path)

    features_48_list = []
    combined_list     = []
    labels_list       = []
    ages_list         = []
    sites_list        = []
    patient_ids       = []
    skipped_denied    = 0

    pbar = tqdm(patient_list, desc='Extracting (48-feature + NF1/NF2 combined)',
                unit='patient', disable=not verbose)

    for record in pbar:
        try:
            pid  = record[HEADERS['bids_folder']]
            site = record[HEADERS['site_id']]
            sess = record[HEADERS['session_id']]

            if verbose:
                pbar.set_postfix({'patient': pid})

            pdata   = load_demographics(demo_path, pid, sess)
            bdsp_id = str(pdata.get('BDSPPatientID', pid))

            if bdsp_id in KNOWN_MISLABELED_PATIENT_IDS:
                skipped_denied += 1
                continue

            demo_f = extract_demographic_features(pdata)
            age    = float(demo_f[0])

            phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                                     site, f'{pid}_ses-{sess}.edf')
            if not os.path.exists(phys_file):
                continue
            phys_data, phys_fs = load_signal_data(phys_file)

            algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                                     site, f'{pid}_ses-{sess}_caisr_annotations.edf')
            if os.path.exists(algo_file):
                algo_data, _ = load_signal_data(algo_file)
                base_f     = extract_caisr_base_features(algo_data)
                enriched_f = extract_caisr_enriched_features(algo_data)
                ratio_f    = extract_physiological_ratio_features(
                    phys_data, phys_fs, algo_data)
                markov_f   = extract_markov_transition_features(algo_data)
                arousal_f  = extract_arousal_clustering_features(algo_data)
                del algo_data
            else:
                base_f     = np.full(_N_BASE,     float('nan'))
                enriched_f = np.full(_N_ENRICHED, float('nan'))
                ratio_f    = np.full(_N_RATIO,    float('nan'))
                markov_f   = np.full(N_MARKOV_FEATURES,             float('nan'))
                arousal_f  = np.full(N_AROUSAL_CLUSTERING_FEATURES, float('nan'))

            del phys_data

            label = load_diagnoses(demo_path, pid)
            if label not in (0, 1):
                continue

            features_48_list.append(
                np.hstack([demo_f, base_f, enriched_f, ratio_f]))
            combined_list.append(np.hstack([markov_f, arousal_f]))
            labels_list.append(label)
            ages_list.append(age)
            sites_list.append(site)
            patient_ids.append(bdsp_id)

            del base_f, enriched_f, ratio_f, markov_f, arousal_f

        except Exception as ex:
            tqdm.write(f'  ! Error on {pid}: {ex}')
            continue

    pbar.close()
    print(f'\nSkipped {skipped_denied} denylisted patient(s).')

    return (
        np.asarray(features_48_list, dtype=np.float32),
        np.asarray(combined_list,    dtype=np.float32),
        np.asarray(labels_list, dtype=int),
        np.asarray(ages_list,   dtype=float),
        np.asarray(sites_list),
        np.asarray(patient_ids),
    )


def collinearity_report(X48, combined):
    """
    Required-before-ablation check (TEST_PLAN.md, NF1/NF2 internal
    dependencies). Pearson correlation, computed on complete-case rows
    only (both sides non-NaN) per pair — sample size varies per pair
    and is reported alongside r, since NF1/NF2's NaN rates differ from
    the existing features'.
    """
    lines = ['=' * 60, '  Collinearity check (required before ablation)', '=' * 60, '']

    stationary_labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
    stationary_cols = combined[:, 4:9]  # NF1 output indices [4:9]
    existing_pct_idx = [_IDX_WAKE_PCT, _IDX_N1_PCT, _IDX_N2_PCT,
                         _IDX_N3_PCT, _IDX_REM_PCT]

    lines.append('NF1 stationary distribution vs. existing stage %:')
    for i, (label, exist_idx) in enumerate(zip(stationary_labels, existing_pct_idx)):
        a = stationary_cols[:, i]
        b = X48[:, exist_idx]
        valid = ~np.isnan(a) & ~np.isnan(b)
        n = int(valid.sum())
        if n > 2:
            r = float(np.corrcoef(a[valid], b[valid])[0, 1])
            lines.append(f'  stationary_{label:<5} vs {label}_pct: r={r:+.3f}  (n={n})')
        else:
            lines.append(f'  stationary_{label:<5} vs {label}_pct: n={n}, too few to compute')

    lines.append('')
    lines.append('NF2 CV/burst_index vs. existing arousal-rate features:')
    cv_col    = combined[:, 9]
    burst_col = combined[:, 10]
    for name, col in [('cv_inter_arousal', cv_col), ('burst_index', burst_col)]:
        for exist_name, exist_idx in [('Arousal_idx', _IDX_AROUSAL_IDX),
                                       ('Spontaneous_arousal_idx', _IDX_SPONT_AROUSAL)]:
            b = X48[:, exist_idx]
            valid = ~np.isnan(col) & ~np.isnan(b)
            n = int(valid.sum())
            if n > 2:
                r = float(np.corrcoef(col[valid], b[valid])[0, 1])
                lines.append(f'  {name} vs {exist_name}: r={r:+.3f}  (n={n})')
            else:
                lines.append(f'  {name} vs {exist_name}: n={n}, too few to compute')

    lines.append('')
    lines.append('High |r| (>0.9) means that component is likely redundant with an')
    lines.append('existing feature — a null OR positive ablation result for that')
    lines.append('specific component should be read in that light, per TEST_PLAN.md.')
    lines.append('Index assumptions (_IDX_* constants) are taken from FEATURES.md\'s')
    lines.append('documented order, NOT independently re-verified against the live')
    lines.append('caisr_base.py/caisr_enriched.py return arrays — confirm before')
    lines.append('trusting these numbers if anything looks implausible (e.g. |r|')
    lines.append('near 0 for stationary_Wake vs Wake_pct would be a red flag).')

    return '\n'.join(lines)


def age_conditioned_auroc(labels, probs, ages, gap=2):
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    numer = 0.0
    denom = 0
    for i in idx_pos:
        for j in idx_neg:
            if abs(ages[i] - ages[j]) <= gap:
                if probs[i] > probs[j]:
                    numer += 1.0
                elif probs[i] == probs[j]:
                    numer += 0.5
                denom += 1
    if denom == 0:
        return float('nan'), 0
    return round(numer / denom, 4), denom


def hanley_mcneil_se(auc, n_pos, n_neg):
    Q1 = auc / (2 - auc)
    Q2 = 2 * auc**2 / (1 + auc)
    var = (auc*(1-auc) + (n_pos-1)*(Q1-auc**2) + (n_neg-1)*(Q2-auc**2)) / (n_pos*n_neg)
    return np.sqrt(var)


def run_fold(holdout_site, X, y, ages, sites, age_to_prevalence, label, C=0.01):
    train_mask = sites != holdout_site
    test_mask  = sites == holdout_site

    y_train = y[train_mask]
    y_test  = y[test_mask]
    ages_test = ages[test_mask]

    if int(y_test.sum()) < 3:
        print(f'  SKIP {label} — fewer than 3 positives in {holdout_site}.')
        return None

    model = build_logreg_pipeline(y_train, calibrated=True,
                                   use_age_residual=True,
                                   C=C, penalty=None)
    model.fit(X[train_mask], y_train)
    probs = model.predict_proba(X[test_mask])[:, 1]

    auroc, n_pairs = age_conditioned_auroc(y_test, probs, ages_test)
    preds = (probs > 0.10).astype(int)  # shipped Entry 5 threshold
    reward = compute_reward(y_test, preds, ages_test, age_to_prevalence)

    n_pos = int(y_test.sum())
    n_neg = int((y_test == 0).sum())

    return {
        'config': label,
        'holdout_site': holdout_site,
        'C': C,
        'n_pos_test': n_pos,
        'n_neg_test': n_neg,
        'age_cond_auroc': auroc,
        'n_age_pairs': n_pairs,
        'reward_at_t0.10': round(reward, 4),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--C', type=float, default=0.01,
                   help='Logistic regression C. Default 0.01 matches shipped '
                        'config; RB1 (2026-07-13) already showed C=1.0 does not '
                        'help single-feature candidates, so this is included for '
                        'completeness, not because it is expected to matter here.')
    p.add_argument('--rotate-i0002', action='store_true')
    p.add_argument('--verbose', action='store_true', default=True)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Extracting 48-feature vector + NF1 (Markov) + NF2 (arousal clustering) ...')
    X48, combined, y, ages, sites, patient_ids = extract_all_features_combined(
        args.data, verbose=args.verbose)

    print(f'\nExtracted {len(y)} patients ({int(y.sum())} positive)')
    print(f'X48 shape: {X48.shape}, combined (NF1+NF2) shape: {combined.shape}')

    nan_rates = np.isnan(combined).mean(axis=0)
    labels = ['transition_entropy', 'n3_to_wake_rate', 'rem_to_n3_rate',
              'escalating_rate', 'stationary_Wake', 'stationary_N1',
              'stationary_N2', 'stationary_N3', 'stationary_REM',
              'cv_inter_arousal', 'burst_index']
    print('\nPer-feature NaN rate:')
    for lbl, rate in zip(labels, nan_rates):
        print(f'  {lbl:<22}: {rate:.1%}')

    report = collinearity_report(X48, combined)
    print('\n' + report)
    (out_dir / 'collinearity_report.txt').write_text(report)

    X59 = np.hstack([X48, combined])

    age_to_prevalence = compute_prevalence(ages, y, ages, gap=2)

    holdout_sites = ['I0006', 'S0001']
    if args.rotate_i0002:
        holdout_sites.append('I0002')

    rows = []
    for holdout in holdout_sites:
        print(f'\n--- Fold: hold out {holdout} ---')
        r_without = run_fold(holdout, X48, y, ages, sites, age_to_prevalence,
                             label='WITHOUT NF1+NF2 (shipped 48-feature)',
                             C=args.C)
        r_with    = run_fold(holdout, X59, y, ages, sites, age_to_prevalence,
                             label='WITH NF1+NF2 combined (59-feature)',
                             C=args.C)
        if r_without: rows.append(r_without)
        if r_with:    rows.append(r_with)

        if r_without and r_with:
            se_a = hanley_mcneil_se(r_without['age_cond_auroc'],
                                     r_without['n_pos_test'], r_without['n_neg_test'])
            se_b = hanley_mcneil_se(r_with['age_cond_auroc'],
                                     r_with['n_pos_test'], r_with['n_neg_test'])
            se_diff = np.sqrt(se_a**2 + se_b**2)
            z = (r_with['age_cond_auroc'] - r_without['age_cond_auroc']) / se_diff
            print(f"  AUROC: without={r_without['age_cond_auroc']:.4f}  "
                  f"with={r_with['age_cond_auroc']:.4f}  "
                  f"delta={r_with['age_cond_auroc']-r_without['age_cond_auroc']:+.4f}  "
                  f"z={z:+.2f}sigma")
            print(f"  Reward@0.10: without={r_without['reward_at_t0.10']:.4f}  "
                  f"with={r_with['reward_at_t0.10']:.4f}")

    results_df = pd.DataFrame(rows)
    out_csv = out_dir / 'ablation_results_combined.csv'
    results_df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')

    summary_lines = [
        '=' * 60,
        '  NF1 (Markov transitions) + NF2 (arousal clustering) — Combined Ablation',
        '=' * 60,
        '',
        'Fixed config: logreg, C={}, l2, calibrated (cv5), AgeResidualizer on'.format(args.C),
        'Only variable: presence/absence of the combined 11-feature block',
        '(NOT one-at-a-time — this is branch (a) after RB1\'s null, testing',
        'for interaction effects invisible to single-feature ablation)',
        '',
        results_df.to_string(index=False),
    ]
    summary = '\n'.join(summary_lines)
    (out_dir / 'ablation_summary_combined.txt').write_text(summary)
    print('\n' + summary)


if __name__ == '__main__':
    main()