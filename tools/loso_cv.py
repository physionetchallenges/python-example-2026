#!/usr/bin/env python3
"""
loso_cv.py — Team Narnia
Leave-One-Site-Out cross-validation for honest local performance estimation.

Trains on two sites, evaluates on the third, rotates through all meaningful
holdout configurations. Produces age-conditioned AUROC to match the actual
challenge scoring metric.

Usage:
    python loso_cv.py \
        --data /path/to/training_set_small \
        --out  ./loso_outputs

Outputs (all in --out directory):
    loso_results.csv        — per-fold per-feature-set results
    loso_feature_impact.csv — feature importance delta vs baseline per fold
    loso_summary.txt        — human-readable summary

Notes:
    - I0002 has ~54 patients (~10 positives) — too small to be a reliable
      holdout. It is always included in training. Only S0001 and I0006 rotate
      as holdout sites.
    - Age-conditioned AUROC uses the same ±2yr window as evaluate_model.py.
    - Run this after train_model.py has completed at least once so you know
      the feature extraction pipeline is working end-to-end.
    - Expect ~35-40 minutes total (two full training runs).
"""
#!/usr/bin/env python3
"""
loso_cv.py — Team Narnia
Leave-One-Site-Out cross-validation for honest local performance estimation.

RECONCILED 2026-07-09: the local copy of this file had fallen behind every
Kaggle dataset version actually used to produce real results (confirmed via
grep against the real local file -- it had none of --model-family,
--calibration, --tune-hyperparams, --diagnostic-models). This version
reconciles the most complete Kaggle snapshot seen (matching the CONFIRMED
real features/pipeline.py signature: build_logreg_pipeline(y_train,
calibrated, use_age_residual, calibration_ensemble, C, penalty, l1_ratio,
max_iter)) back into the local repo, PLUS adds --penalty/--l1-ratio CLI
passthrough (new, 2026-07-09) for the Ridge/Lasso/ElasticNet regularization
sweep.

STILL MISSING, confirmed absent from every version seen so far, local or
Kaggle: --rotate-i0002 (3-site LOSO rotation). A run using 3-site rotation
DID happen (loso_summary.txt printed "LOSO rotation: I0006, S0001, I0002
(3-site)") but the code that produced it hasn't surfaced in any file shared
so far -- holdout_sites is still hardcoded to 2-site here.
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
from features.caisr_enriched       import extract_caisr_enriched_features
from features.physiological_ratios import (extract_physiological_ratio_features,
                                            N_RATIO_FEATURES)
from features import N_CAISR_BASE_FEATURES, N_CAISR_ENRICHED_FEATURES

_N_BASE     = N_CAISR_BASE_FEATURES
_N_ENRICHED = N_CAISR_ENRICHED_FEATURES
_N_RATIO    = N_RATIO_FEATURES

from features.pipeline import build_pipeline, build_logreg_pipeline, PrefitCalibratedModel
from features.age_residuals import AgeResidualizer
from features import IDX_AGE, IDX_CA_RATE, IDX_EEG_VAR_REM_WAKE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

from evaluate_model import compute_reward, compute_prevalence

ENTRY3_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

FEATURE_NAMES = [
    'Age','Sex_F','Sex_M','Sex_U',
    'Race_As','Race_Bl','Race_Ot','Race_Un','Race_Wh','BMI',
    'AHI_total','Arousal_idx','Limb_idx',
    'W_pct','N1_pct','N2_pct','N3_pct','REM_pct','Sleep_eff',
    'Prob_W','Prob_N3','Prob_arous',
    'OA_rate','CA_rate','HY_rate','RERA_rate',
    'CA_total_ratio','REM_AHI','NREM_AHI','REM_NREM_ratio',
    'N3_gradient','Spont_arousal_idx','N3_entropy',
    'EEG_std_N3_Wake','EEG_mav_N3_Wake','EEG_zcr_N3_Wake',
    'EEG_rms_N3_Wake','EEG_var_N3_Wake','EEG_mob_N3_Wake','EEG_cplx_N3_Wake',
    'EEG_std_REM_Wake','EEG_mav_REM_Wake','EEG_zcr_REM_Wake',
    'EEG_rms_REM_Wake','EEG_var_REM_Wake','EEG_mob_REM_Wake','EEG_cplx_REM_Wake',
    'Chin_atonia_ratio',
    'CA_rate_age_resid','EEG_var_REM_Wake_age_resid',
]


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


def standard_auroc(labels, probs):
    from scipy.stats import mannwhitneyu
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    U, _ = mannwhitneyu(pos, neg, alternative='greater')
    return round(U / (len(pos) * len(neg)), 4)


def extract_all_features(data_folder, verbose=True):
    demo_path     = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_list  = find_patients(demo_path)

    features_list = []
    labels_list   = []
    ages_list     = []
    sites_list    = []
    patient_ids   = []

    pbar = tqdm(patient_list, desc='Extracting features', unit='patient',
                disable=not verbose)

    for record in pbar:
        try:
            pid  = record[HEADERS['bids_folder']]
            site = record[HEADERS['site_id']]
            sess = record[HEADERS['session_id']]

            if verbose:
                pbar.set_postfix({'patient': pid})

            pdata  = load_demographics(demo_path, pid, sess)
            demo_f = extract_demographic_features(pdata)
            age    = float(demo_f[0])
            bdsp_id = pdata.get('BDSPPatientID', pid)

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
                del algo_data
            else:
                base_f     = np.full(_N_BASE,     float('nan'))
                enriched_f = np.full(_N_ENRICHED, float('nan'))
                ratio_f    = np.full(_N_RATIO,    float('nan'))

            del phys_data

            label = load_diagnoses(demo_path, pid)
            if label not in (0, 1):
                continue

            features_list.append(
                np.hstack([demo_f, base_f, enriched_f, ratio_f]))
            labels_list.append(label)
            ages_list.append(age)
            sites_list.append(site)
            patient_ids.append(str(bdsp_id))

            del base_f, enriched_f, ratio_f

        except Exception as ex:
            tqdm.write(f'  ! Error on {pid}: {ex}')
            continue

    pbar.close()

    return (
        np.asarray(features_list, dtype=np.float32),
        np.asarray(labels_list,   dtype=int),
        np.asarray(ages_list,     dtype=float),
        np.asarray(sites_list),
        np.asarray(patient_ids),
    )


def build_model(X_train, y_train, use_age_residual=True, model_family='xgboost',
                 C=0.01, penalty=None, l1_ratio=None):
    """Uncalibrated model.

    C/penalty/l1_ratio (2026-07-09): only used when model_family='logreg'.
    Default C=0.01, penalty=None reproduces the original tested config.
    """
    if model_family == 'logreg':
        model = build_logreg_pipeline(y_train, calibrated=False, use_age_residual=use_age_residual,
                                       C=C, penalty=penalty, l1_ratio=l1_ratio)
    elif model_family == 'catboost':
        model = build_catboost_pipeline(y_train, calibrated=False, use_age_residual=use_age_residual)
    else:
        model = build_pipeline(y_train, calibrated=False, use_age_residual=use_age_residual)
    model.fit(X_train, y_train)
    return model


def build_calibrated_model(X_train, y_train, use_age_residual=True,
                            calibration_mode='cv5', model_family='xgboost',
                            C=0.01, penalty=None, l1_ratio=None):
    if calibration_mode == 'prefit':
        if model_family != 'xgboost':
            raise ValueError('calibration_mode="prefit" is XGBoost-specific '
                              '(PrefitCalibratedModel) — not supported with '
                              f'model_family="{model_family}".')
        model = PrefitCalibratedModel(use_age_residual=use_age_residual)
        model.fit(X_train, y_train)
        return model

    calibration_ensemble = (calibration_mode != 'cv5-single-curve')
    if model_family == 'logreg':
        model = build_logreg_pipeline(y_train, calibrated=True, use_age_residual=use_age_residual,
                                       calibration_ensemble=calibration_ensemble,
                                       C=C, penalty=penalty, l1_ratio=l1_ratio)
    elif model_family == 'catboost':
        model = build_catboost_pipeline(y_train, calibrated=True, use_age_residual=use_age_residual,
                                         calibration_ensemble=calibration_ensemble)
    else:
        model = build_pipeline(y_train, calibrated=True, use_age_residual=use_age_residual,
                                calibration_ensemble=calibration_ensemble)
    model.fit(X_train, y_train)
    return model


def sweep_thresholds(probs, labels, ages, age_to_prevalence,
                      thresholds=ENTRY3_THRESHOLDS, label=''):
    print(f'\n  Threshold sweep {label}:')
    rows = []
    for t in thresholds:
        preds = (probs > t).astype(int)
        reward = compute_reward(labels, preds, ages, age_to_prevalence)
        auroc, _ = age_conditioned_auroc(labels, probs, ages)
        rows.append({'threshold': t, 'reward': round(reward, 4), 'auroc': auroc})
        print(f'    t={t:.2f}  reward={reward:.4f}  auroc={auroc:.4f}')
    return rows


def run_fold(holdout_site, features, labels, ages, sites, patient_ids,
             age_to_prevalence, fold_name, use_age_residual=True,
             calibration_mode='cv5', model_family='xgboost',
             C=0.01, penalty=None, l1_ratio=None):
    train_mask = (sites != holdout_site)
    test_mask  = (sites == holdout_site)

    X_train = features[train_mask]
    y_train = labels[train_mask]
    X_test  = features[test_mask]
    y_test  = labels[test_mask]
    ages_test = ages[test_mask]
    ids_test  = patient_ids[test_mask]
    sites_test = sites[test_mask]

    n_pos_train = int(y_train.sum())
    n_neg_train = int((y_train == 0).sum())
    n_pos_test  = int(y_test.sum())
    n_neg_test  = int((y_test == 0).sum())

    print(f'\n  Train: {len(y_train)} patients '
          f'({n_pos_train} pos, {n_neg_train} neg)')
    print(f'  Test:  {len(y_test)} patients '
          f'({n_pos_test} pos, {n_neg_test} neg)')

    if n_pos_test < 3:
        print(f'  SKIP — fewer than 3 positives in holdout site.')
        return None, []

    print('  Fitting model (uncalibrated) ...')
    model = build_model(X_train, y_train, use_age_residual=use_age_residual,
                         model_family=model_family, C=C, penalty=penalty, l1_ratio=l1_ratio)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print(f'  Fitting model (Platt-calibrated, mode={calibration_mode}, family={model_family}) ...')
    calibrated_model = build_calibrated_model(X_train, y_train,
                                               use_age_residual=use_age_residual,
                                               calibration_mode=calibration_mode,
                                               model_family=model_family,
                                               C=C, penalty=penalty, l1_ratio=l1_ratio)
    calibrated_probs = calibrated_model.predict_proba(X_test)[:, 1]

    age_auroc, n_pairs = age_conditioned_auroc(y_test, probs, ages_test)
    std_auroc  = standard_auroc(y_test, probs)
    accuracy   = float((preds == y_test).mean())
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    tn = int(((preds == 0) & (y_test == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float('nan')

    sweep_thresholds(calibrated_probs, y_test, ages_test, age_to_prevalence,
                      label=f'(holdout={holdout_site}, calibrated)')

    classifier_step = model.named_steps['classifier']
    if hasattr(classifier_step, 'feature_importances_'):
        imp = classifier_step.feature_importances_
    elif hasattr(classifier_step, 'coef_'):
        imp = np.abs(classifier_step.coef_[0])
    else:
        imp = np.zeros(features.shape[1])
    top5_idx  = np.argsort(imp)[::-1][:5]
    top5_feat = [(FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f'feat_{i}',
                  round(float(imp[i]), 4)) for i in top5_idx]

    result = {
        'fold':           fold_name,
        'holdout_site':   holdout_site,
        'n_train':        len(y_train),
        'n_pos_train':    n_pos_train,
        'n_neg_train':    n_neg_train,
        'n_test':         len(y_test),
        'n_pos_test':     n_pos_test,
        'n_neg_test':     n_neg_test,
        'age_cond_auroc': age_auroc,
        'n_age_pairs':    n_pairs,
        'std_auroc':      std_auroc,
        'accuracy':       round(accuracy, 4),
        'sensitivity':    round(sensitivity, 4),
        'specificity':    round(specificity, 4),
        'ppv':            round(ppv, 4),
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'top5_features':  str(top5_feat),
    }

    print(f'  Age-conditioned AUROC : {age_auroc:.4f}  ({n_pairs} pairs)')
    print(f'  Standard AUROC        : {std_auroc:.4f}')
    print(f'  Sensitivity           : {sensitivity:.3f}')
    print(f'  Specificity           : {specificity:.3f}')
    print(f'  Top features: {top5_feat[:3]}')

    per_patient_rows = []
    for i in range(len(ids_test)):
        per_patient_rows.append({
            'BDSPPatientID':          ids_test[i],
            'SiteID':                 sites_test[i],
            'Age':                    ages_test[i],
            'label':                  int(y_test[i]),
            'raw_probability':        round(float(probs[i]), 6),
            'calibrated_probability': round(float(calibrated_probs[i]), 6),
        })

    return result, per_patient_rows


def run_ablation(holdout_site, features, labels, ages, sites):
    GROUPS = {
        'demo':           list(range(0,  10)),
        'caisr_base':     list(range(10, 22)),
        'caisr_enriched': list(range(22, 33)),
        'ratios':         list(range(33, 48)),
    }

    train_mask = sites != holdout_site
    test_mask  = sites == holdout_site
    ages_test  = ages[test_mask]
    y_test     = labels[test_mask]

    if y_test.sum() < 3:
        return []

    baseline_model = build_model(features[train_mask], labels[train_mask])
    baseline_probs = baseline_model.predict_proba(features[test_mask])[:, 1]
    baseline_auroc, _ = age_conditioned_auroc(y_test, baseline_probs, ages_test)

    rows = []
    for group_name, indices in GROUPS.items():
        X_ablated = features.copy()
        X_ablated[:, indices] = float('nan')

        ablated_model = build_model(X_ablated[train_mask], labels[train_mask])
        ablated_probs = ablated_model.predict_proba(X_ablated[test_mask])[:, 1]
        ablated_auroc, _ = age_conditioned_auroc(y_test, ablated_probs, ages_test)

        delta = round(ablated_auroc - baseline_auroc, 4)
        rows.append({
            'holdout_site':   holdout_site,
            'ablated_group':  group_name,
            'n_features':     len(indices),
            'baseline_auroc': baseline_auroc,
            'ablated_auroc':  ablated_auroc,
            'delta':          delta,
            'interpretation': (
                'HURTS generalisation'  if delta < -0.02 else
                'HELPS generalisation'  if delta >  0.02 else
                'NEUTRAL'
            ),
        })
        print(f'    Ablate {group_name:<18s}: AUROC {ablated_auroc:.4f}  '
              f'(delta={delta:+.4f})')

    return rows


HYPERPARAM_CANDIDATES = {
    'baseline':            {},
    'more_l2':             {'reg_lambda': 10.0},
    'more_min_child_wt':   {'min_child_weight': 25},
    'shallower':           {'max_depth': 2},
    'fewer_trees':         {'n_estimators': 100},
    'conservative_combo':  {'max_depth': 2, 'reg_lambda': 8.0,
                             'min_child_weight': 25, 'n_estimators': 150},
}


def run_hyperparameter_sweep(holdout_site, features, labels, ages, sites, use_age_residual=True):
    train_mask = sites != holdout_site
    test_mask = sites == holdout_site
    X_train, y_train = features[train_mask], labels[train_mask]
    X_test, y_test = features[test_mask], labels[test_mask]
    ages_test = ages[test_mask]

    if int(y_test.sum()) < 3:
        print(f'  SKIP — fewer than 3 positives in holdout site {holdout_site}.')
        return []

    rows = []
    for name, overrides in HYPERPARAM_CANDIDATES.items():
        model = build_pipeline(y_train, calibrated=False,
                                use_age_residual=use_age_residual,
                                xgb_overrides=overrides)
        model.fit(X_train, y_train)
        raw_probs = model.predict_proba(X_test)[:, 1]

        auroc, n_pairs = age_conditioned_auroc(y_test, raw_probs, ages_test)
        row = {
            'holdout_site': holdout_site,
            'candidate': name,
            'overrides': str(overrides),
            'raw_median': round(float(np.median(raw_probs)), 4),
            'raw_frac_above_0.5': round(float((raw_probs > 0.5).mean()), 4),
            'raw_frac_above_0.12': round(float((raw_probs > 0.12).mean()), 4),
            'age_cond_auroc': auroc,
            'n_age_pairs': n_pairs,
        }
        rows.append(row)
        print(f"    {name:20s}  raw_median={row['raw_median']:.4f}  "
              f"frac>0.5={row['raw_frac_above_0.5']:.3f}  "
              f"AUROC={auroc}")

    return rows


DIAGNOSTIC_MODELS = {
    'xgboost_baseline':  {'family': 'xgboost',   'params': {}},
    'logreg_strong_l2':  {'family': 'logreg',    'params': {'C': 0.01}},
    'logreg_moderate_l2':{'family': 'logreg',    'params': {'C': 1.0}},
}


def _build_diagnostic_pipeline(y_train, family, params, use_age_residual=True):
    if family == 'xgboost':
        return build_pipeline(y_train, calibrated=False,
                               use_age_residual=use_age_residual,
                               xgb_overrides=params)

    if family == 'logreg':
        steps = []
        if use_age_residual:
            steps.append(('age_residual', AgeResidualizer(
                age_idx=IDX_AGE, ca_rate_idx=IDX_CA_RATE,
                eeg_var_rem_wake_idx=IDX_EEG_VAR_REM_WAKE)))
        steps.append(('imputer', SimpleImputer(strategy='median')))
        steps.append(('scaler', StandardScaler()))
        steps.append(('classifier', LogisticRegression(
            penalty='l2', class_weight='balanced', max_iter=2000,
            random_state=42, **params)))
        return Pipeline(steps)

    raise ValueError(f'Unknown model family: {family}')


def run_model_family_diagnostic(holdout_site, features, labels, ages, sites,
                                 use_age_residual=True):
    train_mask = sites != holdout_site
    test_mask = sites == holdout_site
    X_train, y_train = features[train_mask], labels[train_mask]
    X_test, y_test = features[test_mask], labels[test_mask]
    ages_test = ages[test_mask]

    if int(y_test.sum()) < 3:
        print(f'  SKIP — fewer than 3 positives in holdout site {holdout_site}.')
        return []

    rows = []
    for name, spec in DIAGNOSTIC_MODELS.items():
        model = _build_diagnostic_pipeline(y_train, spec['family'], spec['params'],
                                            use_age_residual=use_age_residual)
        model.fit(X_train, y_train)
        raw_probs = model.predict_proba(X_test)[:, 1]

        auroc, n_pairs = age_conditioned_auroc(y_test, raw_probs, ages_test)
        row = {
            'holdout_site': holdout_site,
            'model': name,
            'family': spec['family'],
            'params': str(spec['params']),
            'raw_median': round(float(np.median(raw_probs)), 4),
            'raw_frac_above_0.5': round(float((raw_probs > 0.5).mean()), 4),
            'raw_frac_above_0.12': round(float((raw_probs > 0.12).mean()), 4),
            'age_cond_auroc': auroc,
            'n_age_pairs': n_pairs,
        }
        rows.append(row)
        print(f"    {name:20s}  raw_median={row['raw_median']:.4f}  "
              f"frac>0.5={row['raw_frac_above_0.5']:.3f}  "
              f"AUROC={auroc}")

    return rows


CATBOOST_PARAMS = dict(
    iterations=300,
    depth=3,
    learning_rate=0.05,
    l2_leaf_reg=2.0,
    random_seed=42,
    verbose=0,
)


def _catboost_scale_pos_weight(y_train):
    y_train = np.asarray(y_train)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    return n_neg / n_pos if n_pos > 0 else 1.0


def build_catboost_pipeline(y_train, calibrated=True, use_age_residual=True,
                             calibration_ensemble=True):
    try:
        from catboost import CatBoostClassifier
    except ImportError as e:
        raise ImportError(
            "model_family='catboost' requires the catboost package, which "
            "isn't installed. Run: pip install catboost"
        ) from e

    spw = _catboost_scale_pos_weight(y_train)
    cb = CatBoostClassifier(scale_pos_weight=spw, **CATBOOST_PARAMS)

    classifier = (
        CalibratedClassifierCV(cb, method='sigmoid', cv=5, ensemble=calibration_ensemble)
        if calibrated else cb
    )

    steps = []
    if use_age_residual:
        steps.append(('age_residual', AgeResidualizer(
            age_idx=IDX_AGE,
            ca_rate_idx=IDX_CA_RATE,
            eeg_var_rem_wake_idx=IDX_EEG_VAR_REM_WAKE,
        )))
    steps.append(('imputer', SimpleImputer(strategy='median')))
    steps.append(('classifier', classifier))

    return Pipeline(steps)


def parse_args():
    p = argparse.ArgumentParser(description='LOSO CV — Narnia_ML')
    p.add_argument('--data',     required=True, help='Training data folder')
    p.add_argument('--out',      required=True, help='Output directory')
    p.add_argument('--ablation', action='store_true',
                   help='Run feature group ablation (adds ~2x runtime)')
    p.add_argument('--no-age-residual', action='store_true',
                   help='Build the Entry-3-EQUIVALENT pipeline (no AgeResidualizer).')
    p.add_argument('--calibration', choices=['cv5', 'cv5-single-curve', 'prefit'], default='cv5',
                   help='Calibration mode. See build_calibrated_model() docstring.')
    p.add_argument('--tune-hyperparams', action='store_true',
                   help='Run the HYPERPARAM_CANDIDATES sweep instead of normal fold evaluation.')
    p.add_argument('--diagnostic-models', action='store_true',
                   help='Run the DIAGNOSTIC_MODELS comparison instead of normal fold evaluation.')
    p.add_argument('--model-family', choices=['xgboost', 'logreg', 'catboost'], default='xgboost',
                   help='Base model family.')
    p.add_argument('--C', type=float, default=0.01,
                   help=('Inverse regularization strength for --model-family logreg. '
                         'Default 0.01 matches build_logreg_pipeline()\'s tested default. '
                         'Ignored for other model families. Added 2026-07-09.'))
    p.add_argument('--penalty', choices=['l1', 'l2', 'elasticnet'], default=None,
                   help=('Regularization penalty for --model-family logreg. Default '
                         '(unset) uses LOGREG_PARAMS\' l2 -- the exact original tested '
                         'config. l1/elasticnet are UNTESTED territory as of this flag '
                         '(2026-07-09). Ignored for other model families.'))
    p.add_argument('--l1-ratio', type=float, default=None,
                   help='Required if --penalty elasticnet. Ignored otherwise.')
    p.add_argument('--verbose',  action='store_true', default=True)
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_age_residual = not args.no_age_residual

    if args.penalty and args.model_family != 'logreg':
        print(f'!! WARNING: --penalty was passed but --model-family is '
              f'"{args.model_family}", not "logreg" -- --penalty/--C/--l1-ratio '
              f'are silently ignored for this model family.\n')

    print(f'\n{"="*60}')
    print('  Leave-One-Site-Out CV — Team Narnia')
    print(f'{"="*60}\n')
    print(f'  Pipeline mode: {"Entry 4 (WITH age-residual features, 50-wide)" if use_age_residual else "Entry 3-EQUIVALENT (NO age-residual features, 48-wide)"}')
    print(f'  Calibration mode: {args.calibration}')
    print(f'  Model family: {args.model_family}')
    if args.model_family == 'logreg':
        print(f'  Logreg C={args.C}  penalty={args.penalty or "l2 (default)"}'
              f'{f"  l1_ratio={args.l1_ratio}" if args.penalty == "elasticnet" else ""}')
    print(f'{"="*60}\n')

    print('Step 1: Extracting features from all patients (~16-18 min) ...\n')
    features, labels, ages, sites, patient_ids = extract_all_features(
        args.data, verbose=args.verbose)

    print(f'\nFeature matrix: {features.shape}')
    print(f'Labels: {labels.sum()} positive, {(labels==0).sum()} negative')
    print(f'Sites: {dict(zip(*np.unique(sites, return_counts=True)))}')

    age_to_prevalence = compute_prevalence(ages, labels, ages, gap=2)

    print('\nSite breakdown:')
    for site in np.unique(sites):
        mask = sites == site
        n_pos = int(labels[mask].sum())
        n_neg = int((labels[mask] == 0).sum())
        print(f'  {site}: {mask.sum()} patients  '
              f'({n_pos} pos / {n_neg} neg)  '
              f'CI rate={n_pos/mask.sum():.1%}')

    if args.tune_hyperparams:
        print(f'\n{"="*60}')
        print('  Hyperparameter sweep mode (--tune-hyperparams)')
        print(f'{"="*60}\n')
        print(f'  Candidates: {list(HYPERPARAM_CANDIDATES.keys())}\n')

        sweep_rows = []
        for holdout in ['I0006', 'S0001']:
            print(f'\n{"─"*50}')
            print(f'  Sweep: hold out {holdout}')
            print(f'{"─"*50}')
            rows = run_hyperparameter_sweep(holdout, features, labels, ages, sites,
                                             use_age_residual=use_age_residual)
            sweep_rows.extend(rows)

        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df.to_csv(out_dir / 'loso_hyperparam_sweep.csv', index=False)
        print(f'\nWrote {out_dir / "loso_hyperparam_sweep.csv"}')
        return

    if args.diagnostic_models:
        print(f'\n{"="*60}')
        print('  Model-family diagnostic mode (--diagnostic-models)')
        print(f'{"="*60}\n')
        print(f'  Models: {list(DIAGNOSTIC_MODELS.keys())}\n')

        diag_rows = []
        for holdout in ['I0006', 'S0001']:
            print(f'\n{"─"*50}')
            print(f'  Diagnostic: hold out {holdout}')
            print(f'{"─"*50}')
            rows = run_model_family_diagnostic(holdout, features, labels, ages, sites,
                                                use_age_residual=use_age_residual)
            diag_rows.extend(rows)

        diag_df = pd.DataFrame(diag_rows)
        diag_df.to_csv(out_dir / 'loso_model_diagnostic.csv', index=False)
        print(f'\nWrote {out_dir / "loso_model_diagnostic.csv"}')
        return

    holdout_sites = ['I0006', 'S0001']

    print('\nStep 2: Running LOSO folds ...')
    fold_results     = []
    ablation_rows    = []
    all_patient_rows = []

    for holdout in holdout_sites:
        fold_name = f'holdout_{holdout}'
        print(f'\n{"─"*50}')
        print(f'  Fold: hold out {holdout}')
        print(f'{"─"*50}')

        result, patient_rows = run_fold(holdout, features, labels, ages, sites,
                                         patient_ids, age_to_prevalence, fold_name,
                                         use_age_residual=use_age_residual,
                                         calibration_mode=args.calibration,
                                         model_family=args.model_family,
                                         C=args.C, penalty=args.penalty, l1_ratio=args.l1_ratio)
        if result:
            fold_results.append(result)
        all_patient_rows.extend(patient_rows)

        if args.ablation:
            print(f'\n  Feature group ablation (holdout={holdout}):')
            ab = run_ablation(holdout, features, labels, ages, sites)
            ablation_rows.extend(ab)

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(out_dir / 'loso_results.csv', index=False)

    if ablation_rows:
        ablation_df = pd.DataFrame(ablation_rows)
        ablation_df.to_csv(out_dir / 'loso_feature_impact.csv', index=False)

    pooled_sweep_rows = []
    if all_patient_rows:
        patients_df = pd.DataFrame(all_patient_rows)
        patients_df.to_csv(out_dir / 'loso_probabilities.csv', index=False)
        print(f'\nWrote per-patient probabilities: '
              f'{out_dir / "loso_probabilities.csv"}  '
              f'({len(patients_df)} patients)')

        pooled_probs  = patients_df['calibrated_probability'].to_numpy()
        pooled_labels = patients_df['label'].to_numpy()
        pooled_ages   = patients_df['Age'].to_numpy()
        pooled_sweep_rows = sweep_thresholds(
            pooled_probs, pooled_labels, pooled_ages, age_to_prevalence,
            label='(POOLED — both folds combined, use this to pick submission threshold)')
        pd.DataFrame(pooled_sweep_rows).to_csv(
            out_dir / 'loso_threshold_sweep.csv', index=False)

    lines = [
        '=' * 60,
        '  LOSO CV SUMMARY — Team Narnia',
        '=' * 60,
        '',
        f'  Pipeline mode: {"Entry 4 (WITH age-residual features, 50-wide)" if use_age_residual else "Entry 3-EQUIVALENT (NO age-residual features, 48-wide)"}',
        f'  Calibration mode: {args.calibration}',
        f'  Model family: {args.model_family}',
    ]
    if args.model_family == 'logreg':
        lines.append(f'  Logreg C={args.C}  penalty={args.penalty or "l2 (default)"}')
    lines.append('')

    if len(fold_results) > 0:
        aurocs = [r['age_cond_auroc'] for r in fold_results
                  if not np.isnan(r['age_cond_auroc'])]
        mean_auroc = np.mean(aurocs) if aurocs else float('nan')

        lines += ['── Per-fold results ──────────────────────────────────────', '']
        for r in fold_results:
            lines.append(
                f"  Holdout {r['holdout_site']:<8s}  "
                f"Age-cond AUROC={r['age_cond_auroc']:.4f}  "
                f"Std AUROC={r['std_auroc']:.4f}  "
                f"Sens={r['sensitivity']:.3f}  Spec={r['specificity']:.3f}  "
                f"n_pos={r['n_pos_test']}"
            )
        lines += ['', f'  Mean age-conditioned AUROC: {mean_auroc:.4f}', '']

        if pooled_sweep_rows:
            lines += ['── Pooled threshold sweep (both folds combined) ──', '']
            best_reward_row = max(pooled_sweep_rows, key=lambda r: r['reward'])
            for row in pooled_sweep_rows:
                marker = '  <-- highest reward' if row is best_reward_row else ''
                lines.append(
                    f"  t={row['threshold']:.2f}  reward={row['reward']:.4f}  "
                    f"auroc={row['auroc']:.4f}{marker}"
                )
            lines.append('')

    summary = '\n'.join(lines)
    print('\n' + summary)
    (out_dir / 'loso_summary.txt').write_text(summary)
    print(f'\nOutputs written to: {out_dir}')


if __name__ == '__main__':
    main()