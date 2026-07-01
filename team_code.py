#!/usr/bin/env python

# Team Narnia — PhysioNet Challenge 2026
# Entry 2: Dropped site-confounded absolute Hjorth features.
#           Replaced with within-recording ratio features (site-stable).
#           Tighter XGBoost regularization.
#           Fixed N3 gradient + redesigned spontaneous arousal.
#
# See features/FEATURES.md and LEARNING_LOG.md for full rationale.

################################################################################
# Libraries
################################################################################

import joblib
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tqdm import tqdm

from helper_code import *

# Feature modules
from features.demographic        import extract_demographic_features
from features.caisr_base         import extract_caisr_base_features
from features.caisr_enriched     import extract_caisr_enriched_features
from features.physiological_ratios import (extract_physiological_ratio_features,
                                            N_RATIO_FEATURES)
from features.human              import extract_human_annotations_features
from features import (N_CAISR_BASE_FEATURES, N_CAISR_ENRICHED_FEATURES,
                      N_DEMOGRAPHIC_FEATURES)

################################################################################
# Configuration
################################################################################

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

# Fallback sizes for missing files
_N_BASE     = N_CAISR_BASE_FEATURES      # 12
_N_ENRICHED = N_CAISR_ENRICHED_FEATURES  # 11
_N_RATIO    = N_RATIO_FEATURES           # 15

################################################################################
# Required functions
################################################################################

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    if verbose:
        print('Finding the Challenge data...')

    patient_data_file     = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records           = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print('Extracting features and labels from the data...')

    features = []
    labels   = []

    pbar = tqdm(range(num_records), desc='Extracting Features',
                unit='record', disable=not verbose)
    for i in pbar:
        try:
            record     = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id    = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            if verbose:
                pbar.set_postfix({'patient': patient_id})

            # ── Demographics ─────────────────────────────────────────────────
            demo_file    = os.path.join(data_folder, DEMOGRAPHICS_FILE)
            patient_data = load_demographics(demo_file, patient_id, session_id)
            demo_f = extract_demographic_features(patient_data)

            # ── Physiological EDF ─────────────────────────────────────────────
            phys_file = os.path.join(
                data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                site_id, f'{patient_id}_ses-{session_id}.edf')
            if not os.path.exists(phys_file):
                if verbose:
                    tqdm.write(
                        f'  ! Missing physiological EDF for {patient_id} — skipping.')
                continue
            phys_data, phys_fs = load_signal_data(phys_file)

            # ── CAISR Annotations ─────────────────────────────────────────────
            algo_file = os.path.join(
                data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                site_id, f'{patient_id}_ses-{session_id}_caisr_annotations.edf')
            if os.path.exists(algo_file):
                algo_data, _ = load_signal_data(algo_file)
                caisr_base     = extract_caisr_base_features(algo_data)
                caisr_enriched = extract_caisr_enriched_features(algo_data)
                # Ratio features require BOTH phys + CAISR
                ratio_f = extract_physiological_ratio_features(
                    phys_data, phys_fs, algo_data, csv_path=csv_path)
            else:
                tqdm.write(
                    f'Error loading EDF file: [Errno 2] No such file or directory: '
                    f"'{algo_file}'")
                caisr_base     = np.full(_N_BASE,     float('nan'))
                caisr_enriched = np.full(_N_ENRICHED, float('nan'))
                ratio_f        = np.full(_N_RATIO,    float('nan'))

            # ── Human Annotations (training only — not used in model) ─────────
            human_file = os.path.join(
                data_folder, HUMAN_ANNOTATIONS_SUBFOLDER,
                site_id, f'{patient_id}_ses-{session_id}_expert_annotations.edf')
            if os.path.exists(human_file):
                human_data, _ = load_signal_data(human_file)
                _ = extract_human_annotations_features(human_data)

            # ── Label ─────────────────────────────────────────────────────────
            label = load_diagnoses(
                os.path.join(data_folder, DEMOGRAPHICS_FILE), patient_id)

            if label == 0 or label == 1:
                features.append(np.hstack([
                    demo_f,
                    caisr_base,
                    caisr_enriched,
                    ratio_f,
                ]))
                labels.append(label)

            if 'phys_data'  in locals(): del phys_data
            if 'algo_data'  in locals(): del algo_data
            if 'human_data' in locals(): del human_data

        except Exception as e:
            tqdm.write(f'  !!! Error on record {i+1} ({patient_id}): {e}')
            continue

    pbar.close()

    features = np.asarray(features, dtype=np.float32)
    labels   = np.asarray(labels,   dtype=bool)

    if verbose:
        n_pos = int(labels.sum())
        n_neg = int((~labels).sum())
        print(f'Training on {len(labels)} patients '
              f'({n_pos} positive, {n_neg} negative)...')
        print(f'Feature vector shape: {features.shape}')

    # ── XGBoost — tighter regularization than entry 1 ────────────────────────
    # Entry 1 used depth=4, subsample=0.8, colsample=0.8, no explicit L1/L2.
    # LOSO showed overfitting to within-site patterns → increase regularization.
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 3,       # reduced: 4 → 3
        learning_rate    = 0.05,
        subsample        = 0.7,     # reduced: 0.8 → 0.7
        colsample_bytree = 0.6,     # reduced: 0.8 → 0.6
        reg_alpha        = 0.1,     # new: L1 regularization
        reg_lambda       = 2.0,     # new: L2 regularization (default was 1.0)
        min_child_weight = 5,       # new: minimum samples per leaf
        scale_pos_weight = scale_pos_weight,
        random_state     = 42,
        eval_metric      = 'auc',
        verbosity        = 0,
    )

    model = Pipeline([
        ('imputer',    SimpleImputer(strategy='median')),
        ('classifier', xgb),
    ])
    model.fit(features, labels)

    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()


def load_model(model_folder, verbose):
    return joblib.load(os.path.join(model_folder, 'model.sav'))


def run_model(model, record, data_folder, verbose):
    model = model['model']

    patient_id = record[HEADERS['bids_folder']]
    site_id    = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    # ── Demographics ──────────────────────────────────────────────────────────
    demo_file    = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(demo_file, patient_id, session_id)
    demo_f = extract_demographic_features(patient_data)

    # ── Physiological EDF ─────────────────────────────────────────────────────
    phys_file = os.path.join(
        data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
        site_id, f'{patient_id}_ses-{session_id}.edf')
    phys_data = phys_fs = None
    if os.path.exists(phys_file):
        phys_data, phys_fs = load_signal_data(phys_file)

    # ── CAISR Annotations ─────────────────────────────────────────────────────
    algo_file = os.path.join(
        data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
        site_id, f'{patient_id}_ses-{session_id}_caisr_annotations.edf')
    if os.path.exists(algo_file):
        algo_data, _ = load_signal_data(algo_file)
        caisr_base     = extract_caisr_base_features(algo_data)
        caisr_enriched = extract_caisr_enriched_features(algo_data)
        if phys_data is not None:
            ratio_f = extract_physiological_ratio_features(
                phys_data, phys_fs, algo_data)
        else:
            ratio_f = np.full(_N_RATIO, float('nan'))
    else:
        caisr_base     = np.full(_N_BASE,     float('nan'))
        caisr_enriched = np.full(_N_ENRICHED, float('nan'))
        ratio_f        = np.full(_N_RATIO,    float('nan'))

    features = np.hstack([
        demo_f,
        caisr_base,
        caisr_enriched,
        ratio_f,
    ]).reshape(1, -1)

    binary_output      = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    return binary_output, probability_output


################################################################################
# Utilities
################################################################################

def save_model(model_folder, model):
    joblib.dump({'model': model},
                os.path.join(model_folder, 'model.sav'),
                protocol=0)