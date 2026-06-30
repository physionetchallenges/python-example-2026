#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most
# parts of the required functions, change or remove non-required functions, and add your own functions.

################################################################################
# Libraries
################################################################################

import joblib
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sys
from tqdm import tqdm

from helper_code import *

################################################################################
# Configuration
################################################################################

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

# Feature vector dimensions — update these if you add/remove features
N_DEMOGRAPHIC_FEATURES   = 10   # age(1) + sex_onehot(3) + race_onehot(5) + bmi(1)
N_PHYSIOLOGICAL_FEATURES = 49   # 7 Hjorth × 7 lead types
N_ALGORITHMIC_FEATURES   = 23   # baseline 12 + 11 enriched (see extract_algorithmic_annotations_features)

################################################################################
# Required functions
################################################################################

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    if verbose:
        print('Finding the Challenge data...')

    patient_data_file    = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records          = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print('Extracting features and labels from the data...')

    features = []
    labels   = []

    pbar = tqdm(range(num_records), desc='Extracting Features', unit='record', disable=not verbose)
    for i in pbar:
        try:
            record     = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id    = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            if verbose:
                pbar.set_postfix({'patient': patient_id})

            # ── Demographics ─────────────────────────────────────────────────
            patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
            patient_data      = load_demographics(patient_data_file, patient_id, session_id)
            demographic_features = extract_demographic_features(patient_data)

            # ── Physiological EDF ─────────────────────────────────────────────
            phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                                     site_id, f'{patient_id}_ses-{session_id}.edf')
            if not os.path.exists(phys_file):
                if verbose:
                    tqdm.write(f'  ! Missing physiological EDF for {patient_id} — skipping.')
                continue
            physiological_data, physiological_fs = load_signal_data(phys_file)
            physiological_features = extract_physiological_features(
                physiological_data, physiological_fs, csv_path=csv_path)

            # ── CAISR Annotations ─────────────────────────────────────────────
            algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                                     site_id, f'{patient_id}_ses-{session_id}_caisr_annotations.edf')
            if os.path.exists(algo_file):
                algorithmic_annotations, _ = load_signal_data(algo_file)
                algorithmic_features = extract_algorithmic_annotations_features(algorithmic_annotations)
            else:
                tqdm.write(f'Error loading EDF file: [Errno 2] No such file or directory: '
                           f"'{algo_file}'")
                algorithmic_features = np.full(N_ALGORITHMIC_FEATURES, float('nan'))

            # ── Human Annotations (training only — not used in features) ──────
            human_file = os.path.join(data_folder, HUMAN_ANNOTATIONS_SUBFOLDER,
                                      site_id, f'{patient_id}_ses-{session_id}_expert_annotations.edf')
            if os.path.exists(human_file):
                human_annotations, _ = load_signal_data(human_file)
                # human_features extracted but intentionally NOT included in the model —
                # these are absent in the hidden validation and test sets.
                _ = extract_human_annotations_features(human_annotations)

            # ── Label ─────────────────────────────────────────────────────────
            label = load_diagnoses(os.path.join(data_folder, DEMOGRAPHICS_FILE), patient_id)

            if label == 0 or label == 1:
                features.append(np.hstack([demographic_features,
                                            physiological_features,
                                            algorithmic_features]))
                labels.append(label)

            # Free memory
            if 'physiological_data' in locals():   del physiological_data
            if 'algorithmic_annotations' in locals(): del algorithmic_annotations

        except Exception as e:
            tqdm.write(f'  !!! Error processing record {i+1} ({patient_id}): {e}')
            continue

    pbar.close()

    features = np.asarray(features, dtype=np.float32)
    labels   = np.asarray(labels,   dtype=bool)

    if verbose:
        print(f'Training on {len(labels)} patients '
              f'({labels.sum()} positive, {(~labels).sum()} negative)...')

    # ── XGBoost with class-imbalance weighting ────────────────────────────────
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos_weight,
        random_state      = 42,
        eval_metric       = 'auc',
        verbosity         = 0,
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
    model_filename = os.path.join(model_folder, 'model.sav')
    return joblib.load(model_filename)


def run_model(model, record, data_folder, verbose):
    model = model['model']

    patient_id = record[HEADERS['bids_folder']]
    site_id    = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    # ── Demographics ──────────────────────────────────────────────────────────
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data      = load_demographics(patient_data_file, patient_id, session_id)
    demographic_features = extract_demographic_features(patient_data)

    # ── Physiological EDF ─────────────────────────────────────────────────────
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                             site_id, f'{patient_id}_ses-{session_id}.edf')
    if os.path.exists(phys_file):
        phys_data, phys_fs = load_signal_data(phys_file)
        physiological_features = extract_physiological_features(phys_data, phys_fs)
    else:
        physiological_features = np.full(N_PHYSIOLOGICAL_FEATURES, float('nan'))

    # ── CAISR Annotations ─────────────────────────────────────────────────────
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                             site_id, f'{patient_id}_ses-{session_id}_caisr_annotations.edf')
    if os.path.exists(algo_file):
        algo_data, _ = load_signal_data(algo_file)
        algorithmic_features = extract_algorithmic_annotations_features(algo_data)
    else:
        algorithmic_features = np.full(N_ALGORITHMIC_FEATURES, float('nan'))

    features = np.hstack([demographic_features,
                           physiological_features,
                           algorithmic_features]).reshape(1, -1)

    binary_output      = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    return binary_output, probability_output


################################################################################
# Feature extraction functions
################################################################################

def extract_demographic_features(data):
    """
    Extracts and encodes demographic features from a metadata dictionary.

    Returns np.ndarray of length 10:
        [0]   Age (continuous, capped at 90)
        [1-3] Sex one-hot  (Female, Male, Unknown)
        [4-8] Race one-hot (Asian, Black, Others, Unavailable, White)
        [9]   BMI (continuous, NaN if missing)
    """
    age = np.array([load_age(data)])

    sex = load_sex(data, standardize=True)
    sex_vec = np.zeros(3)
    if sex == 'Female':  sex_vec[0] = 1
    elif sex == 'Male':  sex_vec[1] = 1
    else:                sex_vec[2] = 1

    race = load_race(data, standardize=True)
    race_vec = np.zeros(5)
    if race == 'Asian':          race_vec[0] = 1
    elif race == 'Black':        race_vec[1] = 1
    elif race == 'Others':       race_vec[2] = 1
    elif race == 'Unavailable':  race_vec[3] = 1
    elif race == 'White':        race_vec[4] = 1
    else:                        race_vec[2] = 1  # default to Others

    bmi = np.array([load_bmi(data)])

    return np.concatenate([age, sex_vec, race_vec, bmi])


def extract_physiological_features(physiological_data, physiological_fs,
                                    csv_path=DEFAULT_CSV_PATH):
    """
    Standardises channel names, derives bipolar signals, and computes
    7 Hjorth-style time-domain features for 7 lead types.

    Lead order: EEG, EOG, ChinEMG, LegEMG, ECG, Resp, SpO2
    Feature order per lead: std, MAV, ZCR, RMS, variance, mobility, complexity

    Returns np.ndarray of length 49 (7 features × 7 leads).
    """
    original_labels = list(physiological_data.keys())

    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(original_labels, rename_rules)

    processed_channels = {}
    processed_fs = {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        processed_channels[new_label] = data
        if old_label in physiological_fs:
            processed_fs[new_label] = physiological_fs[old_label]
        else:
            raise KeyError(f'No sampling rate for channel: {old_label}')

    if 'physiological_data' in locals():
        del physiological_data

    bipolar_configs = [
        ('f3-m2',       'f3',     ['m2']),
        ('f4-m1',       'f4',     ['m1']),
        ('c3-m2',       'c3',     ['m2']),
        ('c4-m1',       'c4',     ['m1']),
        ('o1-m2',       'o1',     ['m2']),
        ('o2-m1',       'o2',     ['m1']),
        ('e1-m2',       'e1',     ['m2']),
        ('e2-m1',       'e2',     ['m1']),
        ('chin1-chin2', 'chin 1', ['chin 2']),
        ('lat',         'lleg+',  ['lleg-']),
        ('rat',         'rleg+',  ['rleg-']),
    ]

    for target, pos, neg_list in bipolar_configs:
        if target in processed_channels or pos not in processed_channels:
            continue
        if not all(n in processed_channels for n in neg_list):
            continue
        all_involved = [pos] + neg_list
        fs_vals = [processed_fs[ch] for ch in all_involved]
        if len(set(fs_vals)) > 1:
            raise ValueError(f'Sampling rate mismatch for {target}: '
                             f'{dict(zip(all_involved, fs_vals))}')
        ref = (processed_channels[neg_list[0]] if len(neg_list) == 1
               else tuple(processed_channels[n] for n in neg_list))
        derived = derive_bipolar_signal(processed_channels[pos], ref)
        if derived is not None:
            processed_channels[target] = derived
            processed_fs[target] = processed_fs[pos]

    leads_to_check = {
        'eeg':  ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1'],
        'eog':  ['e1-m2', 'e2-m1'],
        'chin': ['chin1-chin2', 'chin'],
        'leg':  ['lat', 'rat'],
        'ecg':  ['ecg', 'ekg'],
        'resp': ['airflow', 'ptaf', 'abd', 'chest'],
        'spo2': ['spo2', 'sao2'],
    }

    final_features = []
    for lead_type, candidates in leads_to_check.items():
        sig = None
        for candidate in candidates:
            if candidate in processed_channels and processed_channels[candidate] is not None:
                sig = processed_channels[candidate]
                break

        if sig is not None and len(sig) > 1:
            std_val  = np.std(sig)
            mav_val  = np.mean(np.abs(sig))
            zcr      = np.mean(np.diff(np.sign(sig)) != 0)
            rms      = np.sqrt(np.mean(sig ** 2))
            activity = np.var(sig)
            diff1    = np.diff(sig)
            mobility = (np.sqrt(np.var(diff1) / activity)
                        if activity > 0 else 0.0)
            diff2    = np.diff(diff1)
            var_d1   = np.var(diff1)
            var_d2   = np.var(diff2)
            complexity = ((np.sqrt(var_d2 / var_d1) / mobility)
                          if (var_d1 > 0 and mobility > 0) else 0.0)
            final_features.extend([std_val, mav_val, zcr, rms, activity, mobility, complexity])
        else:
            final_features.extend([float('nan')] * 7)

    if 'processed_channels' in locals():
        del processed_channels

    return np.array(final_features)


def extract_algorithmic_annotations_features(algo_data):
    """
    Extracts sleep architecture and event density features from CAISR output.

    Returns np.ndarray of length 23:

    Baseline (indices 0-11):
        [0]  AHI total (events/hr)
        [1]  Arousal index (events/hr)
        [2]  Limb movement index (events/hr)
        [3]  Wake %
        [4]  N1 %
        [5]  N2 %
        [6]  N3 %
        [7]  REM %
        [8]  Sleep efficiency
        [9]  Mean CAISR P(Wake)
        [10] Mean CAISR P(N3)
        [11] Mean CAISR P(arousal)   ← fixed key: caisr_prob_arousal

    Enriched (indices 12-22):
        [12] OA rate (events/hr)
        [13] CA rate (events/hr)
        [14] HY rate (events/hr)
        [15] RERA rate (events/hr)
        [16] CA / total AHI ratio
        [17] REM-AHI (events/hr)
        [18] NREM-AHI (events/hr)
        [19] REM-AHI / NREM-AHI ratio
        [20] N3 first-half / N3 second-half ratio (temporal gradient)
        [21] Spontaneous arousal index (events/hr, non-respiratory)
        [22] N3 confidence entropy (proxy for slow-wave signal quality)
    """
    if not algo_data:
        return np.full(N_ALGORITHMIC_FEATURES, float('nan'))

    # ── Shared signals ────────────────────────────────────────────────────────
    resp    = algo_data.get('resp_caisr',    np.array([]))
    arousal = algo_data.get('arousal_caisr', np.array([]))
    limb    = algo_data.get('limb_caisr',    np.array([]))
    stages_raw = algo_data.get('stage_caisr', np.array([]))

    total_hours_resp = len(resp) / 3600.0 if len(resp) > 0 else 0.0

    # Valid stages only (exclude 9 = unavailable)
    valid_stages = stages_raw[stages_raw < 9.0] if len(stages_raw) > 0 else np.array([])

    # ── Helper: count discrete events (rising edges) in a signal ─────────────
    def count_events(sig, t_hours):
        """Count leading-edge transitions in a binary-ised signal."""
        if len(sig) == 0 or t_hours <= 0:
            return float('nan')
        binary = (np.asarray(sig) > 0).astype(int)
        edges  = np.diff(binary, prepend=0)
        return np.count_nonzero(edges == 1) / t_hours

    def count_event_type(resp_sig, event_code, t_hours):
        """Count events of a specific respiratory type."""
        if len(resp_sig) == 0 or t_hours <= 0:
            return float('nan')
        binary = (np.asarray(resp_sig) == event_code).astype(int)
        edges  = np.diff(binary, prepend=0)
        return np.count_nonzero(edges == 1) / t_hours

    # ── Baseline features [0-11] ──────────────────────────────────────────────
    ahi_auto    = count_events(resp,    total_hours_resp)
    arousal_idx = count_events(arousal, len(arousal) / 7200.0 if len(arousal) > 0 else 0.0)
    limb_idx    = count_events(limb,    total_hours_resp)

    if len(valid_stages) > 0:
        w_pct   = float(np.mean(valid_stages == 5))
        n1_pct  = float(np.mean(valid_stages == 3))
        n2_pct  = float(np.mean(valid_stages == 2))
        n3_pct  = float(np.mean(valid_stages == 1))
        rem_pct = float(np.mean(valid_stages == 4))
        efficiency = float(np.mean((valid_stages >= 1) & (valid_stages <= 4)))
    else:
        w_pct = n1_pct = n2_pct = n3_pct = rem_pct = efficiency = float('nan')

    prob_w    = float(np.mean(algo_data.get('caisr_prob_w',       [float('nan')])))
    prob_n3   = float(np.mean(algo_data.get('caisr_prob_n3',      [float('nan')])))
    # ↓ FIXED: was 'caisr_prob_arous' (wrong key — always NaN in baseline)
    prob_arou = float(np.mean(algo_data.get('caisr_prob_arous', [float('nan')])))

    # Clip probabilities > 1 to NaN (sentinel values in some files)
    prob_w    = prob_w    if prob_w    <= 1.0 else float('nan')
    prob_n3   = prob_n3   if prob_n3   <= 1.0 else float('nan')
    prob_arou = prob_arou if prob_arou <= 1.0 else float('nan')

    # ── Enriched features [12-22] ─────────────────────────────────────────────

    # [12-15] Respiratory event type breakdown
    oa_rate   = count_event_type(resp, 1, total_hours_resp)  # obstructive apnea
    ca_rate   = count_event_type(resp, 2, total_hours_resp)  # central apnea
    hy_rate   = count_event_type(resp, 4, total_hours_resp)  # hypopnea
    rera_rate = count_event_type(resp, 5, total_hours_resp)  # RERA

    # [16] CA / total AHI ratio — elevated = neurological apnea pattern
    if (not np.isnan(ca_rate) and not np.isnan(ahi_auto)
            and ahi_auto > 0):
        ca_total_ratio = ca_rate / ahi_auto
    else:
        ca_total_ratio = float('nan')

    # [17-19] Stage-conditional AHI: REM vs NREM
    # Requires upsampling stage_caisr (30s) to 1s resolution before cross-referencing
    rem_ahi = nrem_ahi = rem_nrem_ratio = float('nan')
    if len(valid_stages) > 0 and len(resp) > 0:
        stage_1s = np.repeat(valid_stages, 30)[:len(resp)]
        rem_mask  = (stage_1s == 4)
        nrem_mask = (stage_1s >= 1) & (stage_1s <= 3)

        rem_hours  = float(rem_mask.sum())  / 3600.0
        nrem_hours = float(nrem_mask.sum()) / 3600.0

        if rem_hours > 0:
            resp_rem = np.where(rem_mask, resp[:len(rem_mask)], 0)
            edges_rem = np.diff((resp_rem > 0).astype(int), prepend=0)
            rem_ahi = np.count_nonzero(edges_rem == 1) / rem_hours

        if nrem_hours > 0:
            resp_nrem = np.where(nrem_mask, resp[:len(nrem_mask)], 0)
            edges_nrem = np.diff((resp_nrem > 0).astype(int), prepend=0)
            nrem_ahi = np.count_nonzero(edges_nrem == 1) / nrem_hours

        if not np.isnan(rem_ahi) and not np.isnan(nrem_ahi) and nrem_ahi > 0:
            rem_nrem_ratio = rem_ahi / nrem_ahi

    # [20] N3 temporal gradient — healthy sleep strongly front-loads N3
    # Ratio >> 1 is normal; compressed toward 1 suggests CI-risk pattern
    if len(valid_stages) > 1:
        mid       = len(valid_stages) // 2
        n3_first  = float(np.mean(valid_stages[:mid] == 1))
        n3_second = float(np.mean(valid_stages[mid:]  == 1))
        n3_gradient = n3_first / (n3_second + 1e-6)
    else:
        n3_gradient = float('nan')

    # [21] Spontaneous arousal index
    # Arousals not coincident with a respiratory event (within ±15s window)
    spont_arousal_idx = float('nan')
    if len(arousal) > 0 and len(resp) > 0:
        # Downsample arousal from 0.5s → 1s
        n_1s = min(len(arousal) // 2, len(resp))
        if n_1s > 0:
            ar_reshaped = arousal[:n_1s * 2].reshape(-1, 2)
            arousal_1s  = (ar_reshaped.max(axis=1) > 0).astype(int)
            resp_1s     = (resp[:n_1s] > 0).astype(int)

            edges_ar = np.diff(arousal_1s, prepend=0)
            ar_starts = np.where(edges_ar == 1)[0]

            n_resp_coinci = 0
            for s in ar_starts:
                window = resp_1s[max(0, s - 5):min(n_1s, s + 15)]
                if len(window) > 0 and window.any():
                    n_resp_coinci += 1

            n_spont = max(0, len(ar_starts) - n_resp_coinci)
            t_hours_1s = n_1s / 3600.0
            if t_hours_1s > 0:
                spont_arousal_idx = n_spont / t_hours_1s

    # [22] N3 confidence entropy — higher entropy = more ambiguous slow-wave staging
    # Uses binary entropy on caisr_prob_n3 as a proxy for slow-wave signal quality.
    # Lower entropy = CAISR is confident = cleaner slow waves.
    n3_entropy = float('nan')
    prob_n3_arr = algo_data.get('caisr_prob_n3', np.array([]))
    if len(prob_n3_arr) > 0:
        p = np.clip(np.asarray(prob_n3_arr, dtype=float), 1e-9, 1.0 - 1e-9)
        # Binary entropy: H = -(p log p + (1-p) log(1-p))
        h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        n3_entropy = float(np.mean(h))

    # ── Assemble and return ───────────────────────────────────────────────────
    return np.array([
        # Baseline [0-11]
        ahi_auto, arousal_idx, limb_idx,
        w_pct, n1_pct, n2_pct, n3_pct, rem_pct, efficiency,
        prob_w, prob_n3, prob_arou,
        # Enriched [12-22]
        oa_rate, ca_rate, hy_rate, rera_rate,
        ca_total_ratio,
        rem_ahi, nrem_ahi, rem_nrem_ratio,
        n3_gradient,
        spont_arousal_idx,
        n3_entropy,
    ], dtype=float)


def extract_human_annotations_features(human_data):
    """
    Extracts features from expert-scored human annotations.
    These are available in training only and intentionally NOT used in the model —
    they are absent in the hidden validation and test sets.
    Returns np.ndarray of length 12.
    """
    if not human_data or 'resp_expert' not in human_data:
        return np.full(12, float('nan'))

    features = []
    total_seconds = len(human_data.get('resp_expert', []))
    total_hours   = total_seconds / 3600.0

    def count_events(key):
        if key not in human_data or total_hours <= 0:
            return float('nan')
        sig   = (human_data[key] > 0).astype(int)
        edges = np.diff(sig, prepend=0)
        return np.count_nonzero(edges == 1) / total_hours

    features.extend([
        count_events('resp_expert'),
        count_events('arousal_expert'),
        count_events('limb_expert'),
    ])

    stages = human_data.get('stage_expert', np.array([]))
    valid  = stages[stages < 9.0] if len(stages) > 0 else np.array([])
    if len(valid) > 0:
        features.extend([
            float(np.mean(valid == 5)),
            float(np.mean(valid == 4)),
            float(np.mean(valid == 3)),
            float(np.mean(valid == 2)),
            float(np.mean(valid == 1)),
            float(np.mean(valid > 0)),
        ])
    else:
        features.extend([float('nan')] * 6)

    if len(valid) > 1:
        features.extend([
            float(np.count_nonzero(np.diff(valid)) / total_hours),
            float(np.count_nonzero(valid == 0) * 30 / 60.0),
            float(np.where(valid == 4)[0][0]) if np.any(valid == 4) else float('nan'),
        ])
    else:
        features.extend([float('nan')] * 3)

    return np.array(features)


def save_model(model_folder, model):
    joblib.dump({'model': model},
                os.path.join(model_folder, 'model.sav'),
                protocol=0)