# features/physiological.py
# Physiological signal feature extraction — features 10-58
#
# Iteration history:
#   v1 (2026-06-30): Initial — 7 Hjorth parameters × 7 lead types (49 features)
#


import numpy as np
import os
from helper_code import (
    load_rename_rules, standardize_channel_names_rename_only,
    derive_bipolar_signal
)

# Resolve channel_table.csv relative to this file's location
_FEATURES_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR     = os.path.dirname(_FEATURES_DIR)
DEFAULT_CSV_PATH = os.path.join(_REPO_DIR, 'channel_table.csv')


def extract_physiological_features(physiological_data, physiological_fs,
                                    csv_path=DEFAULT_CSV_PATH):
    """
    Standardises channel names, derives bipolar signals, and computes
    7 Hjorth-style time-domain features for 7 lead types.

    Lead order: EEG, EOG, ChinEMG, LegEMG, ECG, Resp, SpO2
    Feature order per lead: std, MAV, ZCR, RMS, variance, mobility, complexity

    Returns np.ndarray of length 49 (7 features × 7 leads).
    NaN padding used for unavailable leads.
    """
    original_labels = list(physiological_data.keys())

    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(
        original_labels, rename_rules)

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

    # Bipolar derivations
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
            raise ValueError(
                f'Sampling rate mismatch for {target}: '
                f'{dict(zip(all_involved, fs_vals))}')
        ref = (processed_channels[neg_list[0]] if len(neg_list) == 1
               else tuple(processed_channels[n] for n in neg_list))
        derived = derive_bipolar_signal(processed_channels[pos], ref)
        if derived is not None:
            processed_channels[target] = derived
            processed_fs[target] = processed_fs[pos]

    # Lead selection — first available channel wins per lead type
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
            final_features.extend(
                [std_val, mav_val, zcr, rms, activity, mobility, complexity])
        else:
            final_features.extend([float('nan')] * 7)

    if 'processed_channels' in locals():
        del processed_channels

    return np.array(final_features)