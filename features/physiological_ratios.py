# features/physiological_ratios.py
# Stage-conditional ratio features — replaces absolute Hjorth (entry 2+)
#
# Iteration history:
#   v2 (2026-07-01): LOSO ablation showed absolute Hjorth features hurt
#                    cross-site AUROC by +0.09 (I0006) and +0.03 (S0001).
#                    Root cause: absolute signal amplitude is equipment-
#                    dependent. Within-recording ratios cancel this out.
#
# All 15 features are ratios within a single patient's recording.
# Equipment differences affect numerator and denominator equally → cancel.
#
# Index range: 33-47 in entry 2 feature vector.

import numpy as np
import os
from helper_code import (
    load_rename_rules, standardize_channel_names_rename_only,
    derive_bipolar_signal
)

_FEATURES_DIR    = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR        = os.path.dirname(_FEATURES_DIR)
DEFAULT_CSV_PATH = os.path.join(_REPO_DIR, 'channel_table.csv')

N_RATIO_FEATURES = 15   # EEG N3/Wake (7) + EEG REM/Wake (7) + Chin atonia (1)

# Minimum recording seconds per stage to trust a ratio
_MIN_STAGE_SECONDS = 120   # 2 minutes

# Cap on ratio values — prevents extreme outliers from dominating
_RATIO_CAP = 20.0


# ── Hjorth parameter vector ───────────────────────────────────────────────────
def _hjorth(sig):
    """
    Compute 7 Hjorth-style features for a 1-D signal array.
    Returns np.ndarray(7,) or NaN array if signal is too short.
    """
    if sig is None or len(sig) < 10:
        return np.full(7, float('nan'))
    activity = float(np.var(sig))
    if activity < 1e-20:
        return np.array([0., 0., 0., 0., 0., 0., 0.], dtype=float)
    diff1    = np.diff(sig)
    var_d1   = float(np.var(diff1))
    mobility = float(np.sqrt(var_d1 / activity)) if activity > 0 else 0.
    diff2    = np.diff(diff1)
    var_d2   = float(np.var(diff2))
    complexity = float(np.sqrt(var_d2 / var_d1) / mobility) \
        if (var_d1 > 0 and mobility > 0) else 0.
    return np.array([
        float(np.std(sig)),
        float(np.mean(np.abs(sig))),
        float(np.mean(np.diff(np.sign(sig)) != 0)),
        float(np.sqrt(np.mean(sig ** 2))),
        activity,
        mobility,
        complexity,
    ], dtype=float)


def _safe_ratio(num_vec, denom_vec):
    """
    Element-wise ratio with NaN guards and cap.
    Returns NaN wherever either input is NaN or denominator ≈ 0.
    """
    out = np.full(len(num_vec), float('nan'))
    for i, (n, d) in enumerate(zip(num_vec, denom_vec)):
        if np.isnan(n) or np.isnan(d) or abs(d) < 1e-12:
            continue
        out[i] = float(np.clip(n / d, -_RATIO_CAP, _RATIO_CAP))
    return out


def _get_channel(channels, candidates):
    """Return first available channel from candidates list."""
    for c in candidates:
        if c in channels and channels[c] is not None:
            return channels[c]
    return None


def _get_fs(fs_map, candidates, default=200.0):
    for c in candidates:
        if c in fs_map:
            return float(fs_map[c])
    return default


# ── Main extraction function ──────────────────────────────────────────────────
def extract_physiological_ratio_features(phys_data, phys_fs, algo_data,
                                          csv_path=DEFAULT_CSV_PATH):
    """
    Stage-conditional ratio features requiring both physiological EDF
    (raw signal) and CAISR annotations (stage timing).

    All features are within-recording ratios — site-stable by construction
    because equipment scale differences cancel in numerator and denominator.

    Returns np.ndarray of length 15:
        [0-6]  EEG Hjorth(N3)  / Hjorth(Wake)   — slow-wave quality ratio
               std, MAV, ZCR, RMS, var, mobility, complexity
               Interpretation: <1 means N3 signal weaker than wake (CI-risk)
                               >1 means N3 signal stronger (healthy)

        [7-13] EEG Hjorth(REM) / Hjorth(Wake)   — REM EEG quality ratio
               Same 7 parameters. REM theta/alpha signature vs wake.

        [14]   Chin MAV(REM) / Chin MAV(NREM)   — atonia quality ratio
               Healthy: near 0 (atonia in REM).
               CI-risk: elevated (REM Behaviour Disorder marker).
               Requires physiological EDF + stage_caisr.

    NaN fallback:
        - Missing physio EDF → all 15 NaN
        - Missing CAISR → all 15 NaN
        - Insufficient samples in a stage (<120s) → that ratio NaN
    """
    if not phys_data or not algo_data:
        return np.full(N_RATIO_FEATURES, float('nan'))

    # ── Stage annotations ─────────────────────────────────────────────────────
    stages_raw  = algo_data.get('stage_caisr', np.array([]))
    valid_stages = stages_raw[stages_raw < 9.0] if len(stages_raw) > 0 else np.array([])

    if len(valid_stages) < 10:
        return np.full(N_RATIO_FEATURES, float('nan'))

    # ── Standardise channel names ─────────────────────────────────────────────
    original_labels = list(phys_data.keys())
    rename_rules    = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(
        original_labels, rename_rules)

    channels = {}
    fs_map   = {}
    for old_label, data in phys_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        channels[new_label] = data
        if old_label in phys_fs:
            fs_map[new_label] = phys_fs[old_label]

    # ── Bipolar derivations for EEG and Chin ──────────────────────────────────
    for target, pos, neg_list in [
        ('f3-m2',       'f3',     ['m2']),
        ('f4-m1',       'f4',     ['m1']),
        ('c3-m2',       'c3',     ['m2']),
        ('c4-m1',       'c4',     ['m1']),
        ('chin1-chin2', 'chin 1', ['chin 2']),
    ]:
        if target in channels or pos not in channels:
            continue
        if not all(n in channels for n in neg_list):
            continue
        ref = (channels[neg_list[0]] if len(neg_list) == 1
               else tuple(channels[n] for n in neg_list))
        derived = derive_bipolar_signal(channels[pos], ref)
        if derived is not None:
            channels[target] = derived
            fs_map[target]   = fs_map.get(pos, 200.0)

    # ── EEG stage-conditional Hjorth ratios [0-13] ────────────────────────────
    result = np.full(N_RATIO_FEATURES, float('nan'))

    eeg_sig = _get_channel(channels, ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1'])
    eeg_fs  = _get_fs(fs_map, ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1'])

    if eeg_sig is not None and len(eeg_sig) > 0:
        spe     = int(round(30.0 * eeg_fs))    # samples per 30s epoch
        min_smp = int(_MIN_STAGE_SECONDS * eeg_fs)

        # Upsample stage to EEG sample rate
        stage_eeg = np.repeat(valid_stages, spe)[:len(eeg_sig)]

        wake_mask = (stage_eeg == 5)
        n3_mask   = (stage_eeg == 1)
        rem_mask  = (stage_eeg == 4)

        def stage_hjorth(mask):
            if int(mask.sum()) < min_smp:
                return np.full(7, float('nan'))
            seg = eeg_sig[:len(mask)][mask]
            return _hjorth(seg)

        h_wake = stage_hjorth(wake_mask)
        h_n3   = stage_hjorth(n3_mask)
        h_rem  = stage_hjorth(rem_mask)

        # [0-6] N3 / Wake ratios
        result[0:7]  = _safe_ratio(h_n3,  h_wake)
        # [7-13] REM / Wake ratios
        result[7:14] = _safe_ratio(h_rem, h_wake)

    # ── Chin atonia ratio [14] ────────────────────────────────────────────────
    chin_sig = _get_channel(channels, ['chin1-chin2', 'chin'])
    chin_fs  = _get_fs(fs_map, ['chin1-chin2', 'chin'])

    if chin_sig is not None and len(chin_sig) > 0:
        spe_c   = int(round(30.0 * chin_fs))
        min_c   = int(_MIN_STAGE_SECONDS * chin_fs)

        stage_chin   = np.repeat(valid_stages, spe_c)[:len(chin_sig)]
        rem_mask_c   = (stage_chin == 4)
        nrem_mask_c  = (stage_chin >= 1) & (stage_chin <= 3)

        if (int(rem_mask_c.sum()) >= min_c and
                int(nrem_mask_c.sum()) >= min_c):
            chin_rem  = float(np.mean(
                np.abs(chin_sig[:len(stage_chin)][rem_mask_c])))
            chin_nrem = float(np.mean(
                np.abs(chin_sig[:len(stage_chin)][nrem_mask_c])))
            if chin_nrem > 1e-12:
                result[14] = float(np.clip(chin_rem / chin_nrem,
                                           0., _RATIO_CAP))

    return result