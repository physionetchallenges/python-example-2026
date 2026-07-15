# features/spectral.py
# Raw-EDF spectral features — gated feature category (FEATURES.md,
# "Week 5+: Raw EDF features", gate condition: CAISR-only AUROC < 0.62
# after entry 3-4). Entry 5 landed at 0.6002, treated as satisfied-in-
# spirit given deadline pressure (see TEST_PLAN.md, 2026-07-11 K3 note).
#
# Unlike caisr_base/caisr_enriched (algorithmic annotation summaries) and
# physiological_ratios (Hjorth-derived time-domain features), this module
# computes actual frequency-domain content from the raw EEG signal. This
# is the first feature category that touches real signal content rather
# than CAISR-derived or time-domain-derived summary statistics.
#
# Priority order confirmed by the 2026-07-11 MCI/age cross-tab
# (learning_log.md): MCI is 44% of positives, significantly younger than
# other subtypes (68.4 vs 71.8 yr, p=0.00001), and has the lowest
# sensitivity (0.632 vs 0.766) — landing disproportionately in the
# highest-reward-value age band. This points toward early/subtle-stage
# slow-wave markers (this module) over autonomic/HRV markers (hrv.py,
# not yet built), since MCI is the earlier, subtler-signal stage of
# decline.
#
# Iteration history:
#   v1 (this file, first feature only): N3 delta power. Deliberately
#   scoped to ONE feature first, per the team's one-variable-per-test
#   discipline (learning_log.md decision log) and to get a real LOSO
#   read before committing to the full gated list (delta/theta ratio,
#   spindle density — both planned, not yet implemented, see TODOs below).
#
# NOT YET WIRED IN: this module is standalone. Adding its output to the
# actual feature vector requires updating features/__init__.py (new
# IDX_SPECTRAL_START/END constants, N_TOTAL_FEATURES bump) and every
# hstack call site (team_code.py, tools/loso_cv.py,
# tools/build_features_cache.py) consistently. That's a deliberate
# follow-up step, not done here — do not wire this in without updating
# all three call sites together, per the same discipline that made
# build_features_cache.py's verification necessary in the first place.

import numpy as np
import os
from scipy.signal import welch
from helper_code import (
    load_rename_rules, standardize_channel_names_rename_only,
    derive_bipolar_signal
)

_FEATURES_DIR    = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR        = os.path.dirname(_FEATURES_DIR)
DEFAULT_CSV_PATH = os.path.join(_REPO_DIR, 'channel_table.csv')

N_SPECTRAL_FEATURES = 1   # N3 delta power only, for now. Planned: +2
                          # (delta/theta ratio, N2 spindle density) once
                          # this feature has a real LOSO read.

# Same threshold as physiological_ratios.py — insufficient stage duration
# means the estimate isn't trustworthy, return NaN rather than a noisy
# number.
_MIN_STAGE_SECONDS = 120   # 2 minutes

# Standard delta band definition (AASM convention).
_DELTA_BAND = (0.5, 4.0)   # Hz

# Welch PSD window length. 4s windows give ~0.25Hz frequency resolution,
# fine enough to resolve the delta band's upper edge (4Hz) cleanly, short
# enough that a single N3 segment (often only a few minutes per patient)
# still yields several averaged windows rather than one noisy periodogram.
_WELCH_WINDOW_SECONDS = 4.0


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


def _bandpower(sig, fs, band, window_seconds=_WELCH_WINDOW_SECONDS):
    """
    Welch PSD bandpower for a single 1-D signal segment.

    Returns NaN if the segment is too short for even one Welch window,
    or if the signal is degenerate (near-zero variance — e.g. a flat/
    disconnected channel).
    """
    if sig is None or len(sig) < 10:
        return float('nan')

    nperseg = int(round(window_seconds * fs))
    if nperseg < 8 or len(sig) < nperseg:
        return float('nan')

    if float(np.var(sig)) < 1e-20:
        return 0.0

    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)

    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_mask):
        return float('nan')

    # np.trapz was removed in NumPy 2.0+ (renamed np.trapezoid). Kaggle's
    # image and local dev environments may not be on the same NumPy
    # version, so don't hardcode either name — this bit a local sanity
    # test during development (NumPy 2.4.4 has no np.trapz at all).
    _trapz_fn = getattr(np, 'trapezoid', None) or np.trapz
    return float(_trapz_fn(psd[band_mask], freqs[band_mask]))


def extract_spectral_features(phys_data, phys_fs, algo_data,
                               csv_path=DEFAULT_CSV_PATH):
    """
    Raw-EDF spectral features requiring both physiological EDF (raw
    signal) and CAISR annotations (stage timing) — same dual-source
    requirement as physiological_ratios.py.

    Returns np.ndarray of length N_SPECTRAL_FEATURES (currently 1):
        [0]  EEG delta-band (0.5-4Hz) power during N3 sleep only.
             Absolute power, NOT a ratio — unlike physiological_ratios.py's
             features, this is not expected to be site-stable by
             construction. Equipment scale could plausibly reintroduce
             the same confound that made absolute Hjorth features fail
             cross-site LOSO (2026-07-01 finding). This needs its OWN
             LOSO ablation before being trusted as a keeper, not an
             assumption that the delta-power literature backing
             transfers automatically to this equipment-confound-prone
             measurement style. Flagging explicitly so this isn't
             silently assumed safe.

    NaN fallback:
        - Missing physio EDF → NaN
        - Missing CAISR → NaN
        - Insufficient N3 duration (<120s) → NaN
        - Signal too short for even one Welch window → NaN
    """
    if not phys_data or not algo_data:
        return np.full(N_SPECTRAL_FEATURES, float('nan'))

    stages_raw   = algo_data.get('stage_caisr', np.array([]))
    valid_stages = stages_raw[stages_raw < 9.0] if len(stages_raw) > 0 else np.array([])

    if len(valid_stages) < 10:
        return np.full(N_SPECTRAL_FEATURES, float('nan'))

    # Same channel-standardization pattern as physiological_ratios.py —
    # reused deliberately, not reimplemented, so both modules stay
    # consistent if channel_table.csv or rename rules ever change.
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

    # Same bipolar derivation candidates as physiological_ratios.py.
    for target, pos, neg_list in [
        ('f3-m2', 'f3', ['m2']),
        ('f4-m1', 'f4', ['m1']),
        ('c3-m2', 'c3', ['m2']),
        ('c4-m1', 'c4', ['m1']),
    ]:
        if target in channels or pos not in channels:
            continue
        if not all(n in channels for n in neg_list):
            continue
        ref = channels[neg_list[0]]
        derived = derive_bipolar_signal(channels[pos], ref)
        if derived is not None:
            channels[target] = derived
            fs_map[target]   = fs_map.get(pos, 200.0)

    result = np.full(N_SPECTRAL_FEATURES, float('nan'))

    eeg_sig = _get_channel(channels, ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1'])
    eeg_fs  = _get_fs(fs_map, ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1'])

    if eeg_sig is None or len(eeg_sig) == 0:
        return result

    spe     = int(round(30.0 * eeg_fs))   # samples per 30s epoch
    min_smp = int(_MIN_STAGE_SECONDS * eeg_fs)

    # Same upsample-stage-to-signal-rate pattern as physiological_ratios.py.
    stage_eeg = np.repeat(valid_stages, spe)[:len(eeg_sig)]
    n3_mask   = (stage_eeg == 1)   # N3 stage code — confirmed via
                                    # caisr_base.py's stage convention
                                    # (W=5, N1=3, N2=2, N3=1, REM=4)

    if int(n3_mask.sum()) < min_smp:
        return result

    n3_segment = eeg_sig[:len(stage_eeg)][n3_mask]
    result[0] = _bandpower(n3_segment, eeg_fs, _DELTA_BAND)

    return result


# Standard theta band definition (AASM convention: 4-8Hz). Some clinical
# references use 4-7Hz — 4-8Hz chosen here for a clean non-overlapping
# boundary with the delta band above (0.5-4Hz) and the alpha band below
# (typically 8-13Hz). Worth revisiting if delta/theta ratio results look
# sensitive to this exact boundary.
_THETA_BAND = (4.0, 8.0)   # Hz

N_DELTA_THETA_FEATURES = 1
N_SPINDLE_FEATURES = 1

# Sigma band — standard sleep spindle frequency range (AASM convention).
_SIGMA_BAND = (11.0, 16.0)   # Hz

# Spindle duration constraints (AASM: spindles are 0.5-3s bursts).
_MIN_SPINDLE_DURATION_S = 0.5
_MAX_SPINDLE_DURATION_S = 3.0

# Envelope smoothing window — short enough to preserve individual
# spindle onsets/offsets, long enough to avoid triggering on single-
# sample envelope noise.
_ENVELOPE_SMOOTH_S = 0.1

# Adaptive threshold: mean + K * std of the envelope, computed from the
# SAME N2 segment being scored. This is deliberately per-recording, not
# a fixed absolute value — if equipment scale multiplies the signal (and
# therefore the envelope) by some gain k, both the mean and std scale by
# k too, so "envelope > mean + K*std" is unaffected by k. Same site-
# stability reasoning as the delta/theta ratio, applied to event
# detection instead of a power ratio.
_THRESHOLD_STD_MULTIPLIER = 1.5

# Same cap convention as physiological_ratios.py — prevents a
# near-zero-theta patient from producing an extreme outlier ratio.
_RATIO_CAP = 20.0


def _safe_ratio(num, denom, cap=_RATIO_CAP):
    """Scalar-safe ratio with NaN/near-zero guards. Same discipline as
    physiological_ratios.py's _safe_ratio, just for a single value
    instead of a feature vector (this module only produces one ratio
    per patient so far)."""
    if num is None or denom is None or np.isnan(num) or np.isnan(denom):
        return float('nan')
    if abs(denom) < 1e-12:
        return float('nan')
    return float(np.clip(num / denom, 0.0, cap))


def extract_delta_theta_ratio_features(phys_data, phys_fs, algo_data,
                                        csv_path=DEFAULT_CSV_PATH):
    """
    N3 delta/theta bandpower ratio — kept as a SEPARATE function from
    extract_spectral_features (raw delta power) deliberately, so the two
    can be ablation-tested independently. Raw delta power's small-set
    ablation (learning_log.md, 2026-07-12) came back null-to-mildly-
    negative on the best-powered fold (S0001) — this ratio version is a
    distinct hypothesis, not an extension of that result, and should be
    judged on its own LOSO read.

    Rationale for expecting better cross-site behavior than raw delta
    power: if equipment introduces a multiplicative gain confound on the
    EEG channel, it inflates delta and theta power equally, so it
    cancels in the ratio — same mechanism that made the existing
    within-recording ratio features (physiological_ratios.py) survive
    cross-site LOSO where the original absolute Hjorth features didn't
    (2026-07-01 finding). This is a hypothesis carried over by analogy,
    not yet confirmed for THIS specific ratio — needs its own ablation
    read, same as every other feature in this project.

    Returns np.ndarray of length N_DELTA_THETA_FEATURES (1):
        [0]  N3 delta-band power / N3 theta-band power. Both bands
             computed from the SAME N3-masked segment via the SAME Welch
             PSD call (not two separate extractions) — cheaper and
             guarantees identical windowing between numerator and
             denominator.

    NaN fallback: identical conditions to extract_spectral_features
    (missing physio/CAISR, insufficient N3 duration, degenerate signal).
    """
    if not phys_data or not algo_data:
        return np.full(N_DELTA_THETA_FEATURES, float('nan'))

    stages_raw   = algo_data.get('stage_caisr', np.array([]))
    valid_stages = stages_raw[stages_raw < 9.0] if len(stages_raw) > 0 else np.array([])

    if len(valid_stages) < 10:
        return np.full(N_DELTA_THETA_FEATURES, float('nan'))

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

    for target, pos, neg_list in [
        ('f3-m2', 'f3', ['m2']),
        ('f4-m1', 'f4', ['m1']),
        ('c3-m2', 'c3', ['m2']),
        ('c4-m1', 'c4', ['m1']),
    ]:
        if target in channels or pos not in channels:
            continue
        if not all(n in channels for n in neg_list):
            continue
        ref = channels[neg_list[0]]
        derived = derive_bipolar_signal(channels[pos], ref)
        if derived is not None:
            channels[target] = derived
            fs_map[target]   = fs_map.get(pos, 200.0)

    result = np.full(N_DELTA_THETA_FEATURES, float('nan'))

    eeg_sig = _get_channel(channels, ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1'])
    eeg_fs  = _get_fs(fs_map, ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1'])

    if eeg_sig is None or len(eeg_sig) == 0:
        return result

    spe     = int(round(30.0 * eeg_fs))
    min_smp = int(_MIN_STAGE_SECONDS * eeg_fs)

    stage_eeg = np.repeat(valid_stages, spe)[:len(eeg_sig)]
    n3_mask   = (stage_eeg == 1)

    if int(n3_mask.sum()) < min_smp:
        return result

    n3_segment = eeg_sig[:len(stage_eeg)][n3_mask]

    delta_power = _bandpower(n3_segment, eeg_fs, _DELTA_BAND)
    theta_power = _bandpower(n3_segment, eeg_fs, _THETA_BAND)

    result[0] = _safe_ratio(delta_power, theta_power)

    return result


def _detect_spindles(sig, fs, min_duration_s=_MIN_SPINDLE_DURATION_S,
                      max_duration_s=_MAX_SPINDLE_DURATION_S,
                      threshold_k=_THRESHOLD_STD_MULTIPLIER):
    """
    Simplified amplitude-envelope spindle detector:
      1. Bandpass filter to sigma band (11-16Hz)
      2. Hilbert transform -> amplitude envelope
      3. Smooth envelope (short moving average)
      4. Threshold at mean + K*std of THIS segment's own envelope
         (adaptive, not absolute — see module-level comment on why)
      5. Count contiguous above-threshold runs whose duration falls
         within [min_duration_s, max_duration_s]

    This is a simplified single-channel amplitude detector, not a
    validated clinical-grade algorithm (e.g. it does not implement
    multi-channel consensus, doesn't model the characteristic waxing/
    waning spindle envelope shape beyond a simple threshold, and doesn't
    merge near-adjacent events). Documented explicitly as a limitation —
    do not present spindle counts from this function as clinically
    validated without further work. It IS a legitimate first-pass
    implementation of the standard "filter -> envelope -> threshold ->
    duration-gate" approach used by simpler published detectors.

    Returns the count of qualifying events (int), or NaN if the signal
    is too short to filter/analyze.
    """
    from scipy.signal import butter, filtfilt, hilbert

    if sig is None or len(sig) < int(fs * 2):  # need at least ~2s for a stable filter
        return float('nan')

    nyq = fs / 2.0
    low  = _SIGMA_BAND[0] / nyq
    high = min(_SIGMA_BAND[1] / nyq, 0.99)   # guard against fs too low
    if low <= 0 or high <= low:
        return float('nan')

    try:
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, sig)
    except Exception:
        return float('nan')

    envelope = np.abs(hilbert(filtered))

    smooth_win = max(1, int(round(_ENVELOPE_SMOOTH_S * fs)))
    if smooth_win > 1:
        kernel = np.ones(smooth_win) / smooth_win
        envelope = np.convolve(envelope, kernel, mode='same')

    env_mean = float(np.mean(envelope))
    env_std  = float(np.std(envelope))
    if env_std < 1e-20:
        return 0   # flat/degenerate signal — no spindles detectable, not an error

    threshold = env_mean + threshold_k * env_std
    above = envelope > threshold

    # Count contiguous above-threshold runs meeting the duration gate.
    min_samples = int(round(min_duration_s * fs))
    max_samples = int(round(max_duration_s * fs))

    count = 0
    run_length = 0
    for val in above:
        if val:
            run_length += 1
        else:
            if min_samples <= run_length <= max_samples:
                count += 1
            run_length = 0
    if min_samples <= run_length <= max_samples:  # trailing run at signal end
        count += 1

    return count


def extract_spindle_density_features(phys_data, phys_fs, algo_data,
                                      csv_path=DEFAULT_CSV_PATH):
    """
    N2 sleep spindle density — spindles per minute of N2 sleep.

    Unlike extract_spectral_features (bandpower) and
    extract_delta_theta_ratio_features (power ratio), this is an
    EVENT-RATE measure: it counts discrete oscillatory bursts via
    _detect_spindles() rather than integrating power in a band. See that
    function's docstring for the detection method and its limitations.

    Returns np.ndarray of length N_SPINDLE_FEATURES (1):
        [0]  Spindle count during N2 / N2 duration in minutes.

    NaN fallback: missing physio/CAISR, insufficient N2 duration
    (<120s, same _MIN_STAGE_SECONDS threshold as every other feature in
    this module and physiological_ratios.py), or signal too short to
    filter.
    """
    if not phys_data or not algo_data:
        return np.full(N_SPINDLE_FEATURES, float('nan'))

    stages_raw   = algo_data.get('stage_caisr', np.array([]))
    valid_stages = stages_raw[stages_raw < 9.0] if len(stages_raw) > 0 else np.array([])

    if len(valid_stages) < 10:
        return np.full(N_SPINDLE_FEATURES, float('nan'))

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

    for target, pos, neg_list in [
        ('f3-m2', 'f3', ['m2']),
        ('f4-m1', 'f4', ['m1']),
        ('c3-m2', 'c3', ['m2']),
        ('c4-m1', 'c4', ['m1']),
    ]:
        if target in channels or pos not in channels:
            continue
        if not all(n in channels for n in neg_list):
            continue
        ref = channels[neg_list[0]]
        derived = derive_bipolar_signal(channels[pos], ref)
        if derived is not None:
            channels[target] = derived
            fs_map[target]   = fs_map.get(pos, 200.0)

    result = np.full(N_SPINDLE_FEATURES, float('nan'))

    eeg_sig = _get_channel(channels, ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1'])
    eeg_fs  = _get_fs(fs_map, ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1'])

    if eeg_sig is None or len(eeg_sig) == 0:
        return result

    spe     = int(round(30.0 * eeg_fs))
    min_smp = int(_MIN_STAGE_SECONDS * eeg_fs)

    stage_eeg = np.repeat(valid_stages, spe)[:len(eeg_sig)]
    n2_mask   = (stage_eeg == 2)   # N2 stage code — confirmed via
                                    # caisr_base.py (W=5, N1=3, N2=2,
                                    # N3=1, REM=4)

    if int(n2_mask.sum()) < min_smp:
        return result

    n2_segment = eeg_sig[:len(stage_eeg)][n2_mask]
    n2_minutes = len(n2_segment) / eeg_fs / 60.0

    count = _detect_spindles(n2_segment, eeg_fs)
    if isinstance(count, float) and np.isnan(count):
        return result

    result[0] = float(count) / n2_minutes if n2_minutes > 0 else float('nan')

    return result