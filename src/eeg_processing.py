import pandas as pd
import numpy as np
from src.common.channel_utils import find_matching_label, get_cached_channel_table, normalize_channel_label, split_channel_aliases
from src.common.signal_utils import resample_signal
from .lib import eeg_features

EEG_CHANNEL_SPECS = {
    'C3-M2': {'direct': 'c3-m2', 'positive': 'c3', 'reference': 'm2'},
    'C4-M1': {'direct': 'c4-m1', 'positive': 'c4', 'reference': 'm1'},
    'F3-M2': {'direct': 'f3-m2', 'positive': 'f3', 'reference': 'm2'},
    'F4-M1': {'direct': 'f4-m1', 'positive': 'f4', 'reference': 'm1'},
}
EEG_FEATURE_SPECS = [
    ('C3-M2', 'Hjorth_Complexity'),
    ('C4-M1', 'Hjorth_Complexity'),
    ('F3-M2', 'Hjorth_Complexity'),
    ('F4-M1', 'Hjorth_Complexity'),
    ('C3-M2', 'Hjorth_Mobility'),
    ('F3-M2', 'Hjorth_Mobility'),
    ('F4-M1', 'Hjorth_Mobility'),
    ('C3-M2', 'Ratio_Slow_Fast'),
    ('C4-M1', 'Ratio_Slow_Fast'),
    ('F3-M2', 'Ratio_Slow_Fast'),
    ('F4-M1', 'Ratio_Slow_Fast'),
    ('C3-M2', 'Rel_Beta'),
    ('F3-M2', 'Rel_Beta'),
    ('F4-M1', 'Rel_Beta'),
    ('C4-M1', 'Rel_Sigma'),
    ('F3-M2', 'Rel_Sigma'),
    ('C3-M2', 'Relative_Delta_Power'),
    ('C4-M1', 'Relative_Delta_Power'),
    ('F3-M2', 'Relative_Delta_Power'),
    ('F4-M1', 'Relative_Delta_Power'),
    ('C3-M2', 'Theta_Alpha_Ratio'),
    ('C4-M1', 'Theta_Alpha_Ratio'),
    ('F3-M2', 'Theta_Alpha_Ratio'),
    ('F4-M1', 'Theta_Alpha_Ratio'),
    ('C3-M2', 'Theta_Beta_Ratio'),
    ('C4-M1', 'Theta_Beta_Ratio'),
    ('F3-M2', 'Theta_Beta_Ratio'),
    ('F4-M1', 'Theta_Beta_Ratio'),
    ('C3-M2', 'kurtosis_Alpha'),
    ('C3-M2', 'kurtosis_Beta'),
    ('C4-M1', 'kurtosis_Beta'),
    ('F3-M2', 'kurtosis_Beta'),
    ('F4-M1', 'kurtosis_Beta'),
    ('C3-M2', 'kurtosis_Delta'),
    ('C4-M1', 'kurtosis_Delta'),
    ('F3-M2', 'kurtosis_Delta'),
    ('F4-M1', 'kurtosis_Delta'),
    ('C3-M2', 'kurtosis_Sigma'),
    ('C4-M1', 'kurtosis_Sigma'),
    ('F3-M2', 'kurtosis_Sigma'),
    ('F4-M1', 'kurtosis_Sigma'),
    ('C3-M2', 'kurtosis_Theta'),
    ('C4-M1', 'kurtosis_Theta'),
    ('F3-M2', 'kurtosis_Theta'),
    ('F4-M1', 'kurtosis_Theta'),
    ('C3-M2', 'variability_Delta'),
    ('C4-M1', 'variability_Delta'),
    ('F3-M2', 'variability_Delta'),
    ('F4-M1', 'variability_Delta'),
]
EEG_FEATURE_NAMES = [f'{channel}_{metric}' for channel, metric in EEG_FEATURE_SPECS]
EEG_FEATURE_LENGTH = len(EEG_FEATURE_NAMES)
EEG_ALIASES_CACHE = {}


def _build_eeg_aliases(channels):
    alias_lookup = {}
    for _, row in channels.iterrows():
        aliases = split_channel_aliases(row['Channel_Names'])
        if not aliases:
            continue
        canonical_name = normalize_channel_label(str(row['Channel_Names']).split(';')[0])
        alias_lookup[canonical_name] = aliases
    return alias_lookup


def _get_eeg_aliases(csv_path):
    channels, normalized_csv_path = get_cached_channel_table(csv_path)
    eeg_aliases = EEG_ALIASES_CACHE.get(normalized_csv_path)
    if eeg_aliases is None:
        eeg_aliases = _build_eeg_aliases(channels)
        EEG_ALIASES_CACHE[normalized_csv_path] = eeg_aliases
    return eeg_aliases


def _get_channel_signal(channel_name, physiological_data, physiological_fs, eeg_aliases):
    channel_spec = EEG_CHANNEL_SPECS[channel_name]
    direct_aliases = eeg_aliases.get(normalize_channel_label(channel_spec['direct']), set())
    direct_label = find_matching_label(physiological_data, direct_aliases)
    if direct_label is not None and direct_label in physiological_fs:
        return np.asarray(physiological_data[direct_label], dtype=float), physiological_fs[direct_label]

    positive_aliases = eeg_aliases.get(normalize_channel_label(channel_spec['positive']), set())
    reference_aliases = eeg_aliases.get(normalize_channel_label(channel_spec['reference']), set())
    positive_label = find_matching_label(physiological_data, positive_aliases)
    reference_label = find_matching_label(physiological_data, reference_aliases)
    if positive_label is None or reference_label is None:
        return None, None
    if positive_label not in physiological_fs or reference_label not in physiological_fs:
        return None, None

    positive_fs = physiological_fs[positive_label]
    reference_fs = physiological_fs[reference_label]
    if positive_fs != reference_fs:
        return None, None

    return (
        np.asarray(physiological_data[positive_label], dtype=float)
        - np.asarray(physiological_data[reference_label], dtype=float),
        positive_fs,
    )


def _extract_channel_metrics(signal, fs):
    signal = np.nan_to_num(np.asarray(signal, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if signal.size < max(int(fs * 30), 2):
        return None

    if fs != 200:
        signal, fs = resample_signal(signal, fs, 200)

    filtered = eeg_features.butter_bandpass_filter(signal, lowcut=0.3, highcut=35, fs=fs, order=4)
    signal_std = np.std(filtered)
    if signal_std == 0 or not np.isfinite(signal_std):
        return None

    normalized = (filtered - np.mean(filtered)) / signal_std
    epochs = eeg_features.create_epochs(normalized, fs, epoch_duration=30)
    if epochs.size == 0:
        return None

    band_powers, complexities = eeg_features.extract_band_powers(epochs, fs, win_len=15)
    if len(band_powers) > 60:
        band_powers = band_powers.iloc[60:]
        complexities = complexities.iloc[60:]
    if band_powers.empty:
        return None

    patient_profile = eeg_features.get_patient_profile(band_powers)
    metrics = {
        str(name): float(value)
        for name, value in patient_profile.replace([np.inf, -np.inf], np.nan).items()
    }
    for complexity_name in ('Hjorth_Mobility', 'Hjorth_Complexity'):
        if complexity_name in complexities:
            value = complexities[complexity_name].replace([np.inf, -np.inf], np.nan).std()
            metrics[complexity_name] = float(np.nan if pd.isna(value) else value)
        else:
            metrics[complexity_name] = np.nan
    return metrics


def processEEG(physiological_data, physiological_fs, csv_path):
    eeg_aliases = _get_eeg_aliases(csv_path)
    channel_profiles = {}

    for channel_name in EEG_CHANNEL_SPECS:
        signal, fs = _get_channel_signal(channel_name, physiological_data, physiological_fs, eeg_aliases)
        if signal is None or fs is None:
            continue

        metrics = _extract_channel_metrics(signal, fs)
        if metrics is not None:
            channel_profiles[channel_name] = metrics

    if not channel_profiles:
        return np.full(EEG_FEATURE_LENGTH, np.nan, dtype=np.float32)

    values = []
    for channel_name, metric_name in EEG_FEATURE_SPECS:
        channel_metrics = channel_profiles.get(channel_name)
        if channel_metrics is None:
            values.append(np.nan)
            continue
        values.append(float(channel_metrics.get(metric_name, np.nan)))

    return np.asarray(values, dtype=np.float32)


_normalize_label = normalize_channel_label