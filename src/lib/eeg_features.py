"""EEG feature helpers used by the active submission pipeline."""

from contextlib import nullcontext, redirect_stdout
import io

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis, entropy

from .swa import swa_CalculateReference
from .swa import swa_FindSWChannels
from .swa import swa_FindSWRef
from .swa import swa_getInfoDefaults


SLOW_WAVE_FEATURE_NAMES = (
    'TotalSW',
    'SWdensity',
    'SWpeakAmp_mean',
    'SWpeakAmp_std',
    'SWp2p_mean',
    'SWp2p_std',
    'SWnegSlope_mean',
    'SWnegSlope_std',
    'SWposSlope_mean',
    'SWposSlope_std',
    'SWduration_mean',
    'SWduration_std',
)

_SLOW_WAVE_EVENT_FIELDS = {
    'SWpeakAmp': 'Ref_PeakAmp',
    'SWp2p': 'Ref_P2PAmp',
    'SWnegSlope': 'Ref_NegSlope',
    'SWposSlope': 'Ref_PosSlope',
}

def _safe_sqrt_variance_ratio(numerator_signal, denominator_signal):
    numerator_var = np.var(numerator_signal)
    denominator_var = np.var(denominator_signal)
    if denominator_var <= 0 or not np.isfinite(denominator_var):
        return 0.0
    ratio = numerator_var / denominator_var
    if ratio <= 0 or not np.isfinite(ratio):
        return 0.0
    return float(np.sqrt(ratio))

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    y = filtfilt(b, a, data) 
    return y

def create_epochs(data, fs, epoch_duration=30):
    samples_per_epoch = int(fs * epoch_duration)
    num_epochs = len(data) // samples_per_epoch

    data_trimmed = data[:num_epochs * samples_per_epoch]
    epochs = data_trimmed.reshape(num_epochs, samples_per_epoch)
    return epochs

def extract_band_powers(epochs, fs, win_len = 2):
    features = []
    complexities = []
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Sigma': (11, 16),
        'Beta': (12, 30)
    }

    for epoch in epochs:
        freqs, psd = welch(epoch, fs, nperseg=fs*30)
        epoch_features = {}
        for band_name, (low, high) in bands.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            epoch_features[band_name] = np.mean(psd[idx_band])

        features.append(epoch_features)

        diff = np.diff(epoch)
        mobility = _safe_sqrt_variance_ratio(diff, epoch)
        diff2 = np.diff(diff)
        mobility_diff = _safe_sqrt_variance_ratio(diff2, diff)
        complexity = mobility_diff / mobility if mobility > 0 else 0
        complexities.append({'Hjorth_Mobility': mobility, 'Hjorth_Complexity': complexity})

    return pd.DataFrame(features), pd.DataFrame(complexities)

def _finite_slow_wave_values(slow_waves, field_name):
    values = []
    for wave in slow_waves:
        try:
            value = float(np.asarray(wave[field_name]).squeeze())
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def summarize_slow_waves(slow_waves, fs, signal_duration_seconds):
    """Aggregate the event dictionaries returned by ``swa`` into scalar features.

    Amplitudes and slopes retain the sign and units returned by ``swa``.
    Durations and density are expressed in seconds and waves/minute, respectively.
    """
    if fs <= 0 or not np.isfinite(fs):
        raise ValueError('fs must be a positive finite sampling frequency.')
    if signal_duration_seconds <= 0 or not np.isfinite(signal_duration_seconds):
        raise ValueError('signal_duration_seconds must be positive and finite.')

    slow_waves = [] if slow_waves is None else list(slow_waves)
    features = {name: np.nan for name in SLOW_WAVE_FEATURE_NAMES}
    features['TotalSW'] = float(len(slow_waves))
    features['SWdensity'] = float(len(slow_waves) / (signal_duration_seconds / 60.0))

    for feature_prefix, event_field in _SLOW_WAVE_EVENT_FIELDS.items():
        values = _finite_slow_wave_values(slow_waves, event_field)
        if values.size:
            features[f'{feature_prefix}_mean'] = float(np.mean(values))
            features[f'{feature_prefix}_std'] = float(np.std(values))

    durations = []
    for wave in slow_waves:
        try:
            duration = (
                float(np.asarray(wave['Ref_UpInd']).squeeze())
                - float(np.asarray(wave['Ref_DownInd']).squeeze())
            ) / float(fs)
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(duration) and duration > 0:
            durations.append(duration)

    if durations:
        features['SWduration_mean'] = float(np.mean(durations))
        features['SWduration_std'] = float(np.std(durations))

    return features


def get_SW_features(signal, fs, verbose=False):
    """Detect slow waves with ``swa`` and return fixed-length scalar features."""
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if fs <= 0 or not np.isfinite(fs):
        raise ValueError('fs must be a positive finite sampling frequency.')
    if signal.size == 0:
        raise ValueError('signal must contain at least one sample.')
    if not np.all(np.isfinite(signal)):
        raise ValueError('signal must contain only finite values.')

    info = swa_getInfoDefaults.swa_getInfoDefaults({}, 'SW', method='envelope')
    info['Electrodes'] = ['EEG']
    info['Recording'] = {'sRate': float(fs)}
    info['Parameters']['Ref_InspectionPoint'] = 'ZC'
    # Spatial clustering is undefined for a single-channel invocation.
    info['Parameters']['Channels_ClusterTest'] = False

    # swa consistently uses (channels, samples).
    data = {'Raw': signal[np.newaxis, :]}
    output_context = nullcontext() if verbose else redirect_stdout(io.StringIO())
    with output_context:
        data['SWRef'], info = swa_CalculateReference.swa_CalculateReference(
            data['Raw'], info, False
        )
        data, info, slow_waves = swa_FindSWRef.swa_FindSWRef(data, info)
        data, info, slow_waves = swa_FindSWChannels.swa_FindSWChannels(
            data, info, slow_waves, flag_progress=verbose
        )

    return summarize_slow_waves(
        slow_waves,
        fs=float(fs),
        signal_duration_seconds=signal.size / float(fs),
    )

def get_patient_profile(df_features):
    total_power = df_features.sum(axis=1)
    avg_p = df_features.mean()
    total_avg_p = avg_p.sum()

    variability = df_features.std() / df_features.mean()
    variability.index = ['CV_' + col for col in variability.index]

    kurt = df_features.apply(kurtosis)
    kurt.index = ['Kurt_' + col for col in kurt.index]

    rel_delta = avg_p['Delta'] / total_avg_p

    tar = avg_p['Theta'] / avg_p['Alpha']
    tbr = avg_p['Theta'] / avg_p['Beta']

    spec_entropy = entropy(df_features)

    rel_powers = df_features.div(total_power, axis=0).mean()
    rel_powers.index = ['Rel_' + col for col in rel_powers.index]

    avg_p = df_features.mean()
    ratios = {
        'Ratio_Theta_Alpha': avg_p['Theta'] / avg_p['Alpha'],
        'Ratio_Slow_Fast': (avg_p['Delta'] + avg_p['Theta']) / (avg_p['Alpha'] + avg_p['Beta']),
        'Sigma_Stability': df_features['Sigma'].std() / df_features['Sigma'].mean(),
        'Spectral_Entropy_delta': spec_entropy[0],
        'Spectral_Entropy_theta': spec_entropy[1],
        'Spectral_Entropy_alpha': spec_entropy[2],
        'Spectral_Entropy_sigma': spec_entropy[3],
        'Spectral_Entropy_beta': spec_entropy[4],
        'Theta_Alpha_Ratio': tar,
        'Theta_Beta_Ratio': tbr,
        'Relative_Delta_Power': rel_delta,
        'kurtosis_Delta': kurt['Kurt_Delta'],
        'kurtosis_Theta': kurt['Kurt_Theta'],
        'kurtosis_Alpha': kurt['Kurt_Alpha'],
        'kurtosis_Sigma': kurt['Kurt_Sigma'],
        'kurtosis_Beta': kurt['Kurt_Beta'],
        'variability_Delta': variability['CV_Delta'],
        'variability_Theta': variability['CV_Theta'],
        'variability_Alpha': variability['CV_Alpha'],
        'variability_Sigma': variability['CV_Sigma'],
        'variability_Beta': variability['CV_Beta'],

    }

    profile = pd.concat([rel_powers, pd.Series(ratios)])
    return profile
