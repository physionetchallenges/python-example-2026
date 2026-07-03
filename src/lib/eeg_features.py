"""EEG feature helpers used by the active submission pipeline."""

from scipy.signal import butter, filtfilt
import numpy as np
from scipy.signal import welch
import pandas as pd
from scipy.stats import kurtosis, entropy
from src.lib.swa import swa_getInfoDefaults 
from src.lib.swa import swa_CalculateReference
from src.lib.swa import swa_FindSWRef
from src.lib.swa import swa_FindSWChannels

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

def get_SW_features(signal, fs):
    info = swa_getInfoDefaults.swa_getInfoDefaults({}, 'SW', method='envelope')
    info['Electrodes'] = ['Ej']
    info['Recording'] = {}
    info['Recording']['sRate'] = fs

    data = {}
    data['Raw'] = pd.DataFrame({'Signal': signal})
    data['SWRef'], Info  = swa_CalculateReference.swa_CalculateReference (data['Raw'], info, False)
    Info['Parameters']['Ref_InspectionPoint'] = 'ZC'
    data, Info, SW    = swa_FindSWRef.swa_FindSWRef (data, Info)
    data, Info, SW    = swa_FindSWChannels.swa_FindSWChannels (data, Info, SW)
    return SW

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