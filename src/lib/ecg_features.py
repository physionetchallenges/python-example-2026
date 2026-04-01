import numpy as np
from typing import cast
from scipy.signal import butter, filtfilt, resample
from .ecg_peak_detection import pan_tompkins
from .ecg_hrv_features import compute_hrv_hrf
from .ecg_nn_interpolation import interpolate_nn_pchip
from .ecg_rr_cleaning import remove_ectopic_beats


def compute_ecg_features(ecg_signal, fs, ecg_feature_length):
    fs = int(round(float(fs)))
    if fs <= 0:
        return None

    ecg_signal = ecg_signal - np.mean(ecg_signal)

    target_fs = 200
    length_ecg=len(ecg_signal)
    
    if fs != target_fs:
        num_samples = int(length_ecg * target_fs / fs)
        ecg_signal = resample(ecg_signal, num_samples)
        fs = target_fs

    if np.sum(np.isnan(ecg_signal)) != 0 or np.sum(ecg_signal == 0) > 0.2 * length_ecg:
        return np.full(ecg_feature_length, np.nan, dtype=np.float32)

    b, a = cast(tuple[np.ndarray, np.ndarray], butter(3, [59.5/(fs/2), 60.5/(fs/2)], btype='bandstop', output='ba'))
    ecg_signal = filtfilt(b, a, ecg_signal)

    b, a = cast(tuple[np.ndarray, np.ndarray], butter(3, 0.5/(fs/2), btype='high', output='ba'))
    ecg_signal = filtfilt(b, a, ecg_signal)

    b, a = cast(tuple[np.ndarray, np.ndarray], butter(3, 45/(fs/2), btype='low', output='ba'))
    ecg_signal = filtfilt(b, a, ecg_signal)

    _, r_locs, _ = pan_tompkins(ecg_signal, fs, 0)

    if len(r_locs) < 150:
        return np.full(ecg_feature_length, np.nan, dtype=np.float32)

    nn_intervals = np.diff(r_locs) / fs

    nn_intervals, ectopic_perc = remove_ectopic_beats(nn_intervals, 40, 0.10)
    nn_intervals = interpolate_nn_pchip(nn_intervals, 2)

    valid_ratio = np.sum(~np.isnan(nn_intervals)) / len(nn_intervals)
    nn_intervals = nn_intervals[~np.isnan(nn_intervals)]

    if valid_ratio < 0.75 or len(nn_intervals) == 0:
        return np.full(ecg_feature_length, np.nan, dtype=np.float32)

    metrics = compute_hrv_hrf(nn_intervals, fs, length_ecg)

    features = np.array([
        metrics["PIP"],
        metrics["PNNLS"],
        metrics["PNNSS"],
        metrics["AVNN"],
        metrics["SDNN"],
        metrics["RMSSD"],
        metrics["HF"],
        ectopic_perc,
    ], dtype=np.float32)

    return features
