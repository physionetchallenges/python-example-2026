import numpy as np
from scipy.signal import butter, filtfilt, resample
from .ecg_peak_detection import pan_tompkins
from .ecg_hrv_features import compute_hrv_hrf
from .ecg_nn_interpolation import interpolate_nn_pchip
from .ecg_rr_cleaning import remove_ectopic_beats


def compute_ecg_features(ecg_signal, fs, ecg_feature_length):

    ecg_signal = ecg_signal - np.mean(ecg_signal)

    target_fs = 200

    if fs != target_fs:
        num_samples = int(len(ecg_signal) * target_fs / fs)
        ecg_signal = resample(ecg_signal, num_samples)
        fs = target_fs

    win_sec = 300
    win_samples = int(win_sec * fs)

    total_samples = len(ecg_signal)
    n_windows = total_samples // win_samples

    if n_windows == 0:
        print("Signal too short.")
        return None

    valid_windows = []

    for window_index in range(n_windows):

        idx_start = window_index * win_samples
        idx_end = (window_index + 1) * win_samples

        ecg_win = ecg_signal[idx_start:idx_end]

        if np.sum(np.isnan(ecg_win)) != 0 or np.sum(ecg_win == 0) > 0.2 * len(ecg_win):
            continue

        valid_windows.append(window_index)

    hrv_all = []

    for window_index in valid_windows:

        idx_start = window_index * win_samples
        idx_end = (window_index + 1) * win_samples

        ecg_win = ecg_signal[idx_start:idx_end]

        ecg_win = ecg_win - np.mean(ecg_win)

        b, a = butter(3, [59.5/(fs/2), 60.5/(fs/2)], btype='bandstop')
        ecg_win = filtfilt(b, a, ecg_win)

        b, a = butter(3, 0.5/(fs/2), btype='high')
        ecg_win = filtfilt(b, a, ecg_win)

        b, a = butter(3, 45/(fs/2), btype='low')
        ecg_win = filtfilt(b, a, ecg_win)

        _, r_locs, _ = pan_tompkins(ecg_win, fs, 0)

        if len(r_locs) < 150:
            continue

        nn_intervals = np.diff(r_locs) / fs

        nn_intervals, ectopic_perc = remove_ectopic_beats(nn_intervals, 40, 0.10)

        nn_intervals = interpolate_nn_pchip(nn_intervals, 2)

        valid_ratio = np.sum(~np.isnan(nn_intervals)) / len(nn_intervals)
        nn_intervals = nn_intervals[~np.isnan(nn_intervals)]

        if valid_ratio < 0.75:
            continue

        metrics = compute_hrv_hrf(nn_intervals, fs)

        hrv_all.append([
            metrics["PIP"], metrics["PNNLS"], metrics["PNNSS"],
            metrics["AVNN"], metrics["SDNN"], metrics["RMSSD"], metrics["HF"],
            ectopic_perc
        ])

    if len(hrv_all) == 0:
        return np.zeros(ecg_feature_length, dtype=np.float32)

    hrv_all = np.array(hrv_all)

    median_vals = np.nanmedian(hrv_all, axis=0)
    std_vals = np.nanstd(hrv_all, axis=0)

    features = np.array([
        median_vals[0], std_vals[0],
        median_vals[1], std_vals[1],
        median_vals[2], std_vals[2],
        median_vals[3], std_vals[3],
        median_vals[4], std_vals[4],
        median_vals[5], std_vals[5],
        median_vals[6], std_vals[6],
        median_vals[7], std_vals[7],
    ], dtype=np.float32)

    return features