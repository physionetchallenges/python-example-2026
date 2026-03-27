import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from lib.pan_tompkins import pan_tompkin
from lib.compute_hrv_hrf import compute_HRV_HRF
from lib.interpolate_NN import interpolate_NN_pchip
from lib.remove_ectopic_beat import remove_ectopic_beats
def ECGprocessing(ecg_signal, fs, patient_id):

    all_results = pd.DataFrame()

    ecg_signal = ecg_signal - np.mean(ecg_signal)

    # ===============================
    # RESAMPLE TO 200 Hz IF NEEDED
    # ===============================
    target_fs = 200

    if fs != target_fs:
        num_samples = int(len(ecg_signal) * target_fs / fs)
        ecg_signal = resample(ecg_signal, num_samples)
        fs = target_fs

    # ===============================
    # SEGMENT INTO 5-MIN WINDOWS
    # ===============================
    win_sec = 300
    win_samples = int(win_sec * fs)

    N = len(ecg_signal)
    n_windows = N // win_samples

    if n_windows == 0:
        print("Signal too short.")
        return None

    # ===============================
    # FIND VALID WINDOWS
    # ===============================
    valid_windows = []

    for w in range(n_windows):

        idx_start = w * win_samples
        idx_end = (w + 1) * win_samples

        ecg_win = ecg_signal[idx_start:idx_end]

        # Quality check
        if np.sum(np.isnan(ecg_win)) != 0 or np.sum(ecg_win == 0) > 0.2 * len(ecg_win):
            continue

        valid_windows.append(w)

    # ===============================
    # PROCESS WINDOWS
    # ===============================
    HRV_all = []

    for w in valid_windows:

        idx_start = w * win_samples
        idx_end = (w + 1) * win_samples

        ecg_win = ecg_signal[idx_start:idx_end]

        ecg_win = ecg_win - np.mean(ecg_win)

        # --- Filtering ---
        # Notch
        b, a = butter(3, [59.5/(fs/2), 60.5/(fs/2)], btype='bandstop')
        ecg_win = filtfilt(b, a, ecg_win)

        # High-pass
        b, a = butter(3, 0.5/(fs/2), btype='high')
        ecg_win = filtfilt(b, a, ecg_win)

        # Low-pass
        b, a = butter(3, 45/(fs/2), btype='low')
        ecg_win = filtfilt(b, a, ecg_win)

        # ===============================
        # QRS DETECTION
        # ===============================
        qrs_amp_raw, R_locs, delay = pan_tompkin(ecg_win, fs, 0)

        if len(R_locs) < 150:
            continue

        # ===============================
        # NN INTERVALS
        # ===============================
        NN = np.diff(R_locs) / fs

        # ===============================
        # HRV PREPROCESSING
        # ===============================
        NN, ectopic_perc = remove_ectopic_beats(NN, 40, 0.10)

        NN = interpolate_NN_pchip(NN, 2)

        valid_ratio = np.sum(~np.isnan(NN)) / len(NN)
        NN = NN[~np.isnan(NN)]

        if valid_ratio < 0.75:
            continue

        # ===============================
        # HRV + HRF METRICS
        # ===============================
        res = compute_HRV_HRF(NN, fs)

        meanNN = np.mean(NN)

        HRV_all.append([
            meanNN,
            res["PIP"], res["PNNLS"], res["PNNSS"],
            res["AVNN"], res["SDNN"], res["RMSSD"], res["HF"],
            ectopic_perc
        ])

    # ===============================
    # SUBJECT-LEVEL METRICS
    # ===============================
    if len(HRV_all) == 0:
        print("No valid windows.")
        return None

    HRV_all = np.array(HRV_all)

    median_vals = np.nanmedian(HRV_all, axis=0)
    std_vals = np.nanstd(HRV_all, axis=0)

    # ===============================
    # SAVE RESULTS (DataFrame row)
    # ===============================
    row = pd.DataFrame([{
        "ID": patient_id,
        "mNNmed": median_vals[0], "mNNstd": std_vals[0],
        "PIP_med": median_vals[1], "PIP_std": std_vals[1],
        "PNNLS_med": median_vals[2], "PNNLS_std": std_vals[2],
        "PNNSS_med": median_vals[3], "PNNSS_std": std_vals[3],
        "AVNN_med": median_vals[4], "AVNN_std": std_vals[4],
        "SDNN_med": median_vals[5], "SDNN_std": std_vals[5],
        "RMSSD_med": median_vals[6], "RMSSD_std": std_vals[6],
        "HF_med": median_vals[7], "HF_std": std_vals[7],
        "ECTOPIC_med": median_vals[8], "ECTOPIC_std": std_vals[8],
    }])

    all_results = pd.concat([all_results, row], ignore_index=True)

    return all_results