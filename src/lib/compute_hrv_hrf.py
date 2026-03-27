import numpy as np
from scipy.signal import lombscargle

def compute_HRV_HRF(NN, SF):
    """
    NN: array of NN intervals in seconds
    SF: sampling frequency (Hz)
    """

    NN = np.asarray(NN).flatten()

    # ===============================
    # ΔNN
    # ===============================
    dNN = np.diff(NN)

    n = 1
    thr = n / SF

    # Classification
    acc = dNN <= -thr
    dec = dNN >= thr
    noch = (dNN > -thr) & (dNN < thr)

    # Sign representation
    sign_dNN = np.zeros_like(dNN)
    sign_dNN[acc] = -1
    sign_dNN[dec] = 1

    N = len(dNN)

    # ===============================
    # PIP (Inflection Points)
    # ===============================
    inflection = 0

    for i in range(N - 1):
        if (dNN[i+1] * dNN[i] <= 0) and (dNN[i+1] != dNN[i]):
            inflection += 1

    PIP = (inflection / (N - 1)) * 100 if N > 1 else np.nan

    # ===============================
    # Segment Detection
    # ===============================
    segments = []
    if N > 0:
        current_seg = sign_dNN[0]
        length_seg = 1

        for i in range(1, N):
            if sign_dNN[i] == current_seg and sign_dNN[i] != 0:
                length_seg += 1
            else:
                if current_seg != 0:
                    segments.append(length_seg)
                current_seg = sign_dNN[i]
                length_seg = 1

        # Add last segment
        if current_seg != 0:
            segments.append(length_seg)

    segments = np.array(segments)

    # ===============================
    # PNNLS & PNNSS
    # ===============================
    if len(segments) > 0:
        long_segments = segments[segments >= 3]
        short_segments = segments[segments < 3]

        PNNLS = np.sum(long_segments) / N * 100
        PNNSS = np.sum(short_segments) / np.sum(segments) * 100
    else:
        PNNLS = np.nan
        PNNSS = np.nan

    # ===============================
    # Time-domain HRV
    # ===============================
    win_length = 300  # seconds

    time = np.cumsum(NN)

    AVNN_all = []
    SDNN_all = []
    RMSSD_all = []

    i = 0
    while i < len(NN):
        t_start = time[i]
        t_end = t_start + win_length

        idx = np.where((time >= t_start) & (time < t_end))[0]

        if len(idx) >= 150:
            NN_win = NN[idx]

            AVNN_all.append(np.nanmean(NN_win))
            SDNN_all.append(np.nanstd(NN_win, ddof=1))

            diffNN = np.diff(NN_win)
            RMSSD_all.append(np.sqrt(np.nanmean(diffNN**2)))

        next_i = np.where(time >= t_end)[0]
        if len(next_i) == 0:
            break
        i = next_i[0]

    AVNN = np.nanmean(AVNN_all) if len(AVNN_all) > 0 else np.nan
    SDNN = np.nanmean(SDNN_all) if len(SDNN_all) > 0 else np.nan
    RMSSD = np.nanmean(RMSSD_all) if len(RMSSD_all) > 0 else np.nan

    # ===============================
    # Frequency-domain (HF)
    # ===============================
    HF_all = []

    for _ in range(len(AVNN_all)):
        # NOTE: simplified like MATLAB version
        NN_win = NN.copy()
        t_win = np.cumsum(NN_win)

        # Convert to angular frequency
        f = np.linspace(0.01, 0.5, 1000)
        angular_f = 2 * np.pi * f
       
        # Remove mean (important for Lomb)
        NN_detrended = NN_win - np.mean(NN_win)

        Pxx = lombscargle(t_win, NN_detrended, angular_f, normalize=True)

        HF_band = (f >= 0.15) & (f <= 0.4)

        HF_power = np.trapezoid(Pxx[HF_band], f[HF_band])

        HF_all.append(HF_power)

    HF = np.nanmean(HF_all) if len(HF_all) > 0 else np.nan

    # ===============================
    # OUTPUT
    # ===============================
    results = {
        "PIP": PIP,
        "PNNLS": PNNLS,
        "PNNSS": PNNSS,
        "AVNN": AVNN,
        "SDNN": SDNN,
        "RMSSD": RMSSD,
        "HF": HF
    }

    return results