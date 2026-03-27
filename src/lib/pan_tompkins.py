import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def pan_tompkin(ecg, fs, gr=0):

    ecg = np.asarray(ecg).flatten()
    delay = 0

    skip = 0
    m_selected_RR = 0
    mean_RR = 0
    ser_back = 0

    # ===================== FILTERING ===================== #
    ecg = ecg - np.mean(ecg)

    if fs == 200:
        # Low-pass
        b, a = butter(3, 12*2/fs, btype='low')
        ecg_l = filtfilt(b, a, ecg)
        ecg_l = ecg_l / np.max(np.abs(ecg_l))

        # High-pass
        b, a = butter(3, 5*2/fs, btype='high')
        ecg_h = filtfilt(b, a, ecg_l)
        ecg_h = ecg_h / np.max(np.abs(ecg_h))
    else:
        b, a = butter(3, [5*2/fs, 15*2/fs], btype='band')
        ecg_h = filtfilt(b, a, ecg)
        ecg_h = ecg_h / np.max(np.abs(ecg_h))

    # ===================== DERIVATIVE ===================== #
    if fs != 200:
        int_c = int((5 - 1) / (fs * (1/40)))
        base = np.array([1, 2, 0, -2, -1]) * (1/8) * fs
        x_old = np.linspace(1, 5, 5)
        x_new = np.linspace(1, 5, int_c)
        b = np.interp(x_new, x_old, base)
    else:
        b = np.array([1, 2, 0, -2, -1]) * (1/8) * fs

    ecg_d = filtfilt(b, [1], ecg_h)
    ecg_d = ecg_d / np.max(np.abs(ecg_d))

    # ===================== SQUARING ===================== #
    ecg_s = ecg_d ** 2

    # ===================== MOVING WINDOW ===================== #
    win = int(round(0.150 * fs))
    ecg_m = np.convolve(ecg_s, np.ones(win)/win, mode='same')
    delay += win // 2

    # ===================== PEAK DETECTION ===================== #
    locs, _ = find_peaks(ecg_m, distance=int(0.2 * fs))
    pks = ecg_m[locs]

    LLp = len(pks)

    qrs_i = []
    qrs_c = []
    qrs_i_raw = []
    qrs_amp_raw = []

    nois_i = []
    nois_c = []

    # Threshold initialization
    THR_SIG = np.max(ecg_m[:2*fs]) / 3
    THR_NOISE = np.mean(ecg_m[:2*fs]) / 2
    SIG_LEV = THR_SIG
    NOISE_LEV = THR_NOISE

    THR_SIG1 = np.max(ecg_h[:2*fs]) / 3
    THR_NOISE1 = np.mean(ecg_h[:2*fs]) / 2
    SIG_LEV1 = THR_SIG1
    NOISE_LEV1 = THR_NOISE1

    Beat_C = 0
    Beat_C1 = 0

    for i in range(LLp):

        loc = locs[i]

        # Find peak in filtered signal
        left = max(0, loc - int(0.150 * fs))
        right = loc

        if right < len(ecg_h):
            segment = ecg_h[left:right+1]
            if len(segment) > 0:
                y_i = np.max(segment)
                x_i = np.argmax(segment)
            else:
                continue
        else:
            continue

        # RR interval update
        if len(qrs_i) >= 9:
            diffRR = np.diff(qrs_i[-8:])
            mean_RR = np.mean(diffRR)
            comp = qrs_i[-1] - qrs_i[-2]

            if comp <= 0.92 * mean_RR or comp >= 1.16 * mean_RR:
                THR_SIG *= 0.5
                THR_SIG1 *= 0.5
            else:
                m_selected_RR = mean_RR

        test_m = m_selected_RR if m_selected_RR else mean_RR

        # ===================== SEARCH BACK ===================== #
        if test_m and len(qrs_i) > 0:
            if (loc - qrs_i[-1]) >= int(1.66 * test_m):

                sb_left = qrs_i[-1] + int(0.2 * fs)
                sb_right = loc - int(0.2 * fs)

                if sb_right > sb_left:
                    segment = ecg_m[sb_left:sb_right]
                    if len(segment) > 0:
                        pks_temp = np.max(segment)
                        locs_temp = sb_left + np.argmax(segment)

                        if pks_temp > THR_NOISE:
                            qrs_c.append(pks_temp)
                            qrs_i.append(locs_temp)

                            seg = ecg_h[max(0, locs_temp-int(0.150*fs)):locs_temp]
                            if len(seg) > 0:
                                y_i_t = np.max(seg)
                                x_i_t = np.argmax(seg)

                                if y_i_t > THR_NOISE1:
                                    qrs_i_raw.append(locs_temp - int(0.150*fs) + x_i_t)
                                    qrs_amp_raw.append(y_i_t)
                                    SIG_LEV1 = 0.25*y_i_t + 0.75*SIG_LEV1

                            SIG_LEV = 0.25*pks_temp + 0.75*SIG_LEV

        # ===================== CLASSIFICATION ===================== #
        if pks[i] >= THR_SIG:

            # T-wave rejection
            if len(qrs_i) >= 3:
                if (loc - qrs_i[-1]) <= int(0.36 * fs):

                    slope1 = np.mean(np.diff(ecg_m[max(0, loc-int(0.075*fs)):loc]))
                    slope2 = np.mean(np.diff(ecg_m[max(0, qrs_i[-1]-int(0.075*fs)):qrs_i[-1]]))

                    if abs(slope1) <= 0.5 * abs(slope2):
                        NOISE_LEV1 = 0.125*y_i + 0.875*NOISE_LEV1
                        NOISE_LEV = 0.125*pks[i] + 0.875*NOISE_LEV
                        continue

            # Accept QRS
            qrs_c.append(pks[i])
            qrs_i.append(loc)

            if y_i >= THR_SIG1:
                qrs_i_raw.append(loc - int(0.150*fs) + x_i)
                qrs_amp_raw.append(y_i)
                SIG_LEV1 = 0.125*y_i + 0.875*SIG_LEV1

            SIG_LEV = 0.125*pks[i] + 0.875*SIG_LEV

        elif THR_NOISE <= pks[i] < THR_SIG:
            NOISE_LEV1 = 0.125*y_i + 0.875*NOISE_LEV1
            NOISE_LEV = 0.125*pks[i] + 0.875*NOISE_LEV

        else:
            NOISE_LEV1 = 0.125*y_i + 0.875*NOISE_LEV1
            NOISE_LEV = 0.125*pks[i] + 0.875*NOISE_LEV

        # Update thresholds
        THR_SIG = NOISE_LEV + 0.25 * abs(SIG_LEV - NOISE_LEV)
        THR_NOISE = 0.5 * THR_SIG

        THR_SIG1 = NOISE_LEV1 + 0.25 * abs(SIG_LEV1 - NOISE_LEV1)
        THR_NOISE1 = 0.5 * THR_SIG1

    return np.array(qrs_amp_raw), np.array(qrs_i_raw), delay