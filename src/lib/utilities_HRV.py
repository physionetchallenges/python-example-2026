import numpy as np
from scipy.interpolate import PchipInterpolator

def interpolate_NN_pchip(NN, maxGap):
    """
    NN: array of NN intervals (seconds)
    maxGap: max number of consecutive NaNs allowed for interpolation
    """
    
    NN = np.asarray(NN).flatten()
    NN_interp = NN.copy()

    nan_idx = np.isnan(NN)

    # Find NaN segments
    d = np.diff(np.concatenate(([0], nan_idx.astype(int), [0])))
    start_idx = np.where(d == 1)[0]
    end_idx = np.where(d == -1)[0] - 1

    for k in range(len(start_idx)):
        seg_len = end_idx[k] - start_idx[k] + 1

        if seg_len <= maxGap:
            left = start_idx[k] - 1
            right = end_idx[k] + 1

            # Check bounds
            if (left >= 0 and right < len(NN) and
                not np.isnan(NN[left]) and not np.isnan(NN[right])):

                x = np.array([left, right])
                y = np.array([NN[left], NN[right]])

                xi = np.arange(start_idx[k], end_idx[k] + 1)

                # PCHIP interpolation
                interpolator = PchipInterpolator(x, y)
                NN_interp[xi] = interpolator(xi)

    return NN_interp

def remove_ectopic_beats(NN, window_size, threshold):
    NN = np.asarray(NN).flatten()
    NN_corrected = NN.copy()

    half_win = window_size // 2
    ectopic_count = 0
    valid_count = 0

    for i in range(len(NN)):

        if np.isnan(NN[i]):
            continue

        valid_count += 1

        # Define local window
        left = max(0, i - half_win)
        right = min(len(NN), i + half_win + 1)  # Python slice is exclusive

        local_segment = NN[left:right]
        local_segment = local_segment[~np.isnan(local_segment)]

        if local_segment.size == 0:
            continue

        med_val = np.median(local_segment)

        # Detect ectopic
        if abs(NN[i] - med_val) > threshold * med_val:
            NN_corrected[i] = med_val
            ectopic_count += 1

    # Percentage over valid NN
    if valid_count > 0:
        ectopic_perc = (ectopic_count / valid_count) * 100
    else:
        ectopic_perc = np.nan

    return NN_corrected, ectopic_perc