import numpy as np

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