import numpy as np

def remove_ectopic_beats(nn_intervals, window_size, threshold):
    nn_intervals = np.asarray(nn_intervals).flatten()
    nn_corrected = nn_intervals.copy()

    half_win = window_size // 2
    ectopic_count = 0
    valid_count = 0

    for index in range(len(nn_intervals)):

        if np.isnan(nn_intervals[index]):
            continue

        valid_count += 1

        left = max(0, index - half_win)
        right = min(len(nn_intervals), index + half_win + 1)

        local_segment = nn_intervals[left:right]
        local_segment = local_segment[~np.isnan(local_segment)]

        if local_segment.size == 0:
            continue

        med_val = np.median(local_segment)

        if abs(nn_intervals[index] - med_val) > threshold * med_val:
            nn_corrected[index] = med_val
            ectopic_count += 1

    if valid_count > 0:
        ectopic_perc = (ectopic_count / valid_count) * 100
    else:
        ectopic_perc = np.nan

    return nn_corrected, ectopic_perc