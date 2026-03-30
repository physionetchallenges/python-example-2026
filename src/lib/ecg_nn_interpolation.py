import numpy as np
from scipy.interpolate import PchipInterpolator

def interpolate_nn_pchip(nn_intervals, max_gap):
    """Interpolate short NaN gaps in NN intervals using PCHIP."""

    nn_intervals = np.asarray(nn_intervals).flatten()
    nn_interpolated = nn_intervals.copy()

    nan_indices = np.isnan(nn_intervals)

    boundaries = np.diff(np.concatenate(([0], nan_indices.astype(int), [0])))
    start_indices = np.where(boundaries == 1)[0]
    end_indices = np.where(boundaries == -1)[0] - 1

    for index in range(len(start_indices)):
        segment_length = end_indices[index] - start_indices[index] + 1

        if segment_length <= max_gap:
            left = start_indices[index] - 1
            right = end_indices[index] + 1

            if (left >= 0 and right < len(nn_intervals) and
                not np.isnan(nn_intervals[left]) and not np.isnan(nn_intervals[right])):

                x = np.array([left, right])
                y = np.array([nn_intervals[left], nn_intervals[right]])

                xi = np.arange(start_indices[index], end_indices[index] + 1)

                interpolator = PchipInterpolator(x, y)
                nn_interpolated[xi] = interpolator(xi)

    return nn_interpolated