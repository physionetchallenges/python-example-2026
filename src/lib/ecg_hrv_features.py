import numpy as np
from scipy.signal import lombscargle

def compute_hrv_hrf(nn_intervals, sampling_frequency):
    """Compute time-domain and HF HRV metrics from NN intervals."""

    nn_intervals = np.asarray(nn_intervals).flatten()

    delta_nn = np.diff(nn_intervals)

    n = 1
    threshold = n / sampling_frequency

    acceleration = delta_nn <= -threshold
    deceleration = delta_nn >= threshold

    sign_delta_nn = np.zeros_like(delta_nn)
    sign_delta_nn[acceleration] = -1
    sign_delta_nn[deceleration] = 1

    num_deltas = len(delta_nn)

    inflection = 0

    for index in range(num_deltas - 1):
        if (delta_nn[index + 1] * delta_nn[index] <= 0) and (delta_nn[index + 1] != delta_nn[index]):
            inflection += 1

    pip = (inflection / (num_deltas - 1)) * 100 if num_deltas > 1 else np.nan

    segments = []
    if num_deltas > 0:
        current_segment = sign_delta_nn[0]
        segment_length = 1

        for index in range(1, num_deltas):
            if sign_delta_nn[index] == current_segment and sign_delta_nn[index] != 0:
                segment_length += 1
            else:
                if current_segment != 0:
                    segments.append(segment_length)
                current_segment = sign_delta_nn[index]
                segment_length = 1

        if current_segment != 0:
            segments.append(segment_length)

    segments = np.array(segments)

    if len(segments) > 0:
        long_segments = segments[segments >= 3]
        short_segments = segments[segments < 3]

        pnnls = np.sum(long_segments) / num_deltas * 100
        pnnss = np.sum(short_segments) / np.sum(segments) * 100
    else:
        pnnls = np.nan
        pnnss = np.nan

    window_length = 300
    minimum_intervals_per_window = window_length / 2

    elapsed_time = np.cumsum(nn_intervals)

    avnn_values = []
    sdnn_values = []
    rmssd_values = []

    index = 0
    while index < len(nn_intervals):
        window_start = elapsed_time[index]
        window_end = window_start + window_length

        window_indices = np.where((elapsed_time >= window_start) & (elapsed_time < window_end))[0]

        if len(window_indices) >= minimum_intervals_per_window:
            window_nn = nn_intervals[window_indices]

            avnn_values.append(np.nanmean(window_nn))
            sdnn_values.append(np.nanstd(window_nn, ddof=1))

            diff_nn = np.diff(window_nn)
            rmssd_values.append(np.sqrt(np.nanmean(diff_nn**2)))

        next_indices = np.where(elapsed_time >= window_end)[0]
        if len(next_indices) == 0:
            break
        index = next_indices[0]

    avnn = np.nanmean(avnn_values) if len(avnn_values) > 0 else np.nan
    sdnn = np.nanmean(sdnn_values) if len(sdnn_values) > 0 else np.nan
    rmssd = np.nanmean(rmssd_values) if len(rmssd_values) > 0 else np.nan

    hf_values = []

    for _ in range(len(avnn_values)):
        window_nn = nn_intervals.copy()
        window_time = np.cumsum(window_nn)

        frequencies = np.linspace(0.01, 0.5, 1000)
        angular_frequencies = 2 * np.pi * frequencies
        detrended_nn = window_nn - np.nanmean(window_nn)

        power_spectrum = lombscargle(window_time, detrended_nn, angular_frequencies, normalize=True)

        hf_band = (frequencies >= 0.15) & (frequencies <= 0.4)

        hf_power = np.trapezoid(power_spectrum[hf_band], frequencies[hf_band])

        hf_values.append(hf_power)

    hf = np.nanmean(hf_values) if len(hf_values) > 0 else np.nan

    results = {
        "PIP": pip,
        "PNNLS": pnnls,
        "PNNSS": pnnss,
        "AVNN": avnn,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "HF": hf,
    }

    return results