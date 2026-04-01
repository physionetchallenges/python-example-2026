import numpy as np
from scipy.integrate import trapezoid
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

    avnn = np.nanmean(nn_intervals) if len(nn_intervals) > 0 else np.nan
    sdnn = np.nanstd(nn_intervals, ddof=1) if len(nn_intervals) > 1 else np.nan

    diff_nn = np.diff(nn_intervals)
    rmssd = np.sqrt(np.nanmean(diff_nn**2)) if len(diff_nn) > 0 else np.nan

    if len(nn_intervals) > 1:
        elapsed_time = np.cumsum(nn_intervals)
        frequencies = np.linspace(0.01, 0.5, 1000)
        angular_frequencies = 2 * np.pi * frequencies
        detrended_nn = nn_intervals - np.nanmean(nn_intervals)
        power_spectrum = lombscargle(elapsed_time, detrended_nn, angular_frequencies, normalize=True)
        hf_band = (frequencies >= 0.15) & (frequencies <= 0.4)
        hf = trapezoid(power_spectrum[hf_band], frequencies[hf_band])
    else:
        hf = np.nan

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
