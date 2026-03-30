import numpy as np


def resample_signal(signal, fs, target_fs):
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return signal, target_fs
    if fs == target_fs:
        return signal, target_fs

    duration = signal.size / fs
    target_samples = max(1, int(round(duration * target_fs)))
    time_original = np.linspace(0, duration, signal.size)
    time_target = np.linspace(0, duration, target_samples)
    return np.interp(time_target, time_original, signal), target_fs