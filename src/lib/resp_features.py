"""Respiratory feature helpers used by the active submission pipeline."""

import numpy as np
import pandas as pd

from .resp_peakedness import peakednessCost


def peakedness_application(data, stage, subject_id=1):
    fs = 25
    setup = {
        "K": 5,
        "DT": 5,
        "Ts": 60,
        "Tm": 20,
        "Omega_r": np.array([5, 25]) / 60,
        "Nfft": np.power(2, 13),
    }
    time_axis = np.arange(0, data.shape[0] / fs, 1 / fs)

    if time_axis.shape[0] != data.shape[0]:
        time_axis = np.arange(0, data.shape[0] / fs, 1 / fs)[:data.shape[0]]

    hat_br, sk_br, t_aver, used = peakednessCost(
        data,
        time_axis,
        fs,
        setup,
        title=stage,
        storeGraph=False,
        subjet=subject_id,
    )
    return hat_br, sk_br, t_aver, used


def odi_application(data, fs):
    """Compute oxygen desaturation event rate and average event depth."""
    sp = pd.Series(data)
    if len(sp) == 0 or fs <= 0:
        return 0.0, 0.0

    window = max(1, int(fs * 60))
    baseline = sp.rolling(window, min_periods=1, center=True).median()

    diff = baseline - sp
    mask = diff >= 3

    events = []
    in_event = False
    prev_idx = 0
    for idx, flag in mask.items():
        if flag and not in_event:
            start = idx
            in_event = True
        elif not flag and in_event:
            events.append((start, prev_idx))
            in_event = False
        prev_idx = idx
    if in_event:
        events.append((start, prev_idx))

    duration_hours = len(sp) / fs / 3600.0
    odi_mean = len(events) / duration_hours if duration_hours > 0 else 0.0

    magnitudes = []
    for start, end in events:
        magnitudes.append(diff.loc[start:end].max())
    odi_deepness = np.mean(magnitudes) if magnitudes else 0.0

    return odi_mean, odi_deepness
