"""Respiratory feature helpers used by the active submission pipeline."""

import numpy as np
import pandas as pd

from .peakedness import peakednessCost


def peakedness_application(data, stage, plotflag=False, subjet=1):
    fs = 25
    setup = {
        "K": 5,
        "DT": 5,
        "Ts": 60,
        "Tm": 20,
        "Omega_r": np.array([5, 25]) / 60,
        "plotflag": plotflag,
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
        subjet=subjet,
    )
    return hat_br, sk_br, t_aver, used


def ODI_application(data, fs, plotflag=True, subjet=1):
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

    if plotflag:
        try:
            from importlib import import_module

            plt = import_module('matplotlib.pyplot')
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("matplotlib is required when plotflag=True") from exc

        times = np.arange(len(sp)) / fs / 60.0
        plt.figure(figsize=(10, 4))
        plt.plot(times, sp.values, label='SpO2')
        plt.plot(times, baseline.values, label='Baseline (60s med)')
        for start, end in events:
            plt.axvspan(start / fs / 60.0, end / fs / 60.0, color='red', alpha=0.3)
        plt.xlabel('Tiempo (min)')
        plt.ylabel('SpO2 (%)')
        plt.title(f'Sujeto {subjet} - ODI detectado: {odi_mean:.2f} eventos/h')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return odi_mean, odi_deepness