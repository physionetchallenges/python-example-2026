"""EEG_processing.py

Este módulo contiene funciones para procesar datos EEG de los
hospitales incluidos en el desafío CincChallenge 2026. La principal
función definida es `MetricasHospitlal`, que recorre los archivos EDF
correspondientes a un hospital concreto, extrae las señales EEG,
las filtra, normaliza, crea épocas y calcula potencias de banda y
complejidades. Los resultados se guardan en un CSV resumen por
hospital.

Características principales:

- Soporta datos tanto del conjunto de entrenamiento como del
  conjunto suplementario.
- Selección automática de canales EEG a partir de la tabla
  `notebooks/channel_table.csv`.
- Creación de canales bipolares si están disponibles.
- Filtrado de banda 0.3-35 Hz y normalización de la señal.
- Re-muestreo a 200 Hz si fuese necesario.
- Cálculo de potencias de banda y complejidades usando
  funciones auxiliares (`lib/EEG_functions.py`).
- Exportación de resultados en `results_summaryEEG_{hospital}.csv`.

Uso típico:

>>> from src.scripts.EEG_processing import MetricasHospitlal
>>> MetricasHospitlal('I0002')

El módulo depende de `numpy`, `pandas`, `matplotlib`, `plotly` y de
las utilidades definidas en `lib/helper_code` y `lib/EEG_functions`.
"""
import sys
import os
import pandas as pd
import numpy as np
import helper_code as helper_code
from .lib import EEG_functions


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

EEG_FEATURE_NAMES = [
    'EEG_Channel_Count',
    'EEG_Rel_Delta',
    'EEG_Rel_Theta',
    'EEG_Rel_Alpha',
    'EEG_Rel_Sigma',
    'EEG_Rel_Beta',
    'EEG_Theta_Alpha_Ratio',
    'EEG_Hjorth_Complexity',
]
EEG_FEATURE_LENGTH = len(EEG_FEATURE_NAMES)
EEG_ALIASES_CACHE = {}


def _normalize_label(text):
    normalized = ''.join(ch if ch.isalnum() else ' ' for ch in str(text).lower())
    return ' '.join(normalized.split())


def _split_aliases(raw_aliases):
    return {_normalize_label(alias) for alias in str(raw_aliases).split(';') if alias}


def _build_eeg_aliases(channels):
    eeg_rows = channels[channels['Category'].eq('eeg')]
    aliases = set()
    for _, row in eeg_rows.iterrows():
        aliases.update(_split_aliases(row['Channel_Names']))
    return aliases


def _get_eeg_aliases(csv_path):
    normalized_csv_path = os.path.abspath(csv_path)
    eeg_aliases = EEG_ALIASES_CACHE.get(normalized_csv_path)
    if eeg_aliases is None:
        channels = pd.read_csv(normalized_csv_path)
        eeg_aliases = _build_eeg_aliases(channels)
        EEG_ALIASES_CACHE[normalized_csv_path] = eeg_aliases
    return eeg_aliases


def _resample_signal(signal, fs, target_fs):
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


def _extract_channel_metrics(signal, fs):
    signal = np.nan_to_num(np.asarray(signal, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if signal.size < max(int(fs * 30), 2):
        return None

    if fs != 200:
        signal, fs = _resample_signal(signal, fs, 200)

    filtered = EEG_functions.butter_bandpass_filter(signal, lowcut=0.3, highcut=35, fs=fs, order=4)
    signal_std = np.std(filtered)
    if signal_std == 0 or not np.isfinite(signal_std):
        return None

    normalized = (filtered - np.mean(filtered)) / signal_std
    epochs = EEG_functions.create_epochs(normalized, fs, epoch_duration=30)
    if epochs.size == 0:
        return None

    band_powers, complexities = EEG_functions.extract_band_powers(epochs, fs, win_len=15)
    if len(band_powers) > 60:
        band_powers = band_powers.iloc[60:]
        complexities = complexities.iloc[60:]
    if band_powers.empty:
        return None

    total_power = band_powers.sum(axis=1).replace(0, np.nan)
    relative_powers = band_powers.div(total_power, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0).mean()
    alpha_power = float(relative_powers.get('Alpha', 0.0))
    theta_power = float(relative_powers.get('Theta', 0.0))
    theta_alpha_ratio = theta_power / alpha_power if alpha_power > 0 else 0.0

    complexity_mean = float(
        complexities['Hjorth_Complexity'].replace([np.inf, -np.inf], np.nan).fillna(0.0).mean()
    ) if 'Hjorth_Complexity' in complexities else 0.0

    return np.array([
        float(relative_powers.get('Delta', 0.0)),
        theta_power,
        alpha_power,
        float(relative_powers.get('Sigma', 0.0)),
        float(relative_powers.get('Beta', 0.0)),
        float(theta_alpha_ratio),
        complexity_mean,
    ], dtype=np.float32)


def processEEG(physiological_data, physiological_fs, csv_path):
    eeg_aliases = _get_eeg_aliases(csv_path)
    channel_metrics = []

    for label, signal in physiological_data.items():
        if label not in physiological_fs:
            continue
        if _normalize_label(label) not in eeg_aliases:
            continue

        metrics = _extract_channel_metrics(signal, physiological_fs[label])
        if metrics is not None:
            channel_metrics.append(metrics)

    if not channel_metrics:
        return np.zeros(EEG_FEATURE_LENGTH, dtype=np.float32)

    stacked = np.vstack(channel_metrics)
    aggregated = np.mean(stacked, axis=0)
    return np.hstack([np.array([len(channel_metrics)], dtype=np.float32), aggregated]).astype(np.float32)