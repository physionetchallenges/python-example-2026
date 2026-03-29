import pyedflib
from .lib.ECG_processing import ECGprocessing
import pandas as pd
import numpy as np

ECG_KEYWORDS = ['ecg', 'ekg']

ECG_FEATURE_NAMES = [
        "PIP_med",
        "PIP_std",
        "PNNLS_med",
        "PNNLS_std",
        "PNNSS_med",
        "PNNSS_std",
        "AVNN_med",
        "AVNN_std",
        "SDNN_med",
        "SDNN_std",
        "RMSSD_med",
        "RMSSD_std",
        "HF_med",
        "HF_std",
        "ECTOPIC_med",
        "ECTOPIC_std"
]
ECG_FEATURE_LENGTH = len(ECG_FEATURE_NAMES)

def _normalize_label(text):
    return ''.join(ch if ch.isalnum() else ' ' for ch in str(text).lower()).strip()


def _find_ecg_channel(physiological_data):
    for label in physiological_data.keys():
        label_clean = _normalize_label(label)
        if any(keyword in label_clean for keyword in ECG_KEYWORDS):
            return label
    return None


def processECG(physiological_data, physiological_fs, csv_path):
    results = np.zeros(ECG_FEATURE_LENGTH, dtype=np.float32)

    ecg_label = _find_ecg_channel(physiological_data)

    if ecg_label is None:
        return results  # no ECG found

    if ecg_label not in physiological_fs:
        return results

    ecg_signal = np.asarray(physiological_data[ecg_label], dtype=float)
    fs = physiological_fs[ecg_label]

    if ecg_signal.size == 0:
        return results

    try:
        values = ECGprocessing(ecg_signal, fs, ECG_FEATURE_LENGTH)

        if values is None or len(values) == 0:
            return results

        values = values.astype(np.float32)

        if len(values) >= ECG_FEATURE_LENGTH:
            results[:] = values[:ECG_FEATURE_LENGTH]
        else:
            results[:len(values)] = values

    except Exception:
        pass

    return results

""" def openECG(physiological_data_file, patient_id):

    f = pyedflib.EdfReader(physiological_data_file)

    signal_labels = f.getSignalLabels()
    print(signal_labels)

    ecg_keywords = ['ecg', 'ekg']

    idx = None
    for i, label in enumerate(signal_labels):
        label_clean = label.lower().strip()

        # Check if any ECG keyword is inside the label
        if any(keyword in label_clean for keyword in ecg_keywords):
            idx = i
            break  #first ECG channel only

    if idx is None:
        raise ValueError("No ECG channel found")

    print("ECG channel:", signal_labels[idx])

    ecg_signal = f.readSignal(idx)
    fs = f.getSampleFrequency(idx)

    f.close()

    all_results = ECGprocessing(ecg_signal, fs, patient_id)
  
    if all_results is not None:
        all_patients_ECGresults = pd.concat(
            [all_patients_ECGresults, all_results],
            ignore_index=True
        ) 
    return all_patients_ECGresults """
