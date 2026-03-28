import pyedflib
from .lib.ECG_processing import ECGprocessing
import pandas as pd
def openECG(physiological_data_file, patient_id):

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
    return all_patients_ECGresults
