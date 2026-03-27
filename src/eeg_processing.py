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
import lib.EEG_functions as EEG_functions


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def processEEG(physiological_data, physiological_fs, csv_path):

    channels = pd.read_csv(csv_path)
    selectEEG = channels[channels['Category'].isin(['eeg'])]

    for label in original_labels:
        fs = physiological_fs[label]

        data = []
        original_labels = list(physiological_data.keys())

        # Listar canales para identificar los de interés (ej: C3-M2, O1-M2)
        HayEEG = False
        for i, label in enumerate(original_labels):
            for index in selectEEG.index:
                if label.lower() in selectEEG['Channel_Names'][index].lower():
                    print(f"Canal seleccionado: {label}")
                    labels.append(label)
                    HayEEG = True
                    break
    
    results = []
    labels2 = []
    if HayEEG:
        Bipolar = pd.DataFrame()
        if all(label in labels for label in ["F3", "F4", "M1", "M2"]):
            Bipolar['F3-M2'] = physiological_data["F3"] - physiological_data["M2"]
            Bipolar['F4-M1'] = physiological_data["F4"] - physiological_data["M1"]
            labels2.append('F3-M2')
            labels2.append('F4-M1')
        if all(label in labels for label in ["C3", "C4", "M1", "M2"]):
            Bipolar['C3-M2'] = physiological_data["C3"] - physiological_data["M2"]
            Bipolar['C4-M1'] = physiological_data["C4"] - physiological_data["M1"]
            labels2.append('C3-M2')
            labels2.append('C4-M1')
        if all(label in labels for label in ["O2", "O1", "M1", "M2"]):
            Bipolar['O2-M2'] = physiological_data["O1"] - physiological_data["M2"]
            Bipolar['O1-M1'] = physiological_data["O2"] - physiological_data["M1"]
            labels2.append('O1-M1')
            labels2.append('O2-M2')
        # print(f"Archivo {file} tiene ECG, RESP y EEG. Se procesará con canales bipolares.")
        
        if not Bipolar.empty:
            labels = []
            for col in Bipolar.columns:
                # print(f"Archivo: {file}, Canal: {col}, Frecuencia de muestreo: {sig.sampling_frequency} Hz, Duración: {len(Bipolar[col])/sig.sampling_frequency:.2f} segundos")
                fs = physiological_data["M2"].sampling_frequency  # Asumimos que todos los canales tienen la misma frecuencia de muestreo
                fil = EEG_functions.butter_bandpass_filter(Bipolar[col], lowcut=0.3, highcut=35, fs=fs, order=4)
                norm = (fil-np.mean(fil))/np.std(fil)
                
                data.append(norm)  # Restar la media para centrar la señal
                labels.append(col)
            # columns = Bipolar.columns.tolist()
        else:
            labels = []
            for l in labels:
                # print(f"Archivo: {file}, Canal: {sig.label}, Frecuencia de muestreo: {sig.sampling_frequency} Hz, Duración: {len(sig.data)/sig.sampling_frequency:.2f} segundos")
                fs = physiological_fs[l]
                fil = EEG_functions.butter_bandpass_filter(physiological_data[l], lowcut=0.3, highcut=35, fs=fs, order=4)
                norm = (fil-np.mean(fil))/np.std(fil)
                labels.append(l)
                data.append(norm)  # Restar la media para centrar la señal

            # columns = [selEEG[i][1].label for i in range(len(selEEG))]
        
        
        for i, elec in enumerate(labels):
            epoch_length = 30  # Duración de cada época en segundos
            if Bipolar.empty:
                fs = physiological_fs[l]
            else:
                fs = physiological_fs['M1']

            if fs != 200:
                # print(f"Warning: Sampling frequency for channel {elec} in file {file} is {fs} Hz, expected 200 Hz. Check the data.")
                duration = len(data[i]) / fs
                time_original = np.linspace(0, duration, len(data[i]))
                
                num_samples_target = int(duration * 200 )
                time_target = np.linspace(0, duration, num_samples_target)
                data[i] = np.interp(time_target, time_original, data[i])
                fs = 200  # Update fs to the target sampling frequency after resampling
            
            epochs = EEG_functions.create_epochs(data[i], fs, epoch_duration=epoch_length)

            band_powers, complexities = EEG_functions.extract_band_powers(epochs, fs, win_len=15)
            band_powers = band_powers.iloc[60:]  # Eliminar las primeras 60 épocas (30 min) para evitar el tiempo despierto al inicio de la grabación


            # Ejecución
            patient_summar = EEG_functions.get_patient_profile(band_powers)

            d = complexities.iloc[:].std().to_dict() 
            results.append({
                'Channel': elec,
                **d,
                **patient_summar
            })
    df_results = pd.DataFrame(results)
    return df_results