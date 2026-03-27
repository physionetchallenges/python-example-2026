from .lib import Resp_features
import sys 
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def processResp(physiological_data, physiological_fs, csv_path):

    channels = pd.read_csv(csv_path)
    selectResp = channels[channels['Category'].isin(['resp'])]

    resultados = {}
    UsedFlow = 0
    UsedChest = 0 
    UsedAbdomen = 0
    UsedSpO2 = 0
    UsedNasal = 0
    UsedCepap = 0

    data = []
    original_labels = list(physiological_data.keys())

    for label in original_labels:
        fs = physiological_fs[label]
        sig = physiological_data[label]
        if fs != 25:
            duration = len(sig) / fs
            time_original = np.linspace(0, duration, len(sig))
            num_samples_target = int(duration * 25 )
            time_target = np.linspace(0, duration, num_samples_target)
            data = np.interp(time_target, time_original, sig)
            fs = 25  # Update fs to the target sampling frequency after resampling
        else:
            data = sig

        # Check nan in sig.data
        if np.isnan(sig).any():
            print(f"Warning: NaN values found in signal data for {label}. Filling NaNs with zeros.")
            data = np.nan_to_num(data)

        name = ""
        if label.lower() not in selectResp['Channel_Names'][34].lower():
            d = Resp_features.peakedness_application(data, stage=label, plotflag = False, subjet =label)
            if label.lower() in selectResp['Channel_Names'][28].lower():
                name = "Chest"
                # EFFORT RESPIRATORY Chest
            elif label.lower() in selectResp['Channel_Names'][29].lower():
                # EFFORT RESPIRATORY Abdomen
                name = "Abdomen"
            elif label.lower() in selectResp['Channel_Names'][30].lower():
                # RESPIRATORY NASAL
                name = "Nasal"
            elif label.lower() in selectResp['Channel_Names'][31].lower():
                # RESPIRATORY FLOW
                name = "Flow"
            elif label.lower() in selectResp['Channel_Names'][32].lower():
                # CEPAP
                if np.all(data == 0) or np.std(data) < 5:
                    print(f"Warning: All values in the signal data for {label} are zero. Skipping feature extraction for this channel.")
                else:
                    name = ""
            elif label.lower() in selectResp['Channel_Names'][33].lower():
                # CEPAP
                name = ""
            
            if name != "":
                DSinNan = d[0][~np.isnan(d[0])]  # Eliminar NaN antes de calcular min y max
                if len(DSinNan) != 0:
                    maximo = DSinNan.max()
                    minimo = DSinNan.min()
                    media = np.mean(DSinNan)
                    mediana = np.median(DSinNan)
                    std = DSinNan.std()
                    write = False
                    if name == "Nasal" and UsedNasal< d[-1]:
                        UsedNasal = d[-1]
                        write = True
                    elif name == "Chest" and UsedChest< d[-1]:
                        UsedChest = d[-1]
                        write = True
                    elif name == "Abdomen" and UsedAbdomen< d[-1]:
                        UsedAbdomen = d[-1]
                        write = True
                    elif name == "Flow" and UsedFlow< d[-1]:
                        UsedFlow = d[-1]
                        write = True
                    elif name == "SpO2" and UsedSpO2< d[-1]:
                        UsedSpO2 = d[-1]
                        write = True
                    elif name == "CEPAP" and UsedCepap < d[-1]:
                        UsedCepap = d[-1]
                        write = True
                    if write:
                        resultados.update({
                            name+"_Peakedness_Max": maximo,
                            name+"_Peakedness_Min": minimo,
                            name+"_Peakedness_Mean": media,
                            name+"_Peakedness_Median": mediana,
                            name+"_Peakedness_Std": std
                        })  
        
        elif label.lower() in selectResp['Channel_Names'][34].lower():
            #O2 SATURATION   
            if np.max(data) < 2:
                data = np.round((data/1.055)*100)

            lim = 0.7
            # Quitar los valores por debajo de lim y sus 10 valores anteriores y posteriores para quedarnos solo con los eventos de desaturación
            dataReal = data.copy()
            for i in range(len(data)):
                if data[i] < lim:
                    start = int(max(0, i-fs*2))
                    end = int(min(len(data), i+fs*2))
                    dataReal[start:end] = np.nan  # Marcar los valores por debajo del límite y sus alrededores como NaN

            CET90 = dataReal[dataReal < 90]
            # CET90SinNan = CET90[~np.isnan(CET90)]  # Eliminar NaN antes de calcular min y max
            CET90 = len(CET90)/len(data)
            dataRealSinNan = dataReal[~np.isnan(dataReal)]  # Eliminar NaN antes de calcular min y max
            if len(dataRealSinNan)>0:
                maximo = dataRealSinNan.max()
                minimo = dataRealSinNan.min()
                std = dataRealSinNan.std()
                media = dataRealSinNan.mean()
                ODI_mean, ODI_deepness = Resp_features.ODI_application(dataReal, fs, plotflag=False, subjet=1)

                resultados.update({"SpO2_Max": maximo,
                    "SpO2_Min": minimo,
                    "SpO2_Mean": media,
                    "SpO2_Std": std,
                    "CET90": CET90,
                    "ODI_Mean": ODI_mean,
                    "ODI_deepness": ODI_deepness,
                })
    
    return np.array(resultados)
