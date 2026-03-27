import pandas as pd
import numpy as np
from .peakedness import peakednessCost
from scipy.interpolate import interp1d
from scipy.stats import kruskal
from scipy.signal import resample, detrend
import scipy.fft as fft
from scipy.signal import butter, filtfilt

try:
    import plotly.graph_objs as go
except ModuleNotFoundError:
    go = None

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

def plot_resp(Data, subjet = 1,  DownPrinting = 2):
    """
    Plot resp data using Plotly.
    """
    if go is None:
        raise ModuleNotFoundError("plotly is required for plot_resp")

    if type(Data) == dict:
        Data = pd.DataFrame(Data[str(subjet)])
        Data = Data.iloc[::DownPrinting, :]
        end = -1
    elif type(Data) == type(pd.DataFrame()):
        Data = Data[Data['Subjet'] == str(subjet)]
        Data = Data.iloc[::DownPrinting, :]
        end = -2

    # Data.reset_index(drop=True, inplace=True)
    print(len(Data.columns))
    fig = go.Figure()
    for c in Data.columns[:end]:
        fig.add_trace(go.Line(x=Data.Time, y=Data[c], name = c))
    fig.update_layout(title_text='EDA Data', title_x=0.5)

    fig.show()

def peakedness_application(Data, stage, plotflag = False, subjet = 1):
    # print("Compute BR")
    fs = 25
    Setup = {}
    Setup["K"] = 5
    Setup["DT"] = 5
    Setup["Ts"] = 60 #interval length of Welch periodograms (s)
    Setup["Tm"] = 20 #interval length of subintervals for Welch periodograms (s)
    # Setup["d"] = 0.1 #interval length of subintervals for Welch periodograms (s)
    Setup["Omega_r"] = np.array([5, 25])/60 #respiratory rate range in Hz
    Setup["plotflag"] = plotflag
    Setup["Nfft"] = np.power(2,13)
    tsBR = np.arange(0,Data.shape[0]/fs,1/fs)

    if tsBR.shape[0] != Data.shape[0]:
        # print(f"tsBR.shape[0]: {tsBR.shape[0]}, Data.shape[0]: {Data.shape[0]}")
        tsBR = np.arange(0,Data.shape[0]/fs,1/fs)[:Data.shape[0]]

    hat_Br, Sk_Br, t_aver, used = peakednessCost(Data, tsBR, fs, Setup, title = stage, storeGraph = False, subjet = subjet)
    # print(f"hat_Br: {hat_Br}, Sk_Br: {Sk_Br}, bar_Br: {bar_Br}, t_aver_Br: {t_aver_Br}, f_Br: {f_Br}, used_Br: {used_Br}")
        
    # print(hat_Br)       
    return hat_Br, Sk_Br, t_aver, used

def ODI_application(data, fs, plotflag=True, subjet=1):
    """Detecta desaturaciones de más del 3 % en la señal de saturación de
    oxígeno (SpO2) y devuelve estadísticas básicas de los eventos.

    El índice de desaturación de oxígeno (ODI) se define como el número de
    episodios en los que la saturación cae al menos un 3 % respecto a una
    línea de base móvil, normalizado por hora de grabación. Aquí se calcula
    una línea base mediante la mediana móvil de 60 segundos y se agrupan
    los índices consecutivos que cumplen el criterio en eventos únicos.

    Args:
        data (array-like): valores de SpO2 (0‑100).
        fs (float): frecuencia de muestreo en Hz.
        plotflag (bool): si True, dibuja la señal y marca los eventos.
        subjet (int): identificador de sujeto (utilizado en títulos de gráficas).

    Returns:
        tuple:
            * odi_mean (float): número de desaturaciones normalizado por hora.
            * odi_std (float): desviación estándar de las magnitudes de caída
              entre eventos (en porcentaje).
    """
    # convertir a serie para comodidad
    sp = pd.Series(data)
    if len(sp) == 0 or fs <= 0:
        return 0.0, 0.0

    # base móvil de 60 segundos (median para ser robusto). ventana en muestras
    window = int(fs * 60)
    if window < 1:
        window = 1
    baseline = sp.rolling(window, min_periods=1, center=True).median()

    # diferencia de base menos señal; buscamos caídas >=3
    diff = baseline - sp
    mask = diff >= 3

    # juntar índices contiguos en eventos
    events = []  # lista de (start_idx, end_idx)
    in_event = False
    for idx, flag in mask.items():
        if flag and not in_event:
            start = idx
            in_event = True
        elif not flag and in_event:
            end = prev_idx
            events.append((start, end))
            in_event = False
        prev_idx = idx
    if in_event:
        events.append((start, prev_idx))

    num_events = len(events)
    duration_hours = len(sp) / fs / 3600.0
    odi_mean = num_events / duration_hours if duration_hours > 0 else 0.0

    # calcular magnitudes de caída en cada evento (tomando el valor más bajo)
    magnitudes = []
    for start, end in events:
        mag = diff.loc[start:end].max()
        magnitudes.append(mag)
    odi_deepness = np.mean(magnitudes) if magnitudes else 0.0

    if plotflag:
        if plt is None:
            raise ModuleNotFoundError("matplotlib is required when plotflag=True")
        times = np.arange(len(sp)) / fs / 60.0  # minutos
        plt.figure(figsize=(10, 4))
        plt.plot(times, sp.values, label='SpO2')
        plt.plot(times, baseline.values, label='Baseline (60s med)')
        for (start, end) in events:
            t0 = start / fs / 60.0
            t1 = end / fs / 60.0
            plt.axvspan(t0, t1, color='red', alpha=0.3)
        plt.xlabel('Tiempo (min)')
        plt.ylabel('SpO2 (%)')
        plt.title(f'Sujeto {subjet} - ODI detectado: {odi_mean:.2f} eventos/h')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return odi_mean, odi_deepness

# Butterworth low-pass filter
def lowpass_filter(signal, fs, cutoff=2.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def Metrics_per_segment(Data):
    """
    Compute peakedness per segment. 
    """

    # Results = pd.DataFrame(columns=['Subject', 'Stage', 'Peakedness', 'Slope', 'Intercept', 'Relative Peak', 'Bocanada', 'Contraction', "TidalVolume", "Complexity", "Mobility", "Activity"])
    Results = pd.DataFrame()

    for subjet in Data['Subjet'].unique():
        sel_sujeto = Data[Data['Subjet'] == subjet]
        sel_sujeto_ref = sel_sujeto.iloc[:,:-2]
        Sol_subject = []
        Sol_interSubject = []
        for secc in sel_sujeto_ref.columns:
            if secc == 'Time':
                continue
            else:
                section = sel_sujeto[secc].values
                section = section[~np.isnan(section)]

                hat_Br, Sk_Br, t_aver, _ = peakedness_application(section, stage=secc, plotflag = False, subjet= subjet)
                # print(f"Subjet: {subjet}, section: {secc} hat_Br: {hat_Br}, Sk_Br: {Sk_Br}")
                
                # Ajuste lineal
                coef = np.polyfit(t_aver, hat_Br, 1)  # Grado 1 = línea recta
                pendiente, interseccion = coef
                # print(f"Pendiente: {pendiente:.6f}, Intersección: {interseccion:.6f}")

                #Picudez relativa
                rel_peak_list = []
                for ti in range(len(t_aver)):
                    f_max = np.argmax(Sk_Br[:,ti])
                    rel_peak_list.append(np.sum(Sk_Br[f_max-1:f_max+1,ti]) / np.sum(Sk_Br[:,ti]))
                real_peak = np.mean(rel_peak_list)
                
                # Derivada
                diff = np.diff(section)
                bocanada = max(np.percentile(diff, 90), np.abs(np.percentile(diff, 10)))

                Contraction = np.percentile(np.abs(diff), 10)

                #Tidal Volume 
                TidalVolume = max(np.percentile(section, 99), np.abs(np.percentile(section, 1)))

                # Calculate derivatives
                dx = np.diff(section)
                ddx = np.diff(dx)

                # Calculate variance and its derivatives
                x_var = np.var(section)  # = activity
                dx_var = np.var(dx)
                ddx_var = np.var(ddx)

                # Mobility and complexity
                mobility = np.sqrt(dx_var / x_var)
                complexity = np.sqrt(ddx_var / dx_var) / mobility


                filtered_signal = lowpass_filter(section, 100, cutoff=2.0, order=4)
                segment4Hz = resample(filtered_signal, int(filtered_signal.size/100*4))  # Resample to 4Hz

                fft_signal = fft.fft(detrend(segment4Hz), n=2**12)
                power = np.abs(fft_signal)**2
                freqs = fft.fftfreq(2**12, d = 1/4)
                max_freq_index = np.argmax(power)
                max_freq = freqs[max_freq_index]
                power_at_max_freq = power[max_freq_index-51:max_freq_index+51].sum()
                power_ratio = power_at_max_freq / np.sum(power[:len(power)//2])

                Sol = [subjet, secc[:secc.find('_')], np.mean(hat_Br), pendiente, interseccion, real_peak, bocanada, Contraction, TidalVolume, complexity, mobility, x_var, max_freq, power_ratio] 
                
                Sol_subject.append(Sol)

        
        Sol_subject = pd.DataFrame(Sol_subject, columns=['Subject', 'Stage', 'Peakedness', 'Slope', 
                                                         'Intercept', 'Relative Peak', 'Bocanada', 
                                                         "Contraction","TidalVolume", "Complexity", 
                                                         "Mobility", "Activity", "Max_freq", "Power_ratio"])    

        peakmean = Sol_subject['Peakedness'].mean()
        peakmin = Sol_subject['Peakedness'].min()
        peakmax = Sol_subject['Peakedness'].max()

        slopemean = Sol_subject['Slope'].mean()
        slopemin = Sol_subject['Slope'].min()
        slopemax = Sol_subject['Slope'].max()

        Rel_peak_mean = Sol_subject['Relative Peak'].mean()

        Bocanada_max = Sol_subject['Bocanada'].max()
        Contraction_max = Sol_subject['Contraction'].max()
        TidalVolume_max = Sol_subject['TidalVolume'].max()

        Rel_metrics = ['Subject', 'Stage',"Peakmean", "Peakmin", "Peakmax", "Slopemean", "Slopemin", "Slopemax", "Rel_peak_mean", "Bocanada_max", "Contraction_max", "TidalVolume_max"]
        # Rel_metrics = ["Peakmean", "Peakmin", "Peakmax", "Slopemean", "Slopemin", "Slopemax", "Rel_peak_mean", "Bocanada_max", "Contraction_max", "TidalVolume_max"]

        Sol_interSubject_DF = pd.DataFrame(Sol_interSubject, columns=Rel_metrics)
        Sol_interSubject_DF = pd.DataFrame(Sol_interSubject, columns=Rel_metrics[2:])
        for i in Sol_subject.index:
            # Sol_interSubject_DF.at[i,Rel_metrics[0]] = Sol_subject.iloc[i,0]
            # Sol_interSubject_DF.at[i,Rel_metrics[1]] = Sol_subject.at[i,'Stage']
            Sol_interSubject_DF.at[i,Rel_metrics[2]] = Sol_subject.at[i,'Peakedness']/peakmean
            Sol_interSubject_DF.at[i,Rel_metrics[3]] = Sol_subject.at[i,'Peakedness']/peakmin   
            Sol_interSubject_DF.at[i,Rel_metrics[4]] = Sol_subject.at[i,'Peakedness']/peakmax
            Sol_interSubject_DF.at[i,Rel_metrics[5]] = Sol_subject.at[i,'Slope']/slopemean 
            Sol_interSubject_DF.at[i,Rel_metrics[6]] = Sol_subject.at[i,'Slope']/slopemin
            Sol_interSubject_DF.at[i,Rel_metrics[7]] = Sol_subject.at[i,'Slope']/slopemax  
            Sol_interSubject_DF.at[i,Rel_metrics[8]] = Sol_subject.at[i,'Relative Peak']/Rel_peak_mean
            Sol_interSubject_DF.at[i,Rel_metrics[9]] = Sol_subject.at[i,'Bocanada']/Bocanada_max  
            Sol_interSubject_DF.at[i,Rel_metrics[10]] = Sol_subject.at[i,"Contraction"]/Contraction_max
            Sol_interSubject_DF.at[i,Rel_metrics[11]] = Sol_subject.at[i,"TidalVolume"]/TidalVolume_max

        Sol = pd.concat([Sol_subject, Sol_interSubject_DF], axis=1)
        Results = pd.concat([Results, Sol], ignore_index=True)

        

    return Results

def Significance_tests(RespData):
    """
    Compute significance tests for the features.
    """
    results = {}
    for metrica in RespData.columns[2:]:
        # print(f"Realizando prueba de Kruskal-Wallis para la métrica: {metrica}")
        # Realizar la prueba de Kruskal-Wallis
        estadistico, p_valor = kruskal(
            np.array(RespData[RespData.Stage == "Baseline"][metrica].reset_index(drop=True)),
            np.array(RespData[RespData.Stage == "LOW"][metrica].reset_index(drop=True)),
            np.array(RespData[RespData.Stage == "HIGH"][metrica].reset_index(drop=True)),
            np.array(RespData[RespData.Stage == "REST"][metrica].reset_index(drop=True))
        )

        # Imprimir resultados
        
        # print(f"Estadístico de Kruskal-Wallis: {estadistico}")
        print(f"Metrica: "+metrica+" tiene un valor p: {p_valor}")
        results[metrica] = p_valor
        # if p_valor < 0.05:
        #     print("Se rechaza la hipótesis nula: hay diferencias significativas entre los grupos.")
        # else:
        #     print("No se rechaza la hipótesis nula: no hay diferencias significativas entre los grupos.")

    results = pd.DataFrame.from_dict(results, orient='index', columns=['p_value'])
    results = results.reset_index()
    results.to_excel('./Graphs/kruskal_results.xlsx', index=False)

    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for Significance_tests plotting")

    plt.plot(results['index'], results['p_value'])
    plt.axhline(y=0.05, color='r', linestyle='--') 
    plt.xlabel('Métrica')
    plt.ylabel('Valor p')
    plt.title('Resultados de la prueba de Kruskal-Wallis')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('./Graphs/kruskal_results.png')
    plt.show()