from scipy.signal import butter, filtfilt
import numpy as np
from scipy.signal import welch
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, entropy

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    go = None
    make_subplots = None

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    # Usamos filtfilt para que no haya desfase en la señal
    # w, h = signal.freqz(b, a, worN=8000)
    # frequencies = (w * fs) / (2 * np.pi)
    # plt.figure(figsize=(10, 5))
    # plt.plot(frequencies, 20 * np.log10(abs(h)))
    # plt.xlim(0, highcut + 20)
    # plt.ylim(-40, 5) # Para ver bien la caída
    # plt.title('Respuesta Frecuencial Digital (Bandpass)')
    # plt.xlabel('Frecuencia [Hz]')
    # plt.ylabel('Amplitud [dB]')
    # plt.grid(which='both', axis='both')
    # plt.axvline(lowcut, color='red', linestyle='--', label='Lowcut')
    # plt.axvline(highcut, color='red', linestyle='--', label='Highcut')
    # plt.legend()
    # plt.show()
    y = filtfilt(b, a, data) 
    return y

def plot_EEG(df, columns, fs = 200):
    if go is None or make_subplots is None:
        raise ModuleNotFoundError("plotly is required for plot_EEG")

    fig = make_subplots(rows=len(columns), cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.02,
                subplot_titles=columns)
    limit = int(3000 * fs) 
    x = np.arange(df[0].shape[0]) / fs  # Asumiendo fs=100Hz, ajusta si es diferente
    downsample = 10  # Factor de downsampling para mejorar rendimiento (ajusta según necesidad)
    for i, col in enumerate(columns):
        fig.add_trace(
            go.Scattergl(x=x[:limit:downsample], y=df[i][:limit:downsample], name=col, mode='lines'),
            row=i+1, col=1
        )
    fig.update_layout(
        height=900, 
        title_text="Polisomnografía - Canales EEG",
        showlegend=False,
        template="plotly_white"
    )
    fig.update_xaxes(title_text="Tiempo (segundos)", row=len(columns), col=1)
    fig.show()
      
def plot_EEG_sel(sel, name = "EEG_plot_raw.html"):
    if go is None or make_subplots is None:
        raise ModuleNotFoundError("plotly is required for plot_EEG_sel")

    fig = make_subplots(rows=len(sel), cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.02,
                    subplot_titles=[ch[1].label for ch in sel])

    for i, (idx, sig) in enumerate(sel):
        # Crear eje de tiempo en segundos
        fs = sig.sampling_frequency
        time = np.linspace(0, len(sig.data) / fs, len(sig.data))
        
        # Añadir traza (solo mostramos los primeros 30s por defecto para no saturar el navegador)
        # Puedes quitar el slice [:int(30*fs)] para ver todo, pero cuidado con el rendimiento
        limit = int(3000 * fs) 
        # limit = len(sig.data) if limit > len(sig.data) else limit
        # limit = len(sig.data)
        # downsample = 10  # Factor de downsampling para mejorar rendimiento (ajusta según necesidad)
        fig.add_trace(
            go.Scattergl(x=time[:limit], y=sig.data[:limit], name=sig.label, mode='lines'),
            row=i+1, col=1
        )

    fig.update_layout(
        height=900, 
        title_text="Polisomnografía - Canales EEG",
        showlegend=False,
        template="plotly_white"
    )

    fig.update_xaxes(title_text="Tiempo (segundos)", row=len(sel), col=1)
    fig.write_html(f"graphs/{name}.html")  # Guardar como HTML para visualización interactiva
    # fig.show()

def filtering_and_normalization(sig, sig_fs):
    b, a = signal.butter(4, 0.3, btype='highpass', fs=sig_fs)
    sig_filtered = signal.filtfilt(b, a, sig)
    b, a = signal.butter(4, 35, btype='lowpass', fs=sig_fs)
    sig_filtered = signal.filtfilt(b, a, sig_filtered)
    sig_filtered = normalize(sig_filtered)
    return sig_filtered

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

def remove_impulse_artifacts(sig):
    # Square of second derivative
    aux = np.diff(np.diff(sig)) ** 2
    aux = np.insert(aux, 0, aux[0])
    aux = np.append(aux, aux[-1])

    # Median filter threshold
    wind = 999
    if aux.size < wind:
        wind = aux.size
        if (wind % 2) != 1:
            wind = wind - 1
    mf = signal.medfilt(aux, wind)

    # Find impulses
    margin = 20
    impulses = np.asarray(np.where(aux > mf + 0.005)).ravel()
    for impulse in impulses:
        impulses = np.append(impulses, np.arange(impulse - margin, min(impulse + margin+1, sig.size)))
    impulses = np.sort(impulses)
    impulses = np.unique(impulses)
    impulses = impulses[impulses >= 0]

    # Remove impulses
    output = sig
    output[impulses] = np.nan
    return output

def clean_movement_artifacts(data, fs, threshold_z=10, window_ms=500):
    """
    Identifica y limpia artefactos de gran amplitud.
    
    Args:
        data: Array de la señal.
        fs: Frecuencia de muestreo.
        threshold_z: Umbral de desviaciones estándar para marcar como artefacto.
        window_ms: Tiempo alrededor del artefacto a limpiar para asegurar 
                   que eliminamos la subida y bajada del pico.
    """
    cleaned_data = data.copy()
    
    # 1. Calcular Z-Score de la amplitud
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    
    # 2. Encontrar índices que superan el umbral
    mask = z_scores > threshold_z
    
    # 3. Expandir la máscara (el movimiento suele durar un poco más que el pico)
    padding = int((window_ms / 1000) * fs)
    expanded_mask = np.convolve(mask, np.ones(padding), mode='same') > 0
    
    # 4. Reemplazar artefactos con el valor medio (0 si está centrada)
    cleaned_data[expanded_mask] = 0
    
    artifacts_percentage = (np.sum(expanded_mask) / len(data)) * 100
    print(f"Artefactos eliminados: {artifacts_percentage:.2f}% de la señal.")
    
    return cleaned_data

def adaptive_variance_cleaner(data, fs, win_size_ms=500, alpha=0.1, threshold=3.5):
    """
    Filtro adaptativo que detecta artefactos cuando la varianza local
    excede significativamente la varianza histórica adaptativa.
    
    Args:
        data: Array de la señal (1D).
        fs: Frecuencia de muestreo.
        win_size_ms: Tamaño de la ventana para calcular la varianza local.
        alpha: Factor de adaptación (0 a 1). Cuanto más alto, más rápido olvida el pasado.
        threshold: Multiplicador de la varianza adaptativa para marcar artefacto.
    """
    win_samples = int((win_size_ms / 1000) * fs)
    n_samples = len(data)
    cleaned_data = np.copy(data)
    
    # Inicializamos la varianza adaptativa con la varianza de la primera ventana
    first_win = data[:win_samples]
    adaptive_var = np.var(first_win)
    
    # Para guardar dónde detectamos artefactos
    artifact_mask = np.zeros(n_samples, dtype=bool)

    # Iteramos por ventanas
    for i in range(0, n_samples - win_samples, win_samples):
        current_win_idx = slice(i, i + win_samples)
        current_var = np.var(data[current_win_idx])
        
        # Si la varianza actual es mucho mayor que la adaptativa, es un artefacto
        if current_var > threshold * adaptive_var:
            artifact_mask[current_win_idx] = True
            cleaned_data[current_win_idx] = 0 # O podrías interpolar
            # No actualizamos la varianza adaptativa con un artefacto para no "contaminarla"
        else:
            # Actualización adaptativa (Exponential Moving Average)
            adaptive_var = alpha * current_var + (1 - alpha) * adaptive_var
            
    return cleaned_data, artifact_mask

def create_epochs(data, fs, epoch_duration=30):
    samples_per_epoch = int(fs * epoch_duration)
    num_epochs = len(data) // samples_per_epoch
    
    # Recortamos la señal para que sea divisible exactamente
    data_trimmed = data[:num_epochs * samples_per_epoch]
    
    # Reshape: (Número de épocas, Puntos por época)
    epochs = data_trimmed.reshape(num_epochs, samples_per_epoch)
    return epochs

def extract_band_powers(epochs, fs, win_len = 2):
    features = []
    complexities = []
    # Definición de las bandas
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Sigma': (11, 16),
        'Beta': (12, 30)
    }
    
    for epoch in epochs:
        # Calcular PSD
        freqs, psd = welch(epoch, fs, nperseg=fs*30) # Ventanas de 2 seg para buena resolución
        # Plot de PSD para verificar que las bandas se ven bien (opcional)
        # plt.semilogy(freqs, psd)
        # plt.show()
        epoch_features = {}
        for band_name, (low, high) in bands.items():
            # Encontrar índices de frecuencia para la banda actual
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            # Calcular la potencia media en esa banda
            epoch_features[band_name] = np.mean(psd[idx_band])
        
        features.append(epoch_features)

        diff = np.diff(epoch)
        mobility = np.sqrt(np.var(diff) / np.var(epoch))
        # 2. Complejidad de Hjorth: Qué tan similar es la señal a una onda senoidal
        diff2 = np.diff(diff)
        mobility_diff = np.sqrt(np.var(diff2) / np.var(diff))
        complexity = mobility_diff / mobility if mobility > 0 else 0
        complexities.append({'Hjorth_Mobility': mobility, 'Hjorth_Complexity': complexity})

    return pd.DataFrame(features), pd.DataFrame(complexities)

def get_patient_profile(df_features):
    # 1. Calcular Potencia Total por época
    total_power = df_features.sum(axis=1)
    avg_p = df_features.mean()
    total_avg_p = avg_p.sum()
    
    # 2. Variabilidad (Refleja microdespertares y fragmentación)
    # Coeficiente de Variación (CV = std/mean) para normalizar por amplitud
    variability = df_features.std() / df_features.mean()
    variability.index = ['CV_' + col for col in variability.index]
    
    # 3. Curtosis (Picos súbitos de actividad)
    kurt = df_features.apply(kurtosis)
    kurt.index = ['Kurt_' + col for col in kurt.index]
    
    # 4. Índices de potencia relativa específicos
    rel_delta = avg_p['Delta'] / total_avg_p
    
    # 5. Ratios de enlentecimiento
    tar = avg_p['Theta'] / avg_p['Alpha'] # Theta-Alpha Ratio
    tbr = avg_p['Theta'] / avg_p['Beta']  # Theta-Beta Ratio
    
    # 6. Entropía Espectral (Complejidad del perfil de potencia promedio)
    # Cuanto más baja, más "pobre" es la diversidad de frecuencias del cerebro
    spec_entropy = entropy(df_features)

    # 2. Calcular Potencias Relativas (promedio de toda la noche)
    rel_powers = df_features.div(total_power, axis=0).mean()
    rel_powers.index = ['Rel_' + col for col in rel_powers.index]
    
    # Calculate main frecuencies of oscilation on each band (peak frequency)
    # Esto puede ser un buen indicador de cambios en la arquitectura del sueño
    # peak_freqs = {}
    # for band in ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']:
    #     freqs, psd = welch(df_features[band], fs=1/30, nperseg=25, noverlap = 25 // 2, nfft=1024) # fs=1/30 porque cada punto es un promedio de 30s
    #     idx_peak = np.argmax(psd)
    #     peak_freqs['PeakFreq_' + band] = freqs[idx_peak]

    # 3. Calcular Ratios Críticos
    # Usamos la media de las potencias absolutas para el ratio global
    avg_p = df_features.mean()
    ratios = {
        'Ratio_Theta_Alpha': avg_p['Theta'] / avg_p['Alpha'],
        'Ratio_Slow_Fast': (avg_p['Delta'] + avg_p['Theta']) / (avg_p['Alpha'] + avg_p['Beta']),
        'Sigma_Stability': df_features['Sigma'].std() / df_features['Sigma'].mean(),
        'Spectral_Entropy_delta': spec_entropy[0],
        'Spectral_Entropy_theta': spec_entropy[1],
        'Spectral_Entropy_alpha': spec_entropy[2],
        'Spectral_Entropy_sigma': spec_entropy[3],
        'Spectral_Entropy_beta': spec_entropy[4],
        'Theta_Alpha_Ratio': tar,
        'Theta_Beta_Ratio': tbr,
        'Relative_Delta_Power': rel_delta,
        'kurtosis_Delta': kurt['Kurt_Delta'],
        'kurtosis_Theta': kurt['Kurt_Theta'],
        'kurtosis_Alpha': kurt['Kurt_Alpha'],
        'kurtosis_Sigma': kurt['Kurt_Sigma'],
        'kurtosis_Beta': kurt['Kurt_Beta'],
        'variability_Delta': variability['CV_Delta'],
        'variability_Theta': variability['CV_Theta'],
        'variability_Alpha': variability['CV_Alpha'],
        'variability_Sigma': variability['CV_Sigma'],
        'variability_Beta': variability['CV_Beta'],

    }
    
    # Combinar todo en una sola fila
    profile = pd.concat([rel_powers, pd.Series(ratios)])
    return profile