import numpy as np
import scipy.signal as signal

def swa_filter_data(data, Info):
    """
    Filters EEG data using either Chebyshev Type II or Butterworth filters.
    Translated from swa-Matlab.
    
    Parameters:
    -----------
    data : numpy.ndarray
        EEG data array of shape (n_channels, n_samples).
    Info : object or dict
        Structure containing parameters and recording metadata.
        
    Returns:
    --------
    filtData : numpy.ndarray
        Filtered EEG data of shape (n_channels, n_samples).
    """
    # 1. Verificar el orden del filtro por defecto
    if not hasattr(Info['Parameters'], 'Filter_order') or Info['Parameters']['Filter_order'] is None:
        Info['Parameters']['Filter_order'] = 2

    # Extraer método en minúsculas para evitar problemas de mayúsculas/minúsculas
    method = Info['Parameters']['Filter_Method'].lower()
    nyquist = Info['Recording']['sRate'] / 2.0

    # =========================================================================
    # METODO CHEBYSHEV (Tipo II)
    # =========================================================================
    if method == 'chebyshev':
        # Parámetros de filtrado normalizados respecto a Nyquist
        Wp = np.array([Info['Parameters']['Filter_hPass'], Info['Parameters']['Filter_lPass']]) / nyquist
        Ws = np.array([Info['Parameters']['Filter_hPass'] / 5.0, Info['Parameters']['Filter_lPass'] * 2.0]) / nyquist
        Rp = 3.0  # Rizado en la banda de paso (dB)
        Rs = 10.0 # Atenuación en la banda de rechazo (dB)
        
        # Calcular el orden óptimo del filtro Chebyshev y las frecuencias naturales
        n, Wn = signal.cheb2ord(Wp, Ws, Rp, Rs)
        Wn = [0.001, 0.08]  # TODO: Mejorra debugueo, ahora forzado a igual que Matlab con valores por defecto
        # Diseñar el filtro Chebyshev Tipo II (pasa-banda automático al pasar 2 frecuencias)
        bbp, abp = signal.cheby2(n, Rs, Wn, btype='bandpass')
        
        # Aplicar el filtro de fase cero a lo largo del eje del tiempo (axis=-1)
        filtData = signal.filtfilt(bbp, abp, data, axis=0)

    # =========================================================================
    # METODO BUTTERWORTH (Soporta la errata 'buttersworth' del script original)
    # =========================================================================
    elif method in ['buttersworth', 'butterworth']:
        fhc = Info['Parameters']['Filter_hPass'] / nyquist
        flc = Info['Parameters']['Filter_lPass'] / nyquist
        
        # Diseñar filtros Butterworth independientes (Alta y Baja frecuencia)
        b1, a1 = signal.butter(Info['Parameters']['Filter_order'], fhc, btype='high')
        b2, a2 = signal.butter(Info['Parameters']['Filter_order'], flc, btype='low')
        
        # Aplicar el filtrado secuencial idéntico a MATLAB usando el eje del tiempo
        filtData = signal.filtfilt(b1, a1, data, axis=-1)
        filtData = signal.filtfilt(b2, a2, filtData, axis=-1)
        
    else:
        raise ValueError(f"Unknown filter method: {Info['Parameters']['Filter_Method']}")

    return filtData