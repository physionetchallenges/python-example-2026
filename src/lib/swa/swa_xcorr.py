import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def swa_xcorr(refData, shortData, win):

    """
    Analyses the cross-correlation (Pearson r) between the reference channel
    and all subsequent channels using a sliding window without zero padding.
    Translated from swa-Matlab.
    
    Parameters:
    -----------
    refData : numpy.ndarray
        Reference signal row vector of shape (n_samples_ref,) or (1, n_samples_ref).
    shortData : numpy.ndarray
        Target channels matrix of shape (n_channels, n_samples_short).
    win : int
        Window size parameter.
        
    Returns:
    --------
    R : numpy.ndarray
        Correlation matrix of shape (n_channels, win * 2 + 1).
    """
    # Asegurar que refData sea un vector plano (1D) para las operaciones de correlación
    if refData.ndim > 1:
        refData = refData.flatten()
        
    n_channels = refData.ndim
    n_shifts = win * 2 + 1
    try:
        len_ref = len(refData)
    except:
        len_ref = 1

    # Pre-asignar la matriz de resultados (canales x desplazamientos)
    R = np.zeros((n_channels, n_shifts))
    
    # Centralizar la referencia (restar media) para el cálculo de Pearson
    ref_centered = refData - np.mean(refData)
    ref_norm = np.sqrt(np.sum(ref_centered ** 2))
    
    if ref_norm == 0:
        return R # Evitar división por cero si la referencia es plana
    
    # Bucle a lo largo del tiempo (desplazamientos de la ventana deslizante)
    for t in range(n_shifts):
        # Extraer la porción de datos de todos los canales para el lag 't'
        # Equivalente a shortData(:, t : t + size(refData,2) - 1) en MATLAB
        sub_data = shortData[t : t + len_ref]
        
        # Centralizar cada canal en esta ventana temporal específica
        sub_mean = np.mean(sub_data, keepdims=True)
        sub_centered = sub_data - sub_mean
        
        # Calcular desviaciones estándar (denominador de Pearson)
        sub_norm = np.sqrt(np.sum(sub_centered ** 2))
        
        # Producto punto entre la referencia y los canales (numerador de Pearson)
        numerator = np.dot(sub_centered, ref_centered)
        
        # Coeficiente de correlación r de Pearson: n_channels x 1
        # Usamos np.errstate para evitar advertencias si algún canal tiene varianza cero
        with np.errstate(divide='ignore', invalid='ignore'):
            r_val = numerator / (sub_norm * ref_norm)
            
        # Reemplazar posibles NaNs (por canales planos/sin varianza) con 0
        if np.isnan(r_val):
            r_val = 0.0
        
        R[:, t] = r_val
        
    return R

def swa_xcorr2(refData, shortData, win):
    """
    Calculates the sliding Pearson cross-correlation between a reference channel
    and all other channels without zero padding.
    
    Parameters:
    -----------
    refData : numpy.ndarray
        1D array (or 2D with 1 row) of the reference channel data.
    shortData : numpy.ndarray
        2D array (channels x samples) of the data to compare.
    win : int
        Window reach (the loop will execute win * 2 + 1 times).
    """
    # Asegurar que refData sea un vector 1D plano
    refData = np.squeeze(refData)
    
    n_channels = shortData.shape[1]
    n_samples_ref = len(refData)
    n_lags = win * 2 + 1
    
    # Inicializar la matriz de resultados (canales x retrasos)
    R = np.zeros((n_channels, n_lags))
    
    # Precalcular la media y la desviación de la onda de referencia (fijos)
    ref_mean = np.mean(refData)
    ref_dev = refData - ref_mean
    ref_var = np.sum(ref_dev ** 2)
    
    # Bucle solo para los desplazamientos temporales (lags)
    for t in range(n_lags):
        # Extraer la ventana actual para TODOS los canales a la vez
        short_slice = shortData[t : t + n_samples_ref]
        
        # Calcular la media y desviación local de la ventana por cada canal
        short_mean = np.mean(short_slice, keepdims=True)
        short_dev = short_slice - short_mean
        short_var = np.sum(short_dev ** 2)
        
        # Covarianza cruzada (multiplicación matricial elemento a elemento)
        covariance = np.sum(short_dev * ref_dev)
        
        # Calcular el coeficiente r de Pearson (evitando divisiones por cero)
        with np.errstate(divide='ignore', invalid='ignore'):
            R[:, t] = covariance / np.sqrt(ref_var * short_var)
            
    # Reemplazar posibles NaNs (por canales planos sin variación) con cero
    return np.nan_to_num(R)

def swa_xcorr_ultra(refData, shortData, win):
    """
    Calcula la correlación cruzada móvil de Pearson de forma 100% vectorial,
    sin bucles 'for', utilizando vistas de ventanas deslizantes de NumPy.
    
    shortData debe tener la forma: (canales, muestras)
    """
    refData = np.squeeze(refData)
    n_samples_ref = len(refData)
    
    # 1. Crear las ventanas deslizantes de forma virtual (sin coste de memoria extra)
    # Resultado 'windows' tendrá la forma: (canales, n_lags, n_samples_ref)
    windows = sliding_window_view(shortData, window_shape=n_samples_ref, axis=1)
    
    # 2. Operaciones sobre la señal de referencia (fijas)
    ref_dev = refData - np.mean(refData)
    ref_var = np.sum(ref_dev ** 2)
    
    # 3. Operaciones vectoriales simultáneas para TODOS los canales y TODOS los lags
    short_mean = np.mean(windows, axis=2, keepdims=True)
    short_dev = windows - short_mean
    short_var = np.sum(short_dev ** 2, axis=2)
    
    # 4. Covarianza cruzada mediante 'broadcasting' en la última dimensión (tiempo)
    covariance = np.sum(short_dev * ref_dev, axis=2)
    
    # 5. Coeficiente de correlación de Pearson final
    with np.errstate(divide='ignore', invalid='ignore'):
        R = covariance / np.sqrt(ref_var * short_var)
        
    return np.nan_to_num(R)
