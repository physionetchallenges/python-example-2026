import numpy as np

def swa_get_peaks(slope_data, Info, flag_notch=False):
    """
    Finds the maximum negative peaks (MNP) and maximum positive peaks (MPP) 
    within a channel's slope data and iteratively removes small notches.
    Translated from swa-Matlab.

    Parameters:
    -----------
    slope_data : numpy.ndarray
        1D array containing the derivative/slope of the EEG signal.
    Info : dict
        Configuration dictionary containing parameters and sampling rate.
    flag_notch : bool, optional
        If True, iteratively erases small bumps/notches based on wavelength.

    Returns:
    --------
    MNP : numpy.ndarray
        Indices of Maximum Negative Peaks (local minima).
    MPP : numpy.ndarray
        Indices of Maximum Positive Peaks (local maxima).
    """
    # Encontrar el signo de la pendiente (-1, 0, 1)
    sign_data = np.sign(slope_data)
    # Forzar que los ceros cuenten como positivos para evitar problemas en mesetas
    sign_data[sign_data == 0] = 1 
    
    diff_sign = np.diff(sign_data)
    
    # MNP: La pendiente pasa de negativa a positiva (mínimo local / pico negativo)
    # En Python obtenemos el índice del cambio
    MNP = np.where(diff_sign == 2)[0] + 1
    
    # MPP: La pendiente pasa de positiva a negativa (máximo local / pico positivo)
    MPP = np.where(diff_sign == -2)[0] + 1

    if len(MNP) == 0 or len(MPP) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Asegurar que la secuencia empiece con un MPP (Pico Positivo Anterior)
    if MNP[0] < MPP[0]:
        MNP = MNP[1:]
        
    if len(MNP) == 0 or len(MPP) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Asegurar que la secuencia termine con un MPP (Pico Positivo Posterior)
    if MNP[-1] > MPP[-1]:
        MNP = MNP[:-1]

    # Eliminar muescas/ruido pequeño de forma iterativa
    if flag_notch and len(MNP) > 0 and len(MPP) > 0:
        sRate = Info['Recording']['sRate']
        # Umbral: 10% de la longitud de onda mínima permitida
        thresh = Info['Parameters']['Ref_WaveLength'][0] * sRate / 10.0
        
        nb = 1
        while nb > 0:
            if len(MNP) == 0 or len(MPP) < 2:
                break
            
            # --- 1. Eliminar "Bumps" Positivos ---
            # Distancia entre el MNP actual y el SIGUIENTE MPP
            posBumps = (MPP[1:] - MNP) < thresh
            
            if np.any(posBumps):
                # En MPP queremos borrar el elemento subsiguiente, así que añadimos False al inicio
                mpp_mask = np.concatenate(([False], posBumps))
                MPP = MPP[~mpp_mask]
                MNP = MNP[~posBumps]
            
            if len(MNP) == 0 or len(MPP) < 2:
                break
                
            # --- 2. Eliminar "Bumps" Negativos ---
            # Distancia entre el MNP actual y el ANTERIOR MPP
            negBumps = (MNP - MPP[:-1]) < thresh
            
            if np.any(negBumps):
                # En MPP queremos borrar el elemento previo, así que añadimos False al final
                mpp_mask = np.concatenate((negBumps, [False]))
                MPP = MPP[~mpp_mask]
                MNP = MNP[~negBumps]
            
            # Contar cuántos elementos se eliminaron en esta ronda para decidir si continuar
            sum_pos = np.sum(posBumps) if np.any(posBumps) else 0
            sum_neg = np.sum(negBumps) if np.any(negBumps) else 0
            nb = max(sum_pos, sum_neg)

    return MNP, MPP