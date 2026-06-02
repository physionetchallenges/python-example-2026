import numpy as np
from tqdm import tqdm
from lib.swa.swa_channelNeighbours import swa_channelNeighbours
from lib.swa.swa_filter_data import swa_filter_data
from lib.swa.swa_xcorr import swa_xcorr, swa_xcorr2, swa_xcorr_ultra
from lib.swa.swa_cluster_test import swa_cluster_test

def swa_FindSWChannels(Data, Info, SW, flag_progress=True):
    """
    Finds the slow waves present at each channel given the parameters
    already calculated for the reference wave.
    Translated from swa-Matlab.
    """
    # 1. Check inputs & structures
    if len(SW) == 0:
        print("Warning: Wave structure is empty")
        return Data, Info, SW
        


    # 2. Cluster test parameter check
    if getattr(Info['Parameters'], 'Channels_ClusterTest', True):
        if not hasattr(Info['Recording'], 'ChannelNeighbours'):
            print("Calculating: Channel Neighbours...", end="")
            # Asume la existencia de la función externa swa_channelNeighbours
            Info['Recording']['ChannelNeighbours'] = swa_channelNeighbours(Info['Electrodes'])
            print(" done.")
        else:
            print("Information: Using channels neighbourhood in 'Info'.")

    # 3. Parameter default settings for Envelope method
    if getattr(Info['Parameters'], 'Ref_Method', '') == 'Envelope':
        if not hasattr(Info['Parameters'], 'Channels_Threshold'):
            print("Warning: No further SW parameters found in Info; using defaults")
            Info['Parameters']['Channels_Threshold'] = 0.9
            Info['Parameters']['Channels_WinSize'] = 0.2

    # 4. Filter data if not already done
    if hasattr(Data, 'Filtered') and Data.Filtered is not None:
        if Data.Filtered.size == 0:
            Data.Filtered = swa_filter_data(Data['Raw'], Info)
    else:
        print("Calculation: Filtering Data.")
        Data['Filtered'] = swa_filter_data(Data['Raw'], Info)

    # Calculate window size in samples
    win = int(round(Info['Parameters']['Channels_WinSize'] * Info['Recording']['sRate']))
    
    # Dimensiones del set de datos (0: canales, 1: muestras de tiempo)
    try:
        n_channels = Data['Filtered'].shape[0]
    except:
        n_channels = 1
    n_samples = Data['Filtered'].shape[1]

    to_delete = set()

    # 5. Switch between detection methods
    detection_method = Info['Parameters']['Channels_Detection'].lower()

    # =========================================================================
    # CORRELATION METHOD
    # =========================================================================
    if detection_method == 'correlation':
        
        # Iteración con barra de progreso tqdm
        for nSW, sw in enumerate(tqdm(SW, disable=not flag_progress, desc="Finding Slow Waves (Correlation)")):
            
            # Ajuste de índice de MATLAB (1-based) a Python (0-based)
            ref_peak_ind = int(sw['Ref_PeakInd']) - 1

            # Check search window bounds
            if ref_peak_ind - win * 2 < 0 or ref_peak_ind + win * 2 >= n_samples:
                to_delete.add(nSW)
                continue

            # Extract portion of data around reference peak
            shortData = Data['Filtered'][:,ref_peak_ind - win * 2 : ref_peak_ind + win * 2 + 1]

            # Get canonical wave reference data
            correlate_type = getattr(Info['Parameters'], 'Channels_Correlate2', 'all')
            
            if correlate_type == 'mean':
                # En Python los índices de regiones deben ser enteros o booleanos
                regions = np.array(sw['Ref_Region'], dtype=int) - 1
                refData = np.mean(Data['SWRef'][regions, ref_peak_ind - win : ref_peak_ind + win + 1], axis=0)
            elif correlate_type == 'main':
                main_region = int(sw['Ref_Region'][0]) - 1
                refData = np.mean(Data['SWRef'][[main_region], ref_peak_ind - win : ref_peak_ind + win + 1], axis=0)
            else:
                # if n_channels == 1:
                    refData = Data['SWRef'][0][ref_peak_ind - win : ref_peak_ind + win + 1]
                # else:
                    # refData = np.mean(Data['SWRef'][0,ref_peak_ind - win : ref_peak_ind + win + 1], axis=0)

            # Cross correlate (Asume la existencia de la función externa swa_xcorr)
            cc = swa_xcorr_ultra(refData, shortData, win)

            # Find max correlation and its index location
            maxCC = np.max(cc, axis=1)
            maxID = np.argmax(cc, axis=1)

            # 1. Forzamos a que maxCC sea un vector plano 1D (elimina cualquier dimensión extra como (N, 1))
            maxCC_1d = np.squeeze(np.asarray(maxCC))

            # 2. Creamos channels_active como un vector booleano puramente 1D de tamaño (n_channels,)
            channels_active = maxCC_1d > Info['Parameters']['Channels_Threshold']

            # Si ningún canal correlaciona bien, se elimina la SW y se continúa
            if np.sum(channels_active) == 0:
                to_delete.add(nSW)
                continue

            # 3. Inicializar matriz de amplitudes negativas con NaNs (canales x 1)
            sw['Channels_NegAmp'] = np.full((n_channels, 1), np.nan)

            # 4. Extracción segura de amplitudes para canales activos
            if np.any(channels_active):
                if n_channels == 1:
                    sw['Channels_NegAmp'][channels_active, 0] = np.min(shortData)
                else:
                    # shortData es (n_channels, n_samples). channels_active al ser 1D filtra las filas perfectamente.
                    sw['Channels_NegAmp'][channels_active, 0] = np.min(shortData[channels_active, :], axis=1)

            # 5. Desactivar canales que no superen el umbral de amplitud absoluta mínimo
            amp_thresh = np.mean(Info['Parameters']['Ref_AmplitudeAbsolute']) / 10.0

            # Forzamos que la máscara de desactivación sea 1D para que no altere las dimensiones
            deactivate_mask = (sw['Channels_NegAmp'][:, 0] > amp_thresh).flatten()
            channels_active[deactivate_mask] = False

            if getattr(Info['Parameters'], 'Channels_ClusterTest', True):
                clusters = swa_cluster_test(channels_active.astype(float), Info['Recording']['ChannelNeighbours'], 0.01)
                clusters[np.isnan(clusters)] = 0
                n_clusters = np.unique(clusters)
                
                if len(n_clusters) > 2:  # Más allá del fondo (0) y un cluster válido
                    max_cluster_size = 0
                    keep_cluster_val = n_clusters[1]
                    for val in n_clusters:
                        if val == 0:
                            continue
                        s_cluster = np.sum(clusters == val)
                        if s_cluster > max_cluster_size:
                            max_cluster_size = s_cluster
                            keep_cluster_val = val
                    channels_active = (clusters == keep_cluster_val)

            # Recalcular con el canal prototípico si la correlación de la referencia no es muy alta
            negative_peak_index = np.nanargmin(sw['Channels_NegAmp'][:, 0])
            
            if maxCC[negative_peak_index] < (Info['Parameters']['Channels_Threshold'] + 1) / 2:
                # maxID es 0-based, compensamos el comportamiento de MATLAB (maxID_matlab - win)
                max_delay = int((maxID[negative_peak_index] + 1) - win)
                
                maxData = Data['Filtered'][
                    negative_peak_index, 
                    ref_peak_ind - win + max_delay : ref_peak_ind + win + max_delay + 1
                ]
                
                cc = swa_xcorr_ultra(maxData, shortData, win)
                maxCC = np.max(cc, axis=1)
                maxID = np.argmax(cc, axis=1)
                
                channels_active[maxCC > Info['Parameters']['Channels_Threshold']] = True

            # Limpieza final de Amplitudes Negativas basada en los canales descartados
            sw['Channels_NegAmp'][~channels_active, 0] = np.nan

            # Delay Calculation
            sw['Travelling_Delays'] = np.full((n_channels, 1), np.nan)
            # maxID de Python ya es nativamente relativo a 0
            sw['Travelling_Delays'][channels_active, 0] = maxID[channels_active] - np.min(maxID[channels_active])

            sw['Channels_Globality'] = (np.sum(channels_active) / n_channels) * 100.0
            sw['Channels_Active'] = channels_active

    # =========================================================================
    # THRESHOLD METHOD
    # =========================================================================
    elif detection_method == 'threshold':
        if not getattr(Info['Parameters'], 'Ref_Peak2Peak', None):
            Info['Parameters']['Ref_Peak2Peak'] = abs(Info['Parameters']['Ref_NegAmpMin']) * 1.75

        for nSW, sw in enumerate(tqdm(SW, disable=not flag_progress, desc="Finding Slow Waves (Threshold)")):
            
            ref_peak_ind = int(sw['Ref_PeakInd']) - 1

            if ref_peak_ind - win < 0 or ref_peak_ind + win * 3 >= n_samples:
                to_delete.add(nSW)
                continue

            # Short window for negative peak
            shortData = Data['Filtered'][:, ref_peak_ind - win : ref_peak_ind + win + 1]
            
            sw.Channels_NegAmp = np.min(shortData, axis=1)
            minChId = np.argmax(shortData, axis=1) # El índice relativo del mínimo dentro de este bloque
            
            thresh_val = -np.mean(Info['Parameters']['Ref_AmplitudeAbsolute']) * Info['Parameters']['Channels_Threshold']
            sw.Channels_Active = sw.Channels_NegAmp < thresh_val

            # Peak to Peak Check (ventana más larga hacia adelante)
            shortData_long = Data['Filtered'][:, ref_peak_ind - win : ref_peak_ind + win * 3 + 1]
            
            posPeakAmp = np.full(n_channels, np.nan)
            active_indices = np.where(sw.Channels_Active)[0]
            
            for nCh in active_indices:
                # Buscamos el pico positivo después del pico negativo detectado
                posPeakAmp[nCh] = np.max(shortData_long[nCh, minChId[nCh]:])

            # Aplicar filtro de umbral pico a pico
            p2p_thresh = Info['Parameters']['Ref_Peak2Peak'] * Info['Parameters']['Channels_Threshold']
            sw.Channels_Active[posPeakAmp - sw.Channels_NegAmp < p2p_thresh] = False

            # Eliminar canales sub-umbral de las amplitudes negativas
            sw.Channels_NegAmp[~sw.Channels_Active] = np.nan

            if np.sum(sw.Channels_Active) == 0:
                to_delete.add(nSW)
                continue

            # Delay Calculation usando el identificador del pico negativo
            sw.Travelling_Delays = np.full((len(Info.Electrodes), 1), np.nan)
            active_delays = minChId[sw.Channels_Active]
            sw.Travelling_Delays[sw.Channels_Active, 0] = active_delays - np.min(active_delays)

            sw.Channels_Globality = (np.sum(sw.Channels_Active) / len(sw.Channels_Active)) * 100.0

    # 6. Delete the bad waves from list
    print(f"Information: {len(to_delete)} slow waves were removed due to insufficient criteria.")
    SW = [sw for idx, sw in enumerate(SW) if idx not in to_delete]

    return Data, Info, SW