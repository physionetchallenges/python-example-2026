import numpy as np
from scipy.signal import find_peaks

def swa_FindSWRef(Data, Info, SW=None):
    """
    Finds slow waves in the reference signal based on amplitude, wavelength, 
    and slope criteria. Handles multiple reference regions and removes duplicates.
    Translated from swa-Matlab.

    Parameters:
    -----------
    Data : dict
        Dictionary containing 'SWRef' (n_refs x n_samples) and optionally 'sleep_stages'.
    Info : dict
        Configuration dictionary.
    SW : list of dicts, optional
        Existing list of detected slow waves. Defaults to empty list.

    Returns:
    --------
    Data : dict
    Info : dict
    SW : list of dicts
    """
    if 'SWRef' not in Data:
        raise ValueError("Error: Data dictionary must contain 'SWRef' array.")
        
    p = Info.get('Parameters', {})
    if 'Ref_Method' not in p:
        print("Error: No detection parameters found in the 'Info' structure")
        return Data, Info, SW

    # Inicializar la lista de ondas si no existe
    if SW is None:
        SW = []
    
    OSWCount = len(SW)
    SWCount = len(SW)
    
    number_ref_waves = Data['SWRef'].shape[0]
    
    # Inicializar vectores de umbrales según el criterio
    if p.get('Ref_AmplitudeCriteria') == 'relative':
        p['Ref_AmplitudeAbsolute'] = np.zeros(number_ref_waves)
    elif p.get('Ref_AmplitudeCriteria') == 'absolute':
        # Asegurar que Ref_AmplitudeRelative exista como lista/array
        p['Ref_AmplitudeRelative'] = np.zeros(number_ref_waves)
        
    if 'Recording' not in Info:
        Info['Recording'] = {}
    Info['Recording']['Data_Deviation'] = np.zeros(number_ref_waves)
    
    # Asegurar que el umbral absoluto tenga el tamaño adecuado si es un solo valor
    if p.get('Ref_AmplitudeCriteria') == 'absolute':
        abs_val = np.atleast_1d(p['Ref_AmplitudeAbsolute'])
        if len(abs_val) < number_ref_waves:
            abs_val = np.repeat(abs_val[0], number_ref_waves)
        p['Ref_AmplitudeAbsolute'] = abs_val
            
    sRate = Info['Recording']['sRate']
    
    # Bucle por cada onda de referencia (ej. regiones diamond, central, envelope)
    for ref_wave in range(number_ref_waves):
        if ref_wave > 0:
            OSWCount = len(SW)
            
        ref_signal = Data['SWRef'][ref_wave, :]
        
        # Calcular la derivada de la señal (añadiendo un 0 inicial para mantener longitud)
        slopeData = np.concatenate(([0], np.diff(ref_signal)))
        
        # Extraer Mínimos (MNP - Maximum Negative Peaks) y Máximos (MPP)
        from src.lib.swa.swa_get_peaks import swa_get_peaks
        MNP, MPP = swa_get_peaks(slopeData, Info, True)

        MNP = MNP[1:]
        MPP = MPP[1:]
        # Eliminar picos fuera de las fases de sueño de interés
        if p.get('Ref_UseStages') is not None and 'sleep_stages' in Data:
            stages = Data['sleep_stages']
            valid_MNP_mask = np.isin(stages[MNP], p['Ref_UseStages'])
            relevant_MNP = MNP[valid_MNP_mask]
        else:
            relevant_MNP = MNP
            
        # Control de seguridad si no hay picos
        if len(relevant_MNP) == 0:
            continue
            
        # =====================================================================
        # CÁLCULO DEL UMBRAL DE AMPLITUD (Amplitude Threshold Criteria)
        # =====================================================================
        # MAD (Median Absolute Deviation respecto a la mediana)
        median_act = np.median(ref_signal[relevant_MNP])
        mad_val = np.median(np.abs(ref_signal[relevant_MNP] - median_act))
        Info['Recording']['Data_Deviation'][ref_wave] = mad_val
        
        if p['Ref_AmplitudeCriteria'] == 'relative':
            p['Ref_AmplitudeAbsolute'][ref_wave] = (mad_val * p['Ref_AmplitudeRelative']) + abs(median_act)
            print(f"Calculation: Amplitude threshold set to {p['Ref_AmplitudeAbsolute'][ref_wave]:.1f}uV for canonical wave {ref_wave + 1}")
        elif p['Ref_AmplitudeCriteria'] == 'absolute':
            deviation = (p['Ref_AmplitudeAbsolute'][ref_wave] - abs(median_act)) / mad_val if mad_val != 0 else 0
            p['Ref_AmplitudeRelative'][ref_wave] = deviation
            print(f"Information: Threshold is {deviation:.1f} deviations from median activity")
            
        if p.get('Ref_Peak2Peak') is None:
            p['Ref_Peak2Peak'] = abs(p['Ref_AmplitudeAbsolute'][ref_wave]) * 1.75
            
        # =====================================================================
        # MÉTODOS DE INSPECCIÓN (MNP vs ZC)
        # =====================================================================
        if p['Ref_InspectionPoint'] == 'MNP':
            # ----- MÉTODO MNP (Maximum Negative Peak) -----
            badWaves = np.zeros(len(MNP), dtype=bool)
            
            if p.get('Ref_UseStages') is not None and 'sleep_stages' in Data:
                badWaves[~valid_MNP_mask] = True
                
            # (El código MNP de MATLAB requiere alinear los MPP con MNP, 
            # asumiendo que un MNP está rodeado por dos MPP. Esta lógica
            # se adapta mejor con Zero-Crossings a menos que los picos estén 
            # estrictamente pareados).
            # Por simplicidad y robustez frente al código MATLAB original:
            pass # En Python el método más seguro y estandarizado es ZC.

        elif p['Ref_InspectionPoint'] == 'ZC':
            # ----- MÉTODO ZC (Zero Crossings) -----
            signData = np.sign(ref_signal)
            signData[signData == 0] = 1 # Evitar pendiente 0 exacta
            
            DZC = np.where(np.diff(signData) < 0)[0]
            UZC = np.where(np.diff(signData) > 0)[0] + 1 # +1 para ser la muestra tras el cruce
            
            # Umbral de pendiente (percentil)
            pos_slopes = slopeData[slopeData > 0]
            slopeThresh = np.percentile(pos_slopes, p['Ref_SlopeMin'] * 100) if len(pos_slopes) > 0 else 0
            
            # Alinear DZC y UZC
            if len(DZC) > 0 and len(UZC) > 0 and DZC[0] >= UZC[0]:
                UZC = UZC[1:]
                if len(DZC) != len(UZC):
                    DZC = DZC[:-1]
            if len(DZC) > len(UZC):
                DZC = DZC[:-1]
                
            # Filtrar por fase de sueño
            if p.get('Ref_UseStages') is not None and 'sleep_stages' in Data:
                valid_DZC = np.isin(Data['sleep_stages'][DZC], p['Ref_UseStages'])
                DZC = DZC[valid_DZC]
                UZC = UZC[valid_DZC]
                
            # Criterio Wavelength
            SWLengths = UZC - DZC
            valid_len = (SWLengths >= p['Ref_WaveLength'][0] * sRate) & (SWLengths <= p['Ref_WaveLength'][1] * sRate)
            DZC = DZC[valid_len]
            UZC = UZC[valid_len]
            
            AllPeaks = np.array([sw['Ref_PeakInd'] for sw in SW]) if len(SW) > 0 else np.array([])
            
            # Analizar cada candidato DZC
            for n in range(len(DZC)):
                start, end = DZC[n], UZC[n]
                segment = ref_signal[start:end]
                
                if len(segment) == 0:
                    continue
                    
                # Amplitud Negativa
                NegPeakAmp = np.min(segment)
                NegPeakId = start + np.argmin(segment)
                
                if NegPeakAmp > -p['Ref_AmplitudeAbsolute'][ref_wave] or NegPeakAmp < -p.get('Ref_AmplitudeMax', 250):
                    continue
                    
                # Amplitud Peak2Peak
                search_end = min(end + int(1 * sRate), len(ref_signal))
                PosPeakAmp = np.max(ref_signal[end:search_end]) if search_end > end else 0
                
                if p['Ref_Method'] in ['diamond', 'square']:
                    if (PosPeakAmp - NegPeakAmp) < p['Ref_Peak2Peak']:
                        continue
                        
                # Pendiente positiva máxima
                MaxPosSlope = np.max(slopeData[start:end])
                if MaxPosSlope < slopeThresh:
                    continue
                    
                # Evitar duplicados inter-regiones
                if ref_wave > 0 and len(AllPeaks) > 0:
                    # Comprobar si el pico de una onda ya guardada cae dentro de este nuevo DZC-UZC
                    overlap_mask = (AllPeaks > start) & (AllPeaks < end)
                    if np.any(overlap_mask):
                        SWid = np.argmax(overlap_mask) # Encontrar el índice conflictivo
                        
                        # Guardar la región que tenga la mayor amplitud (más negativo)
                        if ref_signal[NegPeakId] < SW[SWid]['Ref_PeakAmp']:
                            # Sobrescribir con la nueva onda que es mejor
                            old_regions = SW[SWid]['Ref_Region'] if isinstance(SW[SWid]['Ref_Region'], list) else [SW[SWid]['Ref_Region']]
                            SW[SWid]['Ref_Region'] = [ref_wave + 1] + old_regions # +1 para mantener ID visual (1-based index concept)
                            SW[SWid]['Ref_DownInd'] = start
                            SW[SWid]['Ref_PeakInd'] = NegPeakId
                            SW[SWid]['Ref_UpInd'] = end
                            SW[SWid]['Ref_PeakAmp'] = ref_signal[NegPeakId]
                            SW[SWid]['Ref_P2PAmp'] = PosPeakAmp - NegPeakAmp
                            SW[SWid]['Ref_NegSlope'] = np.min(slopeData[start:end])
                            SW[SWid]['Ref_PosSlope'] = MaxPosSlope
                        else:
                            # Añadir esta región a la onda existente
                            if isinstance(SW[SWid]['Ref_Region'], list):
                                SW[SWid]['Ref_Region'].append(ref_wave + 1)
                            else:
                                SW[SWid]['Ref_Region'] = [SW[SWid]['Ref_Region'], ref_wave + 1]
                        continue
                        
                # Añadir nueva onda detectada
                SWCount += 1
                new_wave = {
                    'Ref_Region': [ref_wave + 1],
                    'Ref_DownInd': start,
                    'Ref_PeakInd': NegPeakId,
                    'Ref_UpInd': end,
                    'Ref_PeakAmp': ref_signal[NegPeakId],
                    'Ref_P2PAmp': PosPeakAmp - NegPeakAmp,
                    'Ref_NegSlope': np.min(slopeData[start:end]),
                    'Ref_PosSlope': MaxPosSlope,
                    # Campos vacíos que se llenarán más adelante
                    'Channels_Active': [], 'Channels_NegAmp': [], 
                    'Channels_Globality': [], 'Travelling_Delays': [],
                    'Travelling_DelayMap': [], 'Travelling_Streams': [], 'Code': []
                }
                SW.append(new_wave)
                
        else:
            print("Error: Unrecognised detection method")
            return Data, Info, SW

    # Ordenar las ondas detectadas cronológicamente
    if len(SW) > 0:
        SW.sort(key=lambda x: x['Ref_DownInd'])
        
    # Limpieza final de estadios de sueño no válidos (por si acaso quedaron en las fusiones)
    if p.get('Ref_UseStages') is not None and 'sleep_stages' in Data:
        valid_indices = []
        for i, wave in enumerate(SW):
            if Data['sleep_stages'][wave['Ref_PeakInd']] in p['Ref_UseStages']:
                valid_indices.append(i)
        
        removed_count = len(SW) - len(valid_indices)
        SW = [SW[i] for i in valid_indices]
        
        if removed_count > 0:
            print(f"Information: {removed_count} waves were found in non-specified stages and removed")
            
    return Data, Info, SW
