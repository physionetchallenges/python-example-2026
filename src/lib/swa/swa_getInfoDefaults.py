def swa_getInfoDefaults(Info, type_wave, method='envelope'):
    """
    Get the current default detection parameters for slow waves (SW), spindles (SS),
    or saw-tooth waves (ST).
    Translated from swa-Matlab.

    Parameters:
    -----------
    Info : dict
        The configuration dictionary (equivalent to Info struct in MATLAB).
    type_wave : str
        Type of event to detect: 'SW' (Slow Waves), 'ST' (Saw-tooth), or 'SS' (Spindles).
    method : str, optional
        Method used mainly for 'SW' (e.g., 'envelope' or 'mdc'). Default is 'envelope'.

    Returns:
    --------
    Info : dict
        The updated dictionary containing all default configuration parameters.
    """
    # Asegurar que la clave 'Parameters' exista en el diccionario Info
    if 'Parameters' not in Info:
        Info['Parameters'] = {}
        
    p = Info['Parameters']  # Atajo para simplificar la escritura del código

    # Manejar el valor por defecto del método si viene vacío o nulo
    if method is None or method == '':
        method = 'envelope'
        
    type_wave = type_wave.upper()

    # =========================================================================
    # CASE: SLOW WAVES (SW)
    # =========================================================================
    if type_wave == 'SW':
        # Parámetros de filtrado
        p['Filter_Apply'] = True
        p['Filter_Method'] = 'Chebyshev'  # 'Chebyshev' / 'Buttersworth'
        p['Filter_hPass'] = 0.5
        p['Filter_lPass'] = 4.0
        p['Filter_order'] = 2

        # Detección en canal de referencia (Canonical Wave)
        p['Ref_Method'] = None  # Método canónico ('envelope', 'diamond', 'midline', etc.)
        p['Ref_Electrodes'] = False  # Array lógico de electrodos usados
        p['Ref_InspectionPoint'] = 'MNP'  # 'MNP' (Max Neg Peak) / 'ZC' (Zero Crossing)
        p['Ref_UseInside'] = 1  # 1: Usar canales interiores de la cabeza / 0: Todos
        p['Ref_UseStages'] = None  # Información de estadiaje de sueño (Sleep Scoring)
        p['Ref_AmplitudeCriteria'] = 'relative'  # 'relative' / 'absolute'
        p['Ref_AmplitudeRelative'] = 5.0  # Desviaciones estándar desde la media de negatividad
        p['Ref_AmplitudeAbsolute'] = 60.0
        p['Ref_AmplitudeMax'] = 250.0  # Amplitud máxima para control de artefactos
        p['Ref_WaveLength'] = [0.25, 1.25]  # Criterio de longitud (segundos) entre cruces por cero
        p['Ref_SlopeMin'] = 0.90  # Porcentaje de corte para las pendientes (slopes)
        p['Ref_Peak2Peak'] = None  # Solo para umbralización por canales

        # Detección por canales individuales
        p['Channels_Correlate2'] = 'mean'  # Qué onda canónica usar ('main', 'mean' o 'all')
        p['Channels_Detection'] = 'correlation'  # 'correlation' / 'threshold'
        p['Channels_Threshold'] = 0.9  # Ajuste si se usa el método de umbralización
        p['Channels_ClusterTest'] = True
        p['Channels_WinSize'] = 0.100  # Ventana de búsqueda en segundos

        # Parámetros de propagación (Travelling)
        p['Travelling_GS'] = 40  # Tamaño de la cuadrícula de interpolación
        p['Travelling_MinDelay'] = 40.0  # Tiempo de viaje mínimo (ms)
        p['Travelling_RecalculateDelay'] = True  # False si los mapas de retraso se calculan fuera

        # Configuración específica según el método elegido
        if method.lower() == 'envelope':
            p['Ref_Method'] = 'Envelope'
        elif method.lower() == 'mdc':  # Massimini Detection Criteria
            p['Ref_Method'] = 'diamond'
            p['Ref_AmplitudeCriteria'] = 'absolute'
            p['Ref_InspectionPoint'] = 'ZC'
            p['Ref_AmplitudeAbsolute'] = 80.0
            p['Ref_Peak2Peak'] = 140.0
            p['Channels_Detection'] = 'threshold'
            p['Channels_Threshold'] = 1.0

    # =========================================================================
    # CASE: SAW-TOOTH WAVES (ST)
    # =========================================================================
    elif type_wave == 'ST':
        p['Ref_Method'] = 'Midline'
        p['Ref_Electrodes'] = None
        p['Filter_Apply'] = False  # No se necesita filtro clásico para el método CWT
        
        # Parámetros Wavelet (CWT) para detección
        p['CWT_hPass'] = 2.0
        p['CWT_lPass'] = 5.0
        p['CWT_StdThresh'] = 1.75
        p['CWT_AmpThresh'] = None
        p['CWT_ThetaAlpha'] = 1.2
        
        p['Burst_Length'] = 1.0  # Tiempo máximo entre ondas en segundos
        p['Burst_Adjust'] = 0.75  # Ajuste de criterio si la onda se encuentra en ráfaga
        
        p['Channels_WinSize'] = 0.060  # En segundos
        p['Channel_Adjust'] = 0.9
        p['Travelling_GS'] = 40
        p['Travelling_MinDelay'] = 20.0

    # =========================================================================
    # CASE: SPINDLES (SS)
    # =========================================================================
    elif type_wave == 'SS':
        p['Ref_Method'] = 'Midline'
        p['Ref_Electrodes'] = None
        p['Filter_Apply'] = False
        p['Ref_UseStages'] = None
            
        # Parámetros del filtro de Spindles
        p['Filter_Method'] = 'Chebyshev'
        p['Filter_band'] = [10.0, 16.0]
        p['Filter_checkrange'] = 2.0
        p['Filter_Window'] = 0.150  # Ventana de suavizado RMS de la potencia
        p['Filter_order'] = 2
        p['Wavelet_name'] = 'fbsp1-1-3'  # B-spline wavelet
        p['Wavelet_norm'] = 1
        
        # Criterios del huso de sueño (Spindle)
        p['Ref_AmplitudeCriteria'] = 'relative'
        p['Ref_AmplitudeMetric'] = 'median'
        p['Ref_AmplitudeRelative'] = [4.0, 2.0]  # [umbral alto, umbral bajo] en Desviaciones Estándar
        p['Ref_AmplitudeAbsolute'] = 15.0
        p['Ref_NeighbourRatio'] = 3.0  # Ratio mínimo potencia Spindle/Vecinos
        
        p['Ref_WaveLength'] = [0.3, 3.0]  # Tiempo en segundos sobre el umbral
        p['Ref_MinWaves'] = 3  # Número mínimo de ondas internas del huso
               
        p['Channels_Method'] = 'power'  # Método wavelet o potencia (FFT)
        p['Channels_WinSize'] = 0.150  # Ventana de búsqueda alrededor del huso de referencia
        p['Channels_Threshold'] = 0.75  # Ajuste respecto al criterio de referencia
        
    else:
        raise ValueError(f"Unknown wave type: {type_wave}. Choose 'SW', 'ST', or 'SS'.")

    return Info