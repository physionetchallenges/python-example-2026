import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def swa_CalculateReference(data, Info, display_plot=False):
    """
    Calculates the canonical reference wave(s) for slow wave detection 
    and applies the bandpass filter.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Original EEG data matrix of shape (n_channels, n_samples).
    Info : dict
        Configuration dictionary.
    display_plot : bool, optional
        If True, plots electrode selections and a 15-second data sample.
        
    Returns:
    --------
    filtData : numpy.ndarray
        Filtered reference wave(s). Shape depends on method (e.g., 1 row for envelope, 4 for diamond).
    Info : dict
        Updated Info dictionary with chosen parameters and reference electrodes.
    """
    # 1. Validaciones de estructura iniciales
    if 'Electrodes' not in Info:
        raise ValueError("Error: No electrode information found in Info")
    if 'Recording' not in Info or 'sRate' not in Info['Recording']:
        raise ValueError("Error: No sampling rate information found in Info['Recording']")
    if 'Parameters' not in Info:
        from src.lib.swa.swa_getInfoDefaults import swa_getInfoDefaults  # Evitar importación cíclica
        Info = swa_getInfoDefaults(Info, 'SW', 'envelope')
        print("Warning: No parameters specified; using defaults.")
        
    # Compatibilidad de minúsculas
    Info['Parameters']['Ref_Method'] = Info['Parameters']['Ref_Method'].lower()
    ref_method = Info['Parameters']['Ref_Method']
    print(f"Calculating: Canonical wave ({ref_method})")
    
    # 2. Ajuste y proyección 2D de las coordenadas de los electrodos
    # MATLAB: Th = pi/180*[Info.Electrodes.theta]; Rd = [Info.Electrodes.radius];
    # e_locs = Info['Electrodes']
    # # Soporta si e_locs es una lista de objetos o una lista de dicts (desde el JSON)
    # theta = np.array([getattr(el, 'theta', el.get('theta')) for el in e_locs], dtype=float)
    # radius = np.array([getattr(el, 'radius', el.get('radius')) for el in e_locs], dtype=float)
    
    # Th = (np.pi / 180.0) * theta
    # x = radius * np.cos(Th)
    # y = radius * np.sin(Th)
    
    # # Encajonar las coordenadas en un rango de -0.5 a 0.5
    # intrad = min(1.0, max(np.abs(radius)))
    # intrad = max(intrad, 0.5)
    # squeezefac = 0.5 / intrad
    # x = x * squeezefac
    # y = y * squeezefac
    
    # Inicializar figura si flag_plot está activo
    fig_topo, ax_topo = None, None
    # if display_plot:
    #     fig_topo, ax_topo = plt.subplots(figsize=(6, 6))
    #     ax_topo.scatter(y, x, s=30, edgecolors=[0.5, 0.5, 0.5], facecolors=[0.5, 0.5, 0.5], label='All Electrodes')
    #     ax_topo.set_aspect('equal')
    #     ax_topo.axis('off')
        
    n_samples = data.shape[1]
    n_total_ch = len(Info['Electrodes'])
    
    # =========================================================================
    # SELECCIÓN POR MÉTODOS
    # =========================================================================
    
    if ref_method == 'envelope':
        # Evitar canales externos/periféricos si está configurado
        # if Info['Parameters'].get('Ref_UseInside', True):
        #     distances = np.sqrt(x**2 + y**2)
        #     # Guardar máscara (1 fila, n_canales)
        #     Info['Parameters']['Ref_Electrodes'] = distances < 0.35
        #     working_data = data.iloc[Info['Parameters']['Ref_Electrodes'], :]
        # else:
        Info['Parameters']['Ref_Electrodes'] = np.ones(n_total_ch, dtype=bool)
        working_data = data
            
        # Ordenar muestras para obtener el percentil de negatividad
        rData = np.sort(working_data, axis=0)
        nCh = max(3, int(np.floor(n_total_ch * 0.025)))
        
        # Si hay más de 3 canales, saltamos el más negativo para evitar artefactos
        if nCh > 3:
            nData = np.mean(rData[1:nCh, :], axis=0, keepdims=True)
        else:
            nData = np.mean(rData[0:nCh, :], axis=0, keepdims=True)
            nData = rData
            
    elif ref_method in ['square', 'diamond']:
        distance_from_center = 0.2
        circle_radius = 0.175
        
        if ref_method == 'square':
            RegionCenters = np.array([
                [-distance_from_center, -distance_from_center,  distance_from_center,  distance_from_center],
                [ distance_from_center, -distance_from_center, -distance_from_center,  distance_from_center]
            ])
        else: # diamond
            RegionCenters = np.array([
                [distance_from_center,                   0,                    0, -distance_from_center],
                [                   0, distance_from_center, -distance_from_center,                    0]
            ])
            
        nData = np.zeros((4, n_samples))
        ref_electrodes_mask = np.zeros((4, n_total_ch), dtype=bool)
        
        for n in range(4):
            distances = np.sqrt((x + RegionCenters[0, n])**2 + (y + RegionCenters[1, n])**2)
            ref_electrodes_mask[n, :] = distances < circle_radius
            nData[n, :] = np.mean(data[ref_electrodes_mask[n, :], :], axis=0)
            
        Info['Parameters']['Ref_Electrodes'] = ref_electrodes_mask
        
    elif ref_method == 'grid':
        distance_from_center = 0.225
        circle_radius = 0.10
        
        # Replicar meshgrid (-1 a 1) de MATLAB
        grid_x, grid_y = np.meshgrid([-distance_from_center, 0, distance_from_center], 
                                     [-distance_from_center, 0, distance_from_center])
        centers_x = grid_x.flatten()
        centers_y = grid_y.flatten()
        
        nData = np.zeros((9, n_samples))
        ref_electrodes_mask = np.zeros((9, n_total_ch), dtype=bool)
        
        for n in range(9):
            distances = np.sqrt((x + centers_x[n])**2 + (y + centers_y[n])**2)
            ref_electrodes_mask[n, :] = distances < circle_radius
            nData[n, :] = np.mean(data[ref_electrodes_mask[n, :], :], axis=0)
            
        Info['Parameters']['Ref_Electrodes'] = ref_electrodes_mask
        
    elif ref_method == 'central':
        circle_radius = 0.175
        distances = np.sqrt(x**2 + y**2)
        
        Info['Parameters']['Ref_Electrodes'] = distances < circle_radius
        print(f"Information: Central using {np.sum(Info['Parameters']['Ref_Electrodes'])} channels for reference")
        nData = np.mean(data[Info['Parameters']['Ref_Electrodes'], :], axis=0, keepdims=True)
        
    elif ref_method == 'midline':
        distance_from_center = 0.25
        circle_radius = 0.125
        RegionCenters = np.array([
            [-distance_from_center, 0, distance_from_center],
            [                    0, 0,                 0]
        ])
        
        nData = np.zeros((3, n_samples))
        ref_electrodes_mask = np.zeros((3, n_total_ch), dtype=bool)
        
        for n in range(3):
            distances = np.sqrt((x + RegionCenters[0, n])**2 + (y + RegionCenters[1, n])**2)
            ref_electrodes_mask[n, :] = distances < circle_radius
            nData[n, :] = np.mean(data[ref_electrodes_mask[n, :], :], axis=0)
            
        Info['Parameters']['Ref_Electrodes'] = ref_electrodes_mask
        
    else:
        raise ValueError(f"Unrecognised reference method type: {ref_method}")
        
    # =========================================================================
    # PLOT REGIONES (Si corresponde)
    # =========================================================================
    if display_plot and ax_topo is not None:
        mask_matrix = np.atleast_2d(Info['Parameters']['Ref_Electrodes'])
        no_regions = mask_matrix.shape[0]
        colors = cm.get_cmap('tab10', no_regions)
        
        for n in range(no_regions):
            region_idx = mask_matrix[n, :]
            ax_topo.scatter(y[region_idx], x[region_idx], s=90, 
                            edgecolors=[0.3, 0.3, 0.3], facecolors=colors(n), 
                            label=f'Region {n+1}')
        plt.title(f"Selected Electrodes: {ref_method}")
        plt.show()

    # =========================================================================
    # FILTRADO DE LA SEÑAL CANÓNICA
    # =========================================================================
    if Info['Parameters'].get('Filter_Apply', True):
        if 'Filter_Method' not in Info['Parameters']:
            print("Information: No filter parameters given, using defaults.")
            Info['Parameters']['Filter_Method'] = 'Chebyshev'
            Info['Parameters']['Filter_hPass'] = 0.2
            Info['Parameters']['Filter_lPass'] = 4.0
            Info['Parameters']['Filter_order'] = 2
            
        print(f"Calculation: Applying {Info['Parameters']['Filter_Method']} filter for [{Info['Parameters']['Filter_hPass']:.1f}, {Info['Parameters']['Filter_lPass']:.1f}] Hz...")
        
        # Llamar a la función previamente traducida
        from src.lib.swa.swa_filter_data import swa_filter_data
        filtData = swa_filter_data(nData, Info)
        print("Done")
    else:
        filtData = nData

    # =========================================================================
    # PLOT 15 SEGUNDOS DE SEÑAL
    # =========================================================================
    if display_plot:
        sRate = Info['Recording']['sRate']
        # Tomar un punto aleatorio asegurando espacio para 15 segundos
        max_start = max(1, n_samples - int(15 * sRate))
        random_sample = np.random.randint(0, max_start)
        sample_range = np.arange(random_sample, random_sample + int(15 * sRate))
        time_range = np.arange(len(sample_range)) / sRate
        
        plt.figure(figsize=(10, 4))
        no_waves = filtData.shape[0]
        colors = cm.get_cmap('tab10', no_waves)
        
        for n in range(no_waves):
            if ref_method == 'envelope':
                plt.plot(time_range, filtData[n, sample_range], color=[0.2, 0.2, 0.2], linewidth=2.5, label='Envelope Ref')
            else:
                # Separar las ondas verticalmente como lo hace MATLAB si hay múltiples regiones
                plt.plot(time_range, filtData[n, sample_range] - (n * 60), color=colors(n), linewidth=1.5, label=f'Region {n+1}')
                
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title('15 Second Sample of Calculated Reference Wave(s)')
        plt.grid(True, alpha=0.3)
        plt.show()

    return filtData, Info