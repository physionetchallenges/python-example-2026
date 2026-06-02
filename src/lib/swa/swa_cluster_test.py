import numpy as np

def swa_cluster_test(data, ChN, threshold):
    """
    Finds clusters of spatially contiguous channels that exceed a given threshold
    using a breadth-first search on the channel network.
    Translated from swa-Matlab.
    
    Parameters:
    -----------
    data : numpy.ndarray
        1D array of shape (n_channels,) containing the values to test (e.g., t-values, amplitudes).
    ChN : numpy.ndarray
        Neighbor matrix of shape (n_channels, max_neighbors) from swa_channelNeighbours (1-based indexing).
    threshold : float
        The minimum value required to consider a channel for clustering.
        
    Returns:
    --------
    clusters : numpy.ndarray
        1D array of shape (n_channels,) containing NaN for unclustered channels
        and integer IDs (1, 2, 3...) for each detected spatial cluster.
    """
    # Forzar que data sea un vector plano (1D)
    data = data.flatten()
    n_channels = len(data)
    
    # Pre-asignar salidas y banderas de control
    flag_used = np.zeros(n_channels, dtype=bool)
    clusters = np.full(n_channels, np.nan)  # Relleno de NaNs como en MATLAB
    
    cluster_id = 1
    
    for n in range(n_channels):
        # Si el canal no ha sido usado y supera el umbral, inicia un nuevo cluster
        if not flag_used[n] and data[n] > threshold:
            
            # Marcar el canal inicial como examinado
            flag_used[n] = True
            
            # Inicializar la lista del cluster (cola de exploración)
            cluster_list = [n]
            current_count = 0  # Python usa indexación 0-based
            
            # El bucle se detiene cuando revisamos todos los canales añadidos a la lista
            while current_count < len(cluster_list):
                
                # Canal que estamos analizando actualmente
                current_channel = cluster_list[current_count]
                
                # Obtener vecinos (filtrando los ceros que actúan como relleno en ChN)
                # ChN viene en Base-1, restamos 1 para indexar en Python
                raw_neighbours = ChN[current_channel, :]
                current_neighbours = raw_neighbours[raw_neighbours > 0] - 1
                current_neighbours = current_neighbours.astype(int)
                
                if len(current_neighbours) > 0:
                    # Encontrar cuáles vecinos pasan el umbral y NO han sido usados todavía
                    in_cluster = (data[current_neighbours] > threshold) & (~flag_used[current_neighbours])
                    
                    if np.sum(in_cluster) > 0:
                        # Extraer los índices reales de los vecinos que se unen al cluster
                        valid_neighbours = current_neighbours[in_cluster]
                        
                        # Marcarlos como usados inmediatamente para que no los vuelva a agarrar otro canal
                        flag_used[valid_neighbours] = True
                        
                        # Expandir la lista de canales a revisar (cola del BFS)
                        cluster_list.extend(valid_neighbours)
                
                # Avanzar al siguiente canal de la lista
                current_count += 1
            
            # Asignar el ID del cluster actual a todos los canales descubiertos en este viaje
            clusters[cluster_list] = cluster_id
            
            # Incrementar el identificador para el próximo cluster
            cluster_id += 1
            
    return clusters