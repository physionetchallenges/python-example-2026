import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def swa_channelNeighbours(eLoc, displayNet=False):
    """
    Searches for neighbouring channels using triangulation and reports back 
    each channel's neighbours.
    Translated from swa-Matlab / ept_TFCE.
    """
    nCh = len(eLoc)
    
    # 1. Extraer coordenadas X, Y, Z de forma flexible (soporta objetos o dicts)
    x = np.array([getattr(el, 'X', el.get('X') if isinstance(el, dict) else None) for el in eLoc], dtype=float)
    y = np.array([getattr(el, 'Y', el.get('Y') if isinstance(el, dict) else None) for el in eLoc], dtype=float)
    z = np.array([getattr(el, 'Z', el.get('Z') if isinstance(el, dict) else None) for el in eLoc], dtype=float)
    
    vertices = np.column_stack((x, y, z))
    
    # 2. Proyección plana 2D (Algoritmo simplificado de Brainstorm)
    z2 = z - np.max(z)
    hypotxy = np.hypot(x, y)
    R = np.hypot(hypotxy, z2)
    PHI = np.arctan2(z2, hypotxy)
    TH = np.arctan2(y, x)
    
    # Prevenir valores excesivamente pequeños para PHI
    PHI[PHI < 0.001] = 0.001
    
    # Proyección achatada (Flat projection)
    R2 = R / (np.cos(PHI) ** 0.2)
    X_2d = R2 * np.cos(TH)
    Y_2d = R2 * np.sin(TH)
    
    # 3. Ajuste de Esfera (Equivalente a bst_bfs / fminsearch)
    mass = np.mean(vertices, axis=0)
    diffvert = vertices - mass
    R0 = np.mean(np.sqrt(np.sum(diffvert**2, axis=1)))
    vec0 = np.append(mass, R0)
    
    # Función de optimización interna para encontrar el centro de la cabeza
    def dist_sph(vec, sensloc):
        radius = vec[-1]
        center = vec[:-1]
        diff = sensloc - center
        return np.mean(np.abs(np.sqrt(np.sum(diff**2, axis=1)) - radius))
        
    res = minimize(dist_sph, vec0, args=(vertices,), method='Nelder-Mead')
    HeadCenter = res.x[:-1]
    
    # Normalización sobre el centro estimado de la cabeza
    coordC = vertices - HeadCenter
    coordC = coordC / np.sqrt(np.sum(coordC**2, axis=1))[:, np.newaxis]
    
    # Teselación (Convex Hull 3D) -> Reemplaza a convhulln
    hull_3d = ConvexHull(coordC)
    faces = hull_3d.simplices  # Matriz de Nx3 (índices de los triángulos)
    
    # 4. Eliminar triángulos innecesarios / externos
    # Obtener el borde exterior en 2D
    points_2d = np.column_stack((X_2d, Y_2d))
    hull_2d = ConvexHull(points_2d)
    border = set(hull_2d.vertices)
    
    # Conservar caras que NO tengan todos sus 3 vértices en el borde exterior
    iInside = ~np.all(np.isin(faces, list(border)), axis=1)
    faces = faces[iInside]
    
    # Calcular perímetros para eliminar triángulos desproporcionados (outliers)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    side0 = np.sqrt(np.sum((v0 - v1)**2, axis=1))
    side1 = np.sqrt(np.sum((v0 - v2)**2, axis=1))
    side2 = np.sqrt(np.sum((v1 - v2)**2, axis=1))
    
    triPerimeter = side0 + side1 + side2
    thresholdPerim = np.mean(triPerimeter) + 3 * np.std(triPerimeter)
    
    # Aplicar umbral de distancia
    faces = faces[triPerimeter <= thresholdPerim]
    
    # 5. Visualización de la Red (Opcional)
    if displayNet:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Dibujar las caras del mallado
        mesh = Poly3DCollection(vertices[faces], alpha=0.6, facecolor=[0.5, 0.5, 0.5], edgecolor='k', linewidths=0.5)
        ax.add_collection3d(mesh)
        
        # Dibujar los electrodos como puntos
        ax.scatter(x, y, z, color='r', s=40, depthshade=False)
        
        ax.set_axis_off()
        ax.set_box_aspect([1,1,1])
        plt.show()
        
    # 6. Buscar vecinos para cada canal
    output = []
    for n in range(nCh):
        # Encontrar filas donde aparezca el canal actual 'n'
        rows_with_n = np.any(faces == n, axis=1)
        # Extraer todos los nodos únicos de esos triángulos
        neighbours = np.unique(faces[rows_with_n])
        
        # OJO: MATLAB guarda el propio nodo dentro de sus vecinos.
        # Convertimos a Base-1 para mantener idéntica paridad con MATLAB
        output.append(neighbours + 1)
        
    # Generar matriz final acolchada con ceros (como cell2mat en MATLAB)
    max_neighbors = max(len(ch) for ch in output)
    ChN = np.zeros((nCh, max_neighbors), dtype=int)
    
    for idx, neighbours in enumerate(output):
        ChN[idx, :len(neighbours)] = neighbours
        
    return ChN