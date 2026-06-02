import scipy.io as sio
import numpy as np

def mat_to_dict(mat_element):
    """
    Convierte de forma recursiva estructuras de MATLAB (mat-objects) 
    en diccionarios nativos de Python, limpiando las dimensiones vacías.
    """
    if isinstance(mat_element, sio.matlab.mat_struct):
        # Es una estructura de MATLAB -> La convertimos a diccionario
        return {field: mat_to_dict(getattr(mat_element, field)) for field in mat_element._fieldnames}
    
    elif isinstance(mat_element, np.ndarray):
        # Si es un array de objetos, seguimos buscando estructuras dentro
        if mat_element.dtype == object:
            return [mat_to_dict(element) for element in mat_element]
        # Si es un array normal pero está envuelto en dimensiones extra (ej: [[valor]])
        elif mat_element.ndim <= 2 and mat_element.size == 1:
            return mat_element.item()
        else:
            return mat_element
            
    return mat_element

