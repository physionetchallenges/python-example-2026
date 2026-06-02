import os
import numpy as np
import scipy.io as sio
import tkinter as tk
from tkinter import filedialog

def swa_saveOutput(Data, Info, SW, save_name=None, flag_raw=True, flag_filtered=False, wave_type='SW'):
    """
    Saves the wave detection output.
    Translated from swa-Matlab.

    Parameters:
    -----------
    Data : dict
        Dictionary containing EEG data.
    Info : dict
        Configuration dictionary.
    SW : list of dicts
        List containing all detected waves.
    save_name : str, optional
        File path to save the output. If None, opens a UI dialog.
    flag_raw : bool
        If True, drops raw data from memory and saves only the file pointer.
    flag_filtered : bool
        If True, saves filtered data to an external binary file to save space.
    wave_type : str
        The key under which the wave structure will be saved ('SW', 'SS', 'ST').
    """
    # 1. Manejo de los Datos Crudos (Raw Data)
    if flag_raw:
        # Reemplaza los datos por un puntero al archivo si el espacio es una preocupación
        if 'Recording' in Info and 'dataFile' in Info['Recording']:
            Data['Raw'] = Info['Recording']['dataFile']
    else:
        print("Information: Will save all the data into the new file; this may take some time and memory.")

    # 2. Manejo de los Datos Filtrados
    if flag_filtered:
        if 'Recording' in Info and 'dataFile' in Info['Recording']:
            base_name = os.path.splitext(Info['Recording']['dataFile'])[0]
            filtered_name = f"{base_name}_filtered.fdt"
            
            if not os.path.exists(filtered_name) and 'Filtered' in Data:
                # Guardar los datos filtrados en un binario plano (equivalente a .fdt de MATLAB)
                # Se asume que Data['Filtered'] es un numpy array
                Data['Filtered'].astype(np.float32).tofile(filtered_name)
                print(f"Calculation: Filtered data saved to {filtered_name}")
            
            Data['Filtered'] = filtered_name
            
    elif 'Filtered' in Data:
        # Eliminar el campo si existe y no queremos guardarlo
        del Data['Filtered']

    # 3. Interfaz de usuario para nombrar el archivo si no se proporcionó uno
    if save_name is None or save_name.strip() == '':
        root = tk.Tk()
        root.withdraw() # Ocultar la ventana principal de tkinter
        save_name = filedialog.asksaveasfilename(
            title="Save Output File",
            defaultextension=".mat",
            filetypes=[("MATLAB files", "*.mat"), ("All files", "*.*")]
        )
        if not save_name:
            print("Warning: Save cancelled by user.")
            return

    # 4. Preparar el diccionario final para guardar
    save_dict = {
        'Data': Data,
        'Info': Info,
        wave_type: SW  # Usamos la variable wave_type ('SW', 'SS' o 'ST') como llave
    }

    # 5. Guardar el archivo
    try:
        # Intenta guardarlo usando scipy.io
        sio.savemat(save_name, save_dict)
        print(f"Success: Output successfully saved to {save_name}")
    except OverflowError:
        # Si el archivo es muy grande (> 2GB), scipy fallará al tratar de imitar mat v7.2.
        # Fallback a un archivo Pickle nativo de Python
        import pickle
        pickle_name = os.path.splitext(save_name)[0] + '.pkl'
        print("Warning: File too large for standard MAT format. Saving as Python Pickle (.pkl) instead.")
        with open(pickle_name, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Success: Output successfully saved to {pickle_name}")