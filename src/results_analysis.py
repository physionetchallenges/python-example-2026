import numpy as np
import pandas as pd
import sys
import os
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.lib import EEG_functions
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

hospital = ["I0006","I0002","I0004","I0007", "S0001"]
results = pd.DataFrame()
for h in hospital:
    results = pd.concat([results, pd.read_csv(f"results_summaryEEG_{h}.csv")], ignore_index=True)

demographics = pd.read_csv(os.path.join('C:/BSICoS/CincChallenge2026/CincChallenge_2026/data/training_set', "demographics.csv"))
demographics = pd.concat([demographics, pd.read_csv(os.path.join('C:/BSICoS/CincChallenge2026/CincChallenge_2026/data/supplementary_set', "demographics.csv"))], ignore_index=True)

for index, row in results.iterrows():
    patient_id = row['Patient_ID']
    hospital_id = row['File'][4:9]  # Asumiendo que los primeros 5 caracteres del nombre del archivo indican el hospital
    demographics_row = demographics[(demographics['BDSPPatientID'] == patient_id) & (demographics['SiteID'] == hospital_id)]
    if not demographics_row.empty:
        cognitive_impairment = demographics_row['Cognitive_Impairment'].values[0]
        time_to_event = demographics_row['Time_to_Event'].values[0]
        results.at[index, 'Hospital'] = hospital_id
        results.at[index, 'CognitiveImpairment'] = cognitive_impairment
        results.at[index, 'Time_to_Event'] = time_to_event
    else:
        results.at[index, 'Hospital'] = hospital_id
        results.at[index, 'CognitiveImpairment'] = np.nan  # O cualquier valor que indique que no se encontró información
        results.at[index, 'Time_to_Event'] = np.nan

df = pd.DataFrame(results)
# Agrupar por electrodo
for elec in results['Channel'].unique():
    subset = results[results['Channel'] == elec]

    print(subset.Hospital.unique())
    # Hacer un boxplot de cada característica que separe entre pacientes con congnitive impairment y sin él
    for col in subset.columns[3:-2]:
        print(col)

        fig = px.box(subset, 
                    x='CognitiveImpairment', 
                    y=col, 
                    color='CognitiveImpairment',
                    notched=True,
                    points="all", 
                    hover_data=['Patient_ID', 'Channel'],
                    title=f"{elec} - Comparativa de {col} según Estado Cognitivo")

        fig.update_layout(template="plotly_white")
        fig.write_html(f"graphs/ComparativaCognitiveImpairment/2segundos/PorHospital/html/{elec}_{col}.html")  # Guardar como HTML para visualización interactiva
        # fig.delete_traces([0])  # Eliminar la leyenda para que no se repita en cada gráfico

        # Generar el boxplot con Seaborn (Extremadamente rápido)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='CognitiveImpairment', y=col, hue='CognitiveImpairment', notch=True)
        sns.stripplot(data=df, x='CognitiveImpairment', y=col, color="black", alpha=0.3, size=3) # Equivalente a points="all"

        plt.title(f"Comparativa {elec} - {col}")
        plt.savefig(f"graphs/ComparativaCognitiveImpairment/2segundos/PorHospital/{hospital_id}/png/{elec}_{col}.png", dpi=100)
                    #  \graphs\ComparativaCognitiveImpairment\2segundos\PorHospital\I0006\png
        plt.close() # ¡Importante! Para no saturar la memoria RAM