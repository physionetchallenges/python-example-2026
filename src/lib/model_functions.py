import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, f1_score

def best_threshold(proba_final,y_test):
    thresholds = np.linspace(0, 1, 100)
    best_thr = 0
    best_f1 = 0
    for t in thresholds:
        preds = (proba_final >= t).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
    return best_thr

def evaluar_rendimiento(y_real,y_pred, proba_final, dict_probas, nombre_ensemble="Ensemble"):
    """
    Imprime un reporte completo de métricas de clasificación y AUC.
    
    Args:
        y_real: Etiquetas verdaderas (y_test o y_val).
        proba_final: Probabilidades del modelo combinado.
        dict_probas: Diccionario con formato {'Nombre': lista_probas} para modelos individuales.
        nombre_ensemble: Nombre personalizado para el modelo principal.
    """
    
    # Impresión de métricas principales
    print(f"Reporte de clasificacion {nombre_ensemble}")
    print(classification_report(y_real, y_pred))
    
    # AUC del Ensemble
    roc_auc = roc_auc_score(y_real, proba_final)
    print(f"AUC {nombre_ensemble}: {roc_auc:.2%}")
    print("\n--- Comparativa AUC Modelos Individuales ---")
    
    # 4. AUC de modelos individuales usando el diccionario
    for nombre, probas in dict_probas.items():
        auc_score = roc_auc_score(y_real, probas)
        print(f"AUC {nombre:<8}: {auc_score:.2%}")
    print("-" * 30)
    
def preprocess_multimodal_data(demo, df_model):
    
    #Llamamos al dataset de entrada que contiene las variables calculadas y confirma que no hay IDs duplicados
    df_model = df_model.drop_duplicates(subset='ID')
    demo=demo.drop_duplicates(subset='ID')
    #Haz el merge con las demograficas del paciente, Age, Sex y Label
    # Asegurar tipo string en ID
    for d in [demo, df_model]:
        d['ID'] = d['ID'].astype(str).str.strip() # .strip() por si hay espacios invisibles
        
    df_model2= (demo.merge(df_model, on="ID", how="inner"))

    # Extrae las columnas que se usarán
    df_model2=df_model2[['ID','label', 'Age','Sex','Nasal_Peakedness_Max', 'Nasal_Peakedness_Min', 'Nasal_Peakedness_Median','Nasal_Peakedness_Std', 
                        'Chest_Peakedness_Max', 'Chest_Peakedness_Min','Abdomen_Peakedness_Max','Abdomen_Peakedness_Min','Abdomen_Peakedness_Std',
                        'Flow_Peakedness_Max','Flow_Peakedness_Min','Flow_Peakedness_Median','Flow_Peakedness_Std','SpO2_Max','SpO2_Min','SpO2_Std',
                        'CET90','ODI_Mean','ODI_deepness','C3-M2_Hjorth_Complexity','C4-M1_Hjorth_Complexity','F3-M2_Hjorth_Complexity',
                        'F4-M1_Hjorth_Complexity','C3-M2_Hjorth_Mobility','F3-M2_Hjorth_Mobility','F4-M1_Hjorth_Mobility','C3-M2_Ratio_Slow_Fast',
                        'C4-M1_Ratio_Slow_Fast','F3-M2_Ratio_Slow_Fast','F4-M1_Ratio_Slow_Fast','C3-M2_Rel_Beta','F3-M2_Rel_Beta','F4-M1_Rel_Beta',
                        'C4-M1_Rel_Sigma','F3-M2_Rel_Sigma','C3-M2_Relative_Delta_Power','C4-M1_Relative_Delta_Power','F3-M2_Relative_Delta_Power',
                        'F4-M1_Relative_Delta_Power','C3-M2_Theta_Alpha_Ratio','C4-M1_Theta_Alpha_Ratio','F3-M2_Theta_Alpha_Ratio','F4-M1_Theta_Alpha_Ratio',
                        'C3-M2_Theta_Beta_Ratio','C4-M1_Theta_Beta_Ratio','F3-M2_Theta_Beta_Ratio','F4-M1_Theta_Beta_Ratio','C3-M2_kurtosis_Alpha',
                        'C3-M2_kurtosis_Beta','C4-M1_kurtosis_Beta','F3-M2_kurtosis_Beta','F4-M1_kurtosis_Beta','C3-M2_kurtosis_Delta','C4-M1_kurtosis_Delta',
                        'F3-M2_kurtosis_Delta','F4-M1_kurtosis_Delta','C3-M2_kurtosis_Sigma','C4-M1_kurtosis_Sigma','F3-M2_kurtosis_Sigma','F4-M1_kurtosis_Sigma',
                        'C3-M2_kurtosis_Theta','C4-M1_kurtosis_Theta','F3-M2_kurtosis_Theta','F4-M1_kurtosis_Theta','C3-M2_variability_Delta',
                        'C4-M1_variability_Delta','F3-M2_variability_Delta','F4-M1_variability_Delta','PIP_med','PIP_std','PNNSS_med',
                        'PNNSS_std','AVNN_med','AVNN_std','SDNN_med','SDNN_std','RMSSD_std','HF_med','HF_std','ECTOPIC_med','ECTOPIC_std']]


    # Ajusta el tipo de dato, todas deben ser numericas excepto Sex, Label y ID. Sex y label tendría que ser logical? o categorical?
    df_model2['Sex'] = df_model2['Sex'].astype('category')
    df_model2['label'] = df_model2['label'].astype(int)

    #Unificar los valores faltantes, que sean consistentes
    df_model2 = df_model2.replace([-999, "NA", "null", "None", "nan", ""], np.nan)
    # Eliminar filas con demasiados NaNs (mínimo 50% de datos válidos)
    umbral = int(len(df_model2.columns) * 0.5)
    df_model2 = df_model2.dropna(thresh=umbral)

    #Procesa los datos, imputar nans y estandarizar por la media y desviacion estandar (zscore)
    y = df_model2['label']
    X = df_model2.drop(columns=[ 'label','ID']) # Excluimos ID, label del entrenamiento

    return {"X": X,
            "y": y}

def prepare_multimodal_data(demo, df_model, testsize=0.1):

    proc_data=preprocess_multimodal_data(demo, df_model)
    X = proc_data["X"]
    y  = proc_data["y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=1999, stratify=y)

    cols_categoricas = ['Sex']
    cols_numericas = [c for c in X_train.columns if c not in cols_categoricas]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler())
        ]), cols_numericas),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cols_categoricas)
    ])
    preprocessor.set_output(transform="pandas")

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    resp_mask = np.array([ any(k in col for k in ["Age","Sex","Nasal_Peakedness", "Chest_Peakedness", "Abdomen_Peakedness", "Flow_Peakedness", "SpO2", "CET90", "ODI" ])
        for col in feature_names])
    eeg_mask = np.array([ any(k in col for k in ["Age","Sex","C3-M2", "C4-M1", "F3-M2", "F4-M1"])
        for col in feature_names])
    ecg_mask = np.array([ any(k in col for k in ["Age","Sex","PIP_", "PNNSS_", "AVNN_", "SDNN_", "RMSSD_", "HF_", "ECTOPIC_"])
        for col in feature_names])

    # Subsets finales
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "resp": (X_train.loc[:, resp_mask], X_test.loc[:, resp_mask]),
        "eeg":  (X_train.loc[:, eeg_mask],  X_test.loc[:, eeg_mask]),
        "ecg":  (X_train.loc[:, ecg_mask],  X_test.loc[:, ecg_mask]),
        "preprocessor": preprocessor
    }

def prepare_multimodal_validationdata(demo, df_model, preprocessor):

    #Llamamos al dataset de entrada que contiene las variables calculadas y confirma que no hay IDs duplicados
    df_model = df_model.drop_duplicates(subset='ID')
    demo=demo.drop_duplicates(subset='ID')
    #Haz el merge con las demograficas del paciente, Age, Sex y Label
    # Asegurar tipo string en ID
    for d in [demo, df_model]:
        d['ID'] = d['ID'].astype(str).str.strip() # .strip() por si hay espacios invisibles
        
    df_model2= (demo.merge(df_model, on="ID", how="inner"))

    # Extrae las columnas que se usarán
    columnas_deseadas=['ID', 'Age','Sex','Nasal_Peakedness_Max', 'Nasal_Peakedness_Min', 'Nasal_Peakedness_Median','Nasal_Peakedness_Std', 
                        'Chest_Peakedness_Max', 'Chest_Peakedness_Min','Abdomen_Peakedness_Max','Abdomen_Peakedness_Min','Abdomen_Peakedness_Std',
                        'Flow_Peakedness_Max','Flow_Peakedness_Min','Flow_Peakedness_Median','Flow_Peakedness_Std','SpO2_Max','SpO2_Min','SpO2_Std',
                        'CET90','ODI_Mean','ODI_deepness','C3-M2_Hjorth_Complexity','C4-M1_Hjorth_Complexity','F3-M2_Hjorth_Complexity',
                        'F4-M1_Hjorth_Complexity','C3-M2_Hjorth_Mobility','F3-M2_Hjorth_Mobility','F4-M1_Hjorth_Mobility','C3-M2_Ratio_Slow_Fast',
                        'C4-M1_Ratio_Slow_Fast','F3-M2_Ratio_Slow_Fast','F4-M1_Ratio_Slow_Fast','C3-M2_Rel_Beta','F3-M2_Rel_Beta','F4-M1_Rel_Beta',
                        'C4-M1_Rel_Sigma','F3-M2_Rel_Sigma','C3-M2_Relative_Delta_Power','C4-M1_Relative_Delta_Power','F3-M2_Relative_Delta_Power',
                        'F4-M1_Relative_Delta_Power','C3-M2_Theta_Alpha_Ratio','C4-M1_Theta_Alpha_Ratio','F3-M2_Theta_Alpha_Ratio','F4-M1_Theta_Alpha_Ratio',
                        'C3-M2_Theta_Beta_Ratio','C4-M1_Theta_Beta_Ratio','F3-M2_Theta_Beta_Ratio','F4-M1_Theta_Beta_Ratio','C3-M2_kurtosis_Alpha',
                        'C3-M2_kurtosis_Beta','C4-M1_kurtosis_Beta','F3-M2_kurtosis_Beta','F4-M1_kurtosis_Beta','C3-M2_kurtosis_Delta','C4-M1_kurtosis_Delta',
                        'F3-M2_kurtosis_Delta','F4-M1_kurtosis_Delta','C3-M2_kurtosis_Sigma','C4-M1_kurtosis_Sigma','F3-M2_kurtosis_Sigma','F4-M1_kurtosis_Sigma',
                        'C3-M2_kurtosis_Theta','C4-M1_kurtosis_Theta','F3-M2_kurtosis_Theta','F4-M1_kurtosis_Theta','C3-M2_variability_Delta',
                        'C4-M1_variability_Delta','F3-M2_variability_Delta','F4-M1_variability_Delta','PIP_med','PIP_std','PNNSS_med',
                        'PNNSS_std','AVNN_med','AVNN_std','SDNN_med','SDNN_std','RMSSD_std','HF_med','HF_std','ECTOPIC_med','ECTOPIC_std']
    #Filtrar extraer solo las que existen en el modelo
    columnas_existentes = [col for col in columnas_deseadas if col in df_model2.columns]
    df_model2 = df_model2[columnas_existentes]

    # Ajusta el tipo de dato, todas deben ser numericas excepto Sex, Label y ID. Sex y label tendría que ser logical? o categorical?
    df_model2['Sex'] = df_model2['Sex'].astype('category')
    #df_model2['label'] = df_model2['label'].astype(int)

    #Unificar los valores faltantes, que sean consistentes
    df_model2 = df_model2.replace([-999, "NA", "null", "None", "nan", ""], np.nan)
    # Eliminar filas con demasiados NaNs (mínimo 50% de datos válidos)
    umbral = int(len(df_model2.columns) * 0.5)
    df_model2 = df_model2.dropna(thresh=umbral)

    keys_resp = ["Nasal_Peakedness", "Chest_Peakedness", "Abdomen_Peakedness", "Flow_Peakedness", "SpO2", "CET90", "ODI"]
    keys_eeg  = ["C3-M2", "C4-M1", "F3-M2", "F4-M1"]
    keys_ecg  = ["PIP_", "PNNSS_", "AVNN_", "SDNN_", "RMSSD_", "HF_", "ECTOPIC_"]
    # Identificar qué columnas de 'columnas_deseadas' pertenecen a cada señal
    cols_resp_esperadas = [c for c in columnas_deseadas if any(c.startswith(k) for k in keys_resp)]
    cols_eeg_esperadas  = [c for c in columnas_deseadas if any(c.startswith(k) for k in keys_eeg)]
    cols_ecg_esperadas  = [c for c in columnas_deseadas if any(c.startswith(k) for k in keys_ecg)]

    # Verificación Estricta: ¿Están TODAS en el dataframe de entrada?
    presencia_original = {
        "resp": all(c in df_model2.columns for c in cols_resp_esperadas),
        "eeg":  all(c in df_model2.columns for c in cols_eeg_esperadas),
        "ecg":  all(c in df_model2.columns for c in cols_ecg_esperadas)
    }

    #Procesa los datos, imputar nans y estandarizar por la media y desviacion estandar (zscore)
    #y = df_model2['label']
    #X = df_model2.drop(columns=[ 'label','ID']) # Excluimos ID, label del entrenamiento
    X = df_model2.drop(columns=[ 'ID']) # Excluimos ID, label del entrenamiento
    
    # Reconstrucción técnica (solo para que transform no falle, luego lo filtraremos)
    columnas_que_espera = preprocessor.feature_names_in_
    for col in columnas_que_espera:
        if col not in X.columns:
            X[col] = np.nan
    
    X = X[columnas_que_espera]
    X_val_array = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_validation = pd.DataFrame(X_val_array, columns=feature_names, index=X.index)

    # 6. Extracción Final (Si no estaba completa originalmente, devuelve None)
    def validar_y_extraer(cols_esperadas, existe_completa):
        if not existe_completa:
            return None 

        # Filtramos en los nombres transformados (ej: num__Nasal...)
        cols_senal = [c for c in feature_names if any(k in c for k in cols_esperadas)]
        cols_demo  = [c for c in feature_names if any(d in c for d in ["Age", "Sex"])]
        
        orden_final = [c for c in feature_names if c in (cols_demo + cols_senal)]
        return X_validation[orden_final]

    return { 
        "X_valid": X_validation,        
        "resp": validar_y_extraer(cols_resp_esperadas, presencia_original["resp"]),
        "eeg":  validar_y_extraer(cols_eeg_esperadas,  presencia_original["eeg"]),
        "ecg":  validar_y_extraer(cols_ecg_esperadas,  presencia_original["ecg"])
    }

def reconstruct_multimodal_data(demo, df_model):
    data = prepare_multimodal_data(demo, df_model)
    X_train = data["X_train"]
    X_test  = data["X_test"]
    X_train_resp, X_test_resp = data["resp"]
    X_train_EEG,  X_test_EEG  = data["eeg"]
    X_train_ECG,  X_test_ECG  = data["ecg"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    preprocessor=data["preprocessor"]

    # Configuramos Los parámetros base de los modelos
    est=300 # Número de árboles
    depth=4 # Profundidad (evita overfitting)
    lr=0.05 # Paso de aprendizaje
    ss=0.8 # Usa el 80% de los datos para cada árbol (evita memorizar)
    rs=42  # Para que sea replicable
    categ=False
    met='auc' # Métrica de error interna
    stopi=20 # early stopping si la metrica no mejora en tantas epocas
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    all_xgb = XGBClassifier(scale_pos_weight=scale_pos_weight,n_estimators=est, max_depth=depth,  learning_rate=lr, subsample=ss, random_state=rs, enable_categorical=categ, eval_metric=met, early_stopping_rounds=stopi)
    ECG_xgb = XGBClassifier(scale_pos_weight=scale_pos_weight,n_estimators=est, max_depth=depth,  learning_rate=lr, subsample=ss, random_state=rs, enable_categorical=categ, eval_metric=met, early_stopping_rounds=stopi)
    resp_xgb = XGBClassifier(scale_pos_weight=scale_pos_weight,n_estimators=est, max_depth=depth,  learning_rate=lr, subsample=ss, random_state=rs, enable_categorical=categ, eval_metric=met, early_stopping_rounds=stopi)
    EEG_xgb = XGBClassifier(scale_pos_weight=scale_pos_weight,n_estimators=est, max_depth=depth,  learning_rate=lr, subsample=ss, random_state=rs, enable_categorical=categ, eval_metric=met, early_stopping_rounds=stopi)

    # Entrenamiento
    all_xgb.fit(X_train, y_train,eval_set=[(X_test, y_test)], verbose=False)
    ECG_xgb.fit(X_train_ECG, y_train,eval_set=[(X_test_ECG, y_test)], verbose=False)
    resp_xgb.fit(X_train_resp, y_train,eval_set=[(X_test_resp, y_test)], verbose=False)
    EEG_xgb.fit(X_train_EEG, y_train,eval_set=[(X_test_EEG, y_test)], verbose=False)

    res = prepare_multimodal_validationdata(demo, dfv, preprocessor)

    # Cálculo dinámico de probabilidades
    probas = []
    probas_individuales = {}

    if res["resp"] is not None:
        vproba_resp = resp_xgb.predict_proba(res["resp"])[:, 1]
        probas.append(vproba_resp)
        probas_individuales["RESP"] = vproba_resp

    if res["eeg"] is not None:
        vproba_eeg = EEG_xgb.predict_proba(res["eeg"])[:, 1]
        probas.append(vproba_eeg)
        probas_individuales["EEG"] = vproba_eeg

    if res["ecg"] is not None:
        vproba_ecg = ECG_xgb.predict_proba(res["ecg"])[:, 1]
        probas.append(vproba_ecg)
        probas_individuales["ECG"] = vproba_ecg

    # Promedio de lo que sea que hayamos podido ejecutar
    vproba_final = np.mean(probas, axis=0)
    y_predv =  (vproba_final >= best_thr).astype(int)

    #y_v=res["y_valid"]
    #evaluar_rendimiento(y_v,y_predv,  vproba_final, probas_individuales, nombre_ensemble="Ensemble gboost validation")
