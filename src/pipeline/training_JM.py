import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from xgboost import XGBClassifier, data

from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from helper_code import DEMOGRAPHICS_FILE, HEADERS, find_patients, load_label

from .config import MAX_TRAIN_WORKERS
from .features import get_feature_group_indices, get_feature_names, get_or_create_record_feature_vector


DEFAULT_ENSEMBLE_THRESHOLD = 0.5
ENSEMBLE_MODALITIES = ('resp', 'eeg', 'ecg')
DEFAULT_KNN_NEIGHBORS = 5


def build_training_metadata_cache(patient_data_file):
    metadata = pd.read_csv(patient_data_file)
    demographics_cache = {}
    diagnosis_cache = {}

    for row in metadata.to_dict('records'):
        patient_id = row[HEADERS['bids_folder']]
        session_id = row[HEADERS['session_id']]
        demographics_cache[(patient_id, session_id)] = row
        diagnosis_cache[patient_id] = load_label(row)

    return demographics_cache, diagnosis_cache


def process_training_record(record, data_folder, demographics_cache, diagnosis_cache, csv_path):
    patient_id = record[HEADERS['bids_folder']]
    session_id = record[HEADERS['session_id']]
    site_id = record[HEADERS['site_id']]

    try:
        patient_data = demographics_cache.get((patient_id, session_id), {})
        feature_vector = get_or_create_record_feature_vector(
            record,
            data_folder,
            patient_data,
            csv_path=csv_path,
            require_physiological_data=True,
        )

        label = diagnosis_cache.get(patient_id)
        metadata = {
            'patient_id': patient_id,
            'site_id': site_id,
            'session_id': session_id,
        }

        if label == 0 or label == 1:
            return metadata, feature_vector, label, None

        return metadata, None, None, f"Invalid label for {patient_id}. Skipping..."

    except FileNotFoundError as exc:
        return {
            'patient_id': patient_id,
            'site_id': site_id,
            'session_id': session_id,
        }, None, None, f"{exc} Skipping..."
    except Exception as exc:
        return {
            'patient_id': patient_id,
            'site_id': site_id,
            'session_id': session_id,
        }, None, None, f"Error processing {patient_id}: {exc}"


def _export_feature_matrix_csv(output_path, metadata_rows, labels, feature_matrix, feature_names):
    dataframe = pd.DataFrame(metadata_rows)
    dataframe['label'] = labels
    feature_frame = pd.DataFrame(feature_matrix, columns=feature_names)
    dataframe = pd.concat([dataframe.reset_index(drop=True), feature_frame.reset_index(drop=True)], axis=1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataframe.to_csv(output_path, index=False)






def best_threshold(probabilities, labels):
    thresholds = np.linspace(0, 1, 101)
    best_score = -1.0
    best_value = DEFAULT_ENSEMBLE_THRESHOLD

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(np.int32)
        score = f1_score(labels, predictions, zero_division=0)
        if score > best_score:
            best_score = score
            best_value = float(threshold)

    return best_value


def _build_xgb_model(labels):
    neg = int(np.sum(labels == 0))
    pos = int(np.sum(labels == 1))
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    return XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        eval_metric='auc',
        tree_method='hist',
        enable_categorical=False,
        early_stopping_rounds=20)

def preprocess_multimodal_data( df_model):
    
    #Llamamos al dataset de entrada que contiene las variables calculadas y confirma que no hay IDs duplicados
    df_model2 = df_model.drop_duplicates(subset='ID')
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

def prepare_multimodal_data(dddf, testsize=0.1):

    proc_data=preprocess_multimodal_data(dddf)
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

def _fit_ensembleJM(data):

    models = {}
    y_train = data["y_train"]
    y_test  = data["y_test"]

    probas_test = []

    # ===============================
    # 1. Entrenamiento por modalidad
    # ===============================
    for modality in ['resp', 'eeg', 'ecg']:

        if modality not in data or data[modality] is None:
            continue

        X_train, X_test = data[modality]

        if X_train.shape[1] == 0:
            continue

        model = _build_xgb_model(y_train)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        models[modality] = model

        proba = model.predict_proba(X_test)[:, 1]
        probas_test.append(proba)

    if not probas_test:
        raise ValueError("No hay modalidades válidas para entrenar.")

    # ===============================
    # 2. Ensemble
    # ===============================
    proba_final = np.mean(probas_test, axis=0)

    # ===============================
    # 3. Threshold (mejorado)
    # ===============================
    best_thr = best_threshold(proba_final, y_test)

    return {
        "models": models,
        "threshold": best_thr
    }

from sklearn.model_selection import StratifiedKFold

def _fit_ensemble_with_cv(data, n_splits=5):

    X_train_full = data["X_train"]
    y_train_full = data["y_train"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_probas = []
    oof_labels = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):

        print(f"Fold {fold+1}/{n_splits}")

        probas_fold = []

        for modality in ['resp', 'eeg', 'ecg']:

            X_train_mod, _ = data[modality]

            X_tr = X_train_mod.iloc[train_idx]
            X_val = X_train_mod.iloc[val_idx]

            y_tr = y_train_full.iloc[train_idx]
            y_val = y_train_full.iloc[val_idx]

            if X_tr.shape[1] == 0:
                continue

            model = _build_xgb_model(y_tr)

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            proba = model.predict_proba(X_val)[:, 1]
            probas_fold.append(proba)

        if not probas_fold:
            continue

        # Ensemble por fold
        proba_ensemble = np.mean(probas_fold, axis=0)

        oof_probas.extend(proba_ensemble)
        oof_labels.extend(y_val)

    oof_probas = np.array(oof_probas)
    oof_labels = np.array(oof_labels)

    # Threshold robusto (sin leakage)
    best_thr = best_threshold(oof_probas, oof_labels)

    # ===============================
    # Entrenar modelos finales (full data)
    # ===============================
    final_models = {}

    for modality in ['resp', 'eeg', 'ecg']:

        X_train_mod, X_test_mod = data[modality]

        if X_train_mod.shape[1] == 0:
            continue

        model = _build_xgb_model(y_train_full)

        model.fit(
            X_train_mod,
            y_train_full,
            eval_set=[(X_test_mod, data["y_test"])],
            verbose=False
        )

        final_models[modality] = model

    return {
        "models": final_models,
        "threshold": best_thr
    }

def train_multimodal_ensemble(data_folder, verbose, csv_path, export_folder=None):

    # ===============================
    # 1. CARGA Y EXTRACCIÓN
    # ===============================
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)

    demographics_cache, diagnosis_cache = build_training_metadata_cache(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    features = []
    labels = []
    metadata_rows = []

    with ThreadPoolExecutor(max_workers=MAX_TRAIN_WORKERS) as executor:
        results = executor.map(
            lambda record: process_training_record(
                record,
                data_folder,
                demographics_cache,
                diagnosis_cache,
                csv_path,
            ),
            patient_metadata_list,
        )

        pbar = tqdm(results, total=num_records, desc='Extracting Features',
                    unit='record', disable=not verbose)

        for metadata, feature_vector, label, message in pbar:

            if verbose:
                pbar.set_postfix({'patient': metadata['patient_id']})

            if message is not None:
                tqdm.write(f"  ! {message}")
                continue

            features.append(feature_vector)
            labels.append(label)
            metadata_rows.append(metadata)

        pbar.close()

    # ===============================
    # 2. VALIDACIÓN
    # ===============================
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    if features.size == 0 or features.ndim != 2 or features.shape[0] == 0:
        raise ValueError('No valid training samples were extracted.')

    feature_names = list(get_feature_names())

    # ===============================
    # 3. EXPORT RAW (opcional)
    # ===============================
    export_root = export_folder or os.path.join(os.getcwd(), 'feature_exports')
    raw_feature_export_path = os.path.join(export_root, 'training_features_raw.csv')

    _export_feature_matrix_csv(
        raw_feature_export_path,
        metadata_rows,
        labels,
        features,
        feature_names
    )

     # ===============================
    # 2. CONVERTIR A DATAFRAME 
    df = pd.DataFrame(features, columns=feature_names)
    df["label"] = labels

    data = prepare_multimodal_data(df, testsize=0.1)
    preprocessor=data["preprocessor"]

    models, threshold = _fit_ensembleJM(data)
    #result = _fit_ensemble_with_cv(data, n_splits=5)
    #models = result["models"]
    #threshold = result["threshold"]

    return {
        'type': 'multimodal_xgb_ensemble',
        'threshold': threshold,
        'feature_names': feature_names,
        'models': models,
        'preprocessor': preprocessor,
        'feature_exports': {
            'raw': raw_feature_export_path,
        },
    }