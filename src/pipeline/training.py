import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

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

        if label == 0 or label == 1:
            return patient_id, feature_vector, label, None

        return patient_id, None, None, f"Invalid label for {patient_id}. Skipping..."

    except FileNotFoundError as exc:
        return patient_id, None, None, f"{exc} Skipping..."
    except Exception as exc:
        return patient_id, None, None, f"Error processing {patient_id}: {exc}"


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
    )


def _fit_model(feature_matrix, labels):
    model = _build_xgb_model(labels)
    model.fit(feature_matrix, labels)
    return model


def _build_preprocessor(num_samples):
    neighbors = min(DEFAULT_KNN_NEIGHBORS, max(1, num_samples - 1)) if num_samples > 1 else 1
    return Pipeline([
        ('imputer', KNNImputer(n_neighbors=neighbors, keep_empty_features=True)),
        ('scaler', StandardScaler()),
    ])


def _fit_ensemble(feature_matrix, labels, feature_indices):
    models = {
        'all': _fit_model(feature_matrix[:, feature_indices['all']], labels),
    }

    for modality in ENSEMBLE_MODALITIES:
        models[modality] = _fit_model(feature_matrix[:, feature_indices[modality]], labels)

    return models


def _has_modality_signal(feature_vector, modality_presence_indices):
    modality_values = feature_vector[modality_presence_indices]
    return bool(np.any(np.isfinite(modality_values)))


def predict_ensemble_probabilities(model_bundle, feature_matrix):
    raw_feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
    if raw_feature_matrix.ndim == 1:
        raw_feature_matrix = raw_feature_matrix.reshape(1, -1)
    raw_feature_matrix = raw_feature_matrix.copy()
    raw_feature_matrix[~np.isfinite(raw_feature_matrix)] = np.nan

    models = model_bundle['models']
    feature_indices = {
        name: np.asarray(indices, dtype=np.int32)
        for name, indices in model_bundle['feature_indices'].items()
    }
    modality_presence_indices = {
        name: np.asarray(indices, dtype=np.int32)
        for name, indices in model_bundle['modality_presence_indices'].items()
    }
    preprocessor = model_bundle.get('preprocessor')
    if preprocessor is not None:
        processed_feature_matrix = np.asarray(preprocessor.transform(raw_feature_matrix), dtype=np.float32)
    else:
        processed_feature_matrix = raw_feature_matrix

    probabilities = np.zeros(raw_feature_matrix.shape[0], dtype=np.float32)
    for row_index, raw_feature_vector in enumerate(raw_feature_matrix):
        processed_feature_vector = processed_feature_matrix[row_index]
        modality_probabilities = []
        for modality in ENSEMBLE_MODALITIES:
            if _has_modality_signal(raw_feature_vector, modality_presence_indices[modality]):
                modality_vector = processed_feature_vector[feature_indices[modality]].reshape(1, -1)
                modality_probability = models[modality].predict_proba(modality_vector)[0][1]
                modality_probabilities.append(float(modality_probability))

        if modality_probabilities:
            probabilities[row_index] = float(np.mean(modality_probabilities))
        else:
            all_features = processed_feature_vector[feature_indices['all']].reshape(1, -1)
            probabilities[row_index] = float(models['all'].predict_proba(all_features)[0][1])

    return probabilities


def predict_ensemble_labels(model_bundle, feature_matrix):
    threshold = float(model_bundle.get('threshold', DEFAULT_ENSEMBLE_THRESHOLD))
    probabilities = predict_ensemble_probabilities(model_bundle, feature_matrix)
    labels = (probabilities >= threshold).astype(np.int32)
    return labels, probabilities


def _calibrate_threshold(feature_matrix, labels, feature_indices, modality_presence_indices):
    classes, class_counts = np.unique(labels, return_counts=True)
    if len(classes) != 2 or np.min(class_counts) < 2 or len(labels) < 10:
        return DEFAULT_ENSEMBLE_THRESHOLD

    try:
        train_features, validation_features, train_labels, validation_labels = train_test_split(
            feature_matrix,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
    except ValueError:
        return DEFAULT_ENSEMBLE_THRESHOLD

    preprocessor = _build_preprocessor(len(train_labels))
    processed_train_features = np.asarray(preprocessor.fit_transform(train_features), dtype=np.float32)

    calibration_bundle = {
        'models': _fit_ensemble(processed_train_features, train_labels, feature_indices),
        'feature_indices': feature_indices,
        'modality_presence_indices': modality_presence_indices,
        'threshold': DEFAULT_ENSEMBLE_THRESHOLD,
        'preprocessor': preprocessor,
    }
    validation_probabilities = predict_ensemble_probabilities(calibration_bundle, validation_features)
    return best_threshold(validation_probabilities, validation_labels)


def train_multimodal_ensemble(data_folder, verbose, csv_path):
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    demographics_cache, diagnosis_cache = build_training_metadata_cache(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    features = []
    labels = []

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

        pbar = tqdm(results, total=num_records, desc='Extracting Features', unit='record', disable=not verbose)
        for patient_id, feature_vector, label, message in pbar:
            if verbose:
                pbar.set_postfix({'patient': patient_id})

            if message is not None:
                tqdm.write(f"  ! {message}")
                continue

            features.append(feature_vector)
            labels.append(label)

        pbar.close()

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    if features.size == 0 or features.ndim != 2 or features.shape[0] == 0:
        raise ValueError('No valid training samples were extracted. Review feature extraction logs for the skipped records.')

    feature_indices = get_feature_group_indices(include_demographics=True)
    modality_presence_indices = get_feature_group_indices(include_demographics=False)
    preprocessor = _build_preprocessor(len(labels))
    processed_features = np.asarray(preprocessor.fit_transform(features), dtype=np.float32)
    threshold = _calibrate_threshold(features, labels, feature_indices, modality_presence_indices)
    models = _fit_ensemble(processed_features, labels, feature_indices)

    return {
        'type': 'multimodal_xgb_ensemble',
        'threshold': threshold,
        'feature_names': list(get_feature_names()),
        'feature_indices': {
            name: indices.tolist()
            for name, indices in feature_indices.items()
            if name in {'all', 'resp', 'eeg', 'ecg'}
        },
        'modality_presence_indices': {
            modality: modality_presence_indices[modality].tolist()
            for modality in ENSEMBLE_MODALITIES
        },
        'models': models,
        'preprocessor': preprocessor,
    }