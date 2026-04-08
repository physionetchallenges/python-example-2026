import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

from helper_code import DEMOGRAPHICS_FILE, HEADERS, find_patients, load_label

from .config import (
    CV_RANDOM_STATE,
    CV_SEARCH_ITERATIONS,
    DEFAULT_CV_HYPERPARAMETERS,
    MAX_TRAIN_WORKERS,
    OPTIMIZE_HYPERPARAMETER_SEARCH,
    RANDOM_CV_N_SPLITS,
    USE_SITE_GROUPED_CV,
)
from .cross_validation import CrossValidationConfig, EnsembleCrossValidator, normalize_site_group
from .features import get_feature_group_indices, get_feature_names, get_or_create_record_feature_vector


DEFAULT_ENSEMBLE_THRESHOLD = 0.5
ENSEMBLE_MODALITIES = ('resp', 'eeg', 'ecg')
DEFAULT_KNN_NEIGHBORS = 5

# Hyperparameter search space
PARAM_DIST = {
    'max_depth':        [3, 4, 5], 
    'min_child_weight': [1, 2, 3], 
    'subsample':        [0.7, 0.8, 0.9], 
    'colsample_bytree': [0.6, 0.7, 0.8], 
    'reg_lambda':       [0.5, 1.0, 2.0], 
    'reg_alpha':        [0.0, 0.05, 0.1], 
}


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

def prepare_feature_matrix(feature_matrix, preprocessor=None):
    raw_feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
    if raw_feature_matrix.ndim == 1:
        raw_feature_matrix = raw_feature_matrix.reshape(1, -1)
    raw_feature_matrix = raw_feature_matrix.copy()
    raw_feature_matrix[~np.isfinite(raw_feature_matrix)] = np.nan

    if preprocessor is not None:
        processed_feature_matrix = np.asarray(preprocessor.transform(raw_feature_matrix), dtype=np.float32)
    else:
        processed_feature_matrix = raw_feature_matrix

    return raw_feature_matrix, processed_feature_matrix

def export_feature_matrix_csv(output_path, metadata_rows, feature_matrix, feature_names, labels=None):
    dataframe = pd.DataFrame(metadata_rows)
    if labels is not None:
        dataframe['label'] = labels
    feature_frame = pd.DataFrame(feature_matrix, columns=feature_names)
    dataframe = pd.concat([dataframe.reset_index(drop=True), feature_frame.reset_index(drop=True)], axis=1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataframe.to_csv(output_path, index=False)

def get_feature_export_paths(export_root, prefix):
    return {
        'raw': os.path.join(export_root, f'{prefix}_features_raw.csv'),
        'preprocessed': os.path.join(export_root, f'{prefix}_features_preprocessed.csv'),
    }
    
def export_feature_views(export_root, prefix, metadata_rows, feature_matrix, feature_names, preprocessor=None, labels=None):
    raw_feature_matrix, processed_feature_matrix = prepare_feature_matrix(
        feature_matrix,
        preprocessor=preprocessor,
    )
    export_paths = get_feature_export_paths(export_root, prefix)
    export_feature_matrix_csv(
        export_paths['raw'],
        metadata_rows,
        raw_feature_matrix,
        feature_names,
        labels=labels,
    )
    export_feature_matrix_csv(
        export_paths['preprocessed'],
        metadata_rows,
        processed_feature_matrix,
        feature_names,
        labels=labels,
    )
    return export_paths    


def _build_xgb_model(labels, extra_params=None):
   
    neg = int(np.sum(labels == 0))
    pos = int(np.sum(labels == 1))
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
 
    base_params = dict(
        scale_pos_weight=scale_pos_weight,
        n_estimators=500,
        learning_rate=0.05, 
        max_depth=4, 
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=2, 
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='auc',
        tree_method='hist',
    )
    if extra_params:
        base_params.update(extra_params)
 
    return XGBClassifier(**base_params)

def _fit_model(feature_matrix, labels, consensus_params=None):
    model = _build_xgb_model(labels, extra_params=consensus_params)
    model.fit(feature_matrix, labels)
    return model


def _build_search_model(labels):
    return XGBClassifier(
        scale_pos_weight=(int(np.sum(labels == 0)) / max(int(np.sum(labels == 1)), 1)),
        n_estimators=500,
        learning_rate=0.05,
        random_state=CV_RANDOM_STATE,
        eval_metric='auc',
        tree_method='hist',
    )



def _build_preprocessor(num_samples, categorical_indices=None):

    neighbors = min(DEFAULT_KNN_NEIGHBORS, max(1, num_samples - 1)) if num_samples > 1 else 1
 
    if categorical_indices is None or len(categorical_indices) == 0:
        # Original pipeline — no categorical columns
        return Pipeline([
            ('imputer', KNNImputer(n_neighbors=neighbors, keep_empty_features=True)),
            ('scaler', StandardScaler()),
        ])

    return _CategoricalAwarePreprocessor(
        n_neighbors=neighbors,
        categorical_indices=np.asarray(categorical_indices, dtype=np.int32),
    )

class _CategoricalAwarePreprocessor:
 
    def __init__(self, n_neighbors, categorical_indices):
        self.n_neighbors = n_neighbors
        self.categorical_indices = categorical_indices
        self.imputer = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
        self.scaler = StandardScaler()
        self._numerical_indices = None  # set during fit
 
    def _get_numerical_indices(self, n_features):
        all_idx = np.arange(n_features)
        return np.setdiff1d(all_idx, self.categorical_indices)
 
    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float32).copy()
        X[~np.isfinite(X)] = np.nan

        # Step 1: impute everything
        X_imputed = np.asarray(self.imputer.fit_transform(X), dtype=np.float32)
 
        # Step 2: fit scaler on numerical columns only
        self._numerical_indices = self._get_numerical_indices(X.shape[1])
        X_num = X_imputed[:, self._numerical_indices]
        X_num_scaled = np.asarray(self.scaler.fit_transform(X_num), dtype=np.float32)
 
        # Step 3: assemble output — categorical columns keep imputed values
        X_out = X_imputed.copy()
        X_out[:, self._numerical_indices] = X_num_scaled
        return X_out
 
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32).copy()
        X[~np.isfinite(X)] = np.nan
 
        X_imputed = np.asarray(self.imputer.transform(X), dtype=np.float32)
 
        X_num = X_imputed[:, self._numerical_indices]
        X_num_scaled = np.asarray(self.scaler.transform(X_num), dtype=np.float32)
 
        X_out = X_imputed.copy()
        X_out[:, self._numerical_indices] = X_num_scaled
        return X_out

def _fit_ensemble(feature_matrix, labels, feature_indices, consensus_params=None):
    models = {
        'all': _fit_model(
            feature_matrix[:, feature_indices['all']], labels, consensus_params
        ),
    }
    for modality in ENSEMBLE_MODALITIES:
        models[modality] = _fit_model(
            feature_matrix[:, feature_indices[modality]], labels, consensus_params
        )
    return models


def _has_modality_signal(feature_vector, modality_presence_indices):
    modality_values = feature_vector[modality_presence_indices]
    return bool(np.any(np.isfinite(modality_values)))


def predict_ensemble_probabilities(model_bundle, feature_matrix):
    raw_feature_matrix, processed_feature_matrix = prepare_feature_matrix(
        feature_matrix,
        preprocessor=model_bundle.get('preprocessor'),
    )

    models = model_bundle['models']
    feature_indices = {
        name: np.asarray(indices, dtype=np.int32)
        for name, indices in model_bundle['feature_indices'].items()
    }
    modality_presence_indices = {
        name: np.asarray(indices, dtype=np.int32)
        for name, indices in model_bundle['modality_presence_indices'].items()
    }
    
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


def train_multimodal_ensemble(data_folder, verbose, csv_path, export_folder=None):
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

        pbar = tqdm(results, total=num_records, desc='Extracting Features', unit='record', disable=not verbose)
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

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    if features.size == 0 or features.ndim != 2 or features.shape[0] == 0:
        raise ValueError('No valid training samples were extracted. Review feature extraction logs for the skipped records.')

    feature_names = list(get_feature_names())
    feature_indices = get_feature_group_indices(include_demographics=True)
    modality_presence_indices = get_feature_group_indices(include_demographics=False)

    categorical_indices = [
        i for i, name in enumerate(feature_names)
        if name.lower() in ('sex', 'gender')
    ]
    site_groups = np.asarray([
        normalize_site_group(metadata_row['site_id'])
        for metadata_row in metadata_rows
    ])
    cv_config = CrossValidationConfig(
        use_site_grouped_cv=USE_SITE_GROUPED_CV,
        optimize_hyperparameter_search=OPTIMIZE_HYPERPARAMETER_SEARCH,
        outer_random_splits=RANDOM_CV_N_SPLITS,
        random_state=CV_RANDOM_STATE,
        search_iterations=CV_SEARCH_ITERATIONS,
        fixed_hyperparameters=DEFAULT_CV_HYPERPARAMETERS,
    )
    cv_runner = EnsembleCrossValidator(
        config=cv_config,
        param_dist=PARAM_DIST,
        default_threshold=DEFAULT_ENSEMBLE_THRESHOLD,
        build_preprocessor=_build_preprocessor,
        build_search_model=_build_search_model,
        fit_ensemble=_fit_ensemble,
        predict_probabilities=predict_ensemble_probabilities,
    )
    print(f"  Categorical feature indices: {categorical_indices} "
          f"({[feature_names[i] for i in categorical_indices]})")
    print(f"  Hospital CV groups: {sorted(np.unique(site_groups).tolist())}")
    print(f"  CV strategy: {'grouped by hospital' if cv_config.use_site_grouped_cv else 'random stratified folds'}")
    print(f"  Hyperparameter search: {'enabled' if cv_config.optimize_hyperparameter_search else 'disabled'}")
 
    # --- Step 1: Nested CV for threshold calibration and consensus hyperparameters ---
    print("Running nested CV for threshold calibration and hyperparameter consensus...")
    cv_result = cv_runner.run(
        features,
        labels,
        feature_indices,
        modality_presence_indices,
        categorical_indices=categorical_indices if categorical_indices else None,
        site_groups=site_groups,
    )
    threshold = cv_result.threshold
    consensus = cv_result.consensus_params
    cv_metrics = cv_result.metrics
 
    # --- Step 2: Fit final models on ALL data using consensus hyperparameters ---
    print("Fitting final ensemble on all training data with consensus hyperparameters...")
    preprocessor = _build_preprocessor(len(labels), categorical_indices if categorical_indices else None)
    processed_features = np.asarray(preprocessor.fit_transform(features), dtype=np.float32)
    models = _fit_ensemble(processed_features, labels, feature_indices, consensus_params=consensus)

    export_root = export_folder or os.path.join(os.getcwd(), 'feature_exports')
    feature_exports = export_feature_views(
    export_root,
    'training',
    metadata_rows,
    features,
    feature_names,
    preprocessor=preprocessor,
    labels=labels,
    )
      
    return {
        'type': 'multimodal_xgb_ensemble',
        'threshold': threshold,
        'feature_names': feature_names,
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
        'feature_exports': feature_exports,
        'cv_metrics': cv_metrics,
    }
