import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier

from helper_code import DEMOGRAPHICS_FILE, HEADERS, find_patients, load_label

from .config import MAX_TRAIN_WORKERS
from .features import get_or_create_record_feature_vector


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


def train_xgb_model(data_folder, verbose, csv_path):
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

    neg = int(np.sum(labels == 0))
    pos = int(np.sum(labels == 1))
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        eval_metric='auc',
        tree_method='hist',
    )
    model.fit(features, labels)
    return model