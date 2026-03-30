#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
import atexit
import builtins
import hashlib
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor
from xgboost import XGBClassifier
import sys
from tqdm import tqdm

from helper_code import *
from src.resp_processing import RESP_FEATURE_LENGTH, processResp
from src.eeg_processing import EEG_FEATURE_LENGTH, processEEG
from src.ecg_processing import ECG_FEATURE_LENGTH, processECG
################################################################################
# Path & Constant Configuration (Added for Robustness)
################################################################################

# Get the absolute directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the CSV file relative to the script location
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

# Progress bar state for run_model (initialized lazily)
RUN_MODEL_PBAR = None
RUN_MODEL_PBAR_TOTAL = None
ORIGINAL_PRINT = builtins.print
PRINT_FILTER_ACTIVE = False
RUN_PROGRESS_LINE_RE = re.compile(r'^-\s+\d+/\d+:\s')
RENAME_RULES_CACHE = {}
MAX_TRAIN_WORKERS = max(1, min(4, os.cpu_count() or 1))
FEATURE_CACHE_FOLDER_NAME = '.feature_cache'
TOTAL_PHYSIOLOGICAL_FEATURE_LENGTH = (
    RESP_FEATURE_LENGTH
    + EEG_FEATURE_LENGTH
    + ECG_FEATURE_LENGTH
)


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


def get_rename_rules(csv_path):
    normalized_csv_path = os.path.abspath(csv_path)
    rename_rules = RENAME_RULES_CACHE.get(normalized_csv_path)
    if rename_rules is None:
        rename_rules = load_rename_rules(normalized_csv_path)
        RENAME_RULES_CACHE[normalized_csv_path] = rename_rules
    return rename_rules


def _coerce_feature_vector(features):
    vector = np.asarray(features, dtype=np.float32).reshape(-1)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def _extract_optional_features(extractor, expected_length, *args, **kwargs):
    vector = _coerce_feature_vector(extractor(*args, **kwargs))
    if vector.size != expected_length:
        raise ValueError(
            f"{extractor.__name__} returned {vector.size} features; expected {expected_length}."
        )
    return vector


def _get_record_file_paths(data_folder, site_id, patient_id, session_id):
    physiological_data_file = os.path.join(
        data_folder,
        PHYSIOLOGICAL_DATA_SUBFOLDER,
        site_id,
        f"{patient_id}_ses-{session_id}.edf"
    )
    algorithmic_annotations_file = os.path.join(
        data_folder,
        ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
        site_id,
        f"{patient_id}_ses-{session_id}_caisr_annotations.edf"
    )
    return physiological_data_file, algorithmic_annotations_file


def _get_feature_cache_file(data_folder, site_id, patient_id, session_id):
    folder_hash = hashlib.sha1(os.path.abspath(data_folder).encode('utf-8')).hexdigest()[:12]
    cache_dir = os.path.join(
        SCRIPT_DIR,
        FEATURE_CACHE_FOLDER_NAME,
        folder_hash,
        site_id,
    )
    return os.path.join(cache_dir, f"{patient_id}_ses-{session_id}.sav")


def _load_cached_feature_vector(cache_file):
    if not os.path.exists(cache_file):
        return None

    try:
        payload = joblib.load(cache_file)
    except Exception:
        return None

    if isinstance(payload, dict):
        payload = payload.get('features')

    if payload is None:
        return None

    return _coerce_feature_vector(payload)


def _save_cached_feature_vector(cache_file, feature_vector):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    temp_cache_file = f"{cache_file}.tmp"
    payload = _coerce_feature_vector(feature_vector)

    try:
        joblib.dump(payload, temp_cache_file, protocol=0)
        os.replace(temp_cache_file, cache_file)
    finally:
        if os.path.exists(temp_cache_file):
            os.remove(temp_cache_file)


def _compute_record_feature_vector(patient_data, data_folder, site_id, patient_id, session_id, csv_path, require_physiological_data):
    demographic_features = extract_demographic_features(patient_data)
    physiological_data_file, _ = _get_record_file_paths(
        data_folder,
        site_id,
        patient_id,
        session_id,
    )

    if os.path.exists(physiological_data_file):
        physiological_data, physiological_fs = load_signal_data(physiological_data_file)
        physiological_features = extract_extended_physiological_features(
            physiological_data,
            physiological_fs,
            csv_path=csv_path,
        )
    elif require_physiological_data:
        raise FileNotFoundError(f"Missing physiological data for {patient_id}.")
    else:
        physiological_features = np.zeros(TOTAL_PHYSIOLOGICAL_FEATURE_LENGTH, dtype=np.float32)

    return np.hstack([demographic_features, physiological_features]).astype(np.float32)


def get_or_create_record_feature_vector(record, data_folder, patient_data, csv_path=DEFAULT_CSV_PATH, require_physiological_data=True):
    patient_id = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]
    cache_file = _get_feature_cache_file(data_folder, site_id, patient_id, session_id)
    cached_features = _load_cached_feature_vector(cache_file)
    if cached_features is not None:
        return cached_features

    feature_vector = _compute_record_feature_vector(
        patient_data,
        data_folder,
        site_id,
        patient_id,
        session_id,
        csv_path,
        require_physiological_data,
    )
    _save_cached_feature_vector(cache_file, feature_vector)
    return feature_vector


def extract_extended_physiological_features(physiological_data, physiological_fs, csv_path=DEFAULT_CSV_PATH):
    try:
        resp_features = _extract_optional_features(
            processResp,
            RESP_FEATURE_LENGTH,
            physiological_data,
            physiological_fs,
            csv_path=csv_path,
        )
    except Exception:
        resp_features = np.zeros(RESP_FEATURE_LENGTH, dtype=np.float32)

    try:
        eeg_features = _extract_optional_features(
            processEEG,
            EEG_FEATURE_LENGTH,
            physiological_data,
            physiological_fs,
            csv_path=csv_path,
        )
    except Exception:
        eeg_features = np.zeros(EEG_FEATURE_LENGTH, dtype=np.float32)

    try:
        ecg_features = _extract_optional_features(
            processECG,
            ECG_FEATURE_LENGTH,
            physiological_data,
            physiological_fs,
            csv_path=csv_path,
        )
    except Exception:
        ecg_features = np.zeros(ECG_FEATURE_LENGTH, dtype=np.float32)

    return np.hstack([resp_features, eeg_features, ecg_features]).astype(np.float32)


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

    except FileNotFoundError as e:
        return patient_id, None, None, f"{e} Skipping..."
    except Exception as e:
        return patient_id, None, None, f"Error processing {patient_id}: {e}"


def _close_run_model_pbar():
    global RUN_MODEL_PBAR
    if RUN_MODEL_PBAR is not None:
        RUN_MODEL_PBAR.close()
        RUN_MODEL_PBAR = None


def _install_run_print_filter():
    global PRINT_FILTER_ACTIVE
    if PRINT_FILTER_ACTIVE:
        return

    def _filtered_print(*args, **kwargs):
        message = kwargs.get('sep', ' ').join(str(a) for a in args) if args else ''
        if RUN_PROGRESS_LINE_RE.match(message):
            return
        return ORIGINAL_PRINT(*args, **kwargs)

    builtins.print = _filtered_print
    PRINT_FILTER_ACTIVE = True


def _restore_print():
    global PRINT_FILTER_ACTIVE
    if PRINT_FILTER_ACTIVE:
        builtins.print = ORIGINAL_PRINT
        PRINT_FILTER_ACTIVE = False


atexit.register(_close_run_model_pbar)
atexit.register(_restore_print)


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    demographics_cache, diagnosis_cache = build_training_metadata_cache(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    features = list()
    labels = list()

    with ThreadPoolExecutor(max_workers=MAX_TRAIN_WORKERS) as executor:
        results = executor.map(
            lambda record: process_training_record(
                record,
                data_folder,
                demographics_cache,
                diagnosis_cache,
                csv_path
            ),
            patient_metadata_list
        )

        pbar = tqdm(results, total=num_records, desc="Extracting Features", unit="record", disable=not verbose)
        for patient_id, feature_vector, label, message in pbar:
            if verbose:
                pbar.set_postfix({"patient": patient_id})

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

    # Train the models on the features.
    if verbose:
        print('Training the model on the data...')

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

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    if verbose:
        _install_run_print_filter()

    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(model, record, data_folder, verbose):
    global RUN_MODEL_PBAR, RUN_MODEL_PBAR_TOTAL

    # Load the model.
    model = model['model']

    # Extract identifiers from the record dictionary
    patient_id = record[HEADERS['bids_folder']]
    site_id    = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    # Initialize tqdm progress bar lazily so it advances across run_model calls.
    if verbose and RUN_MODEL_PBAR is None:
        patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
        try:
            RUN_MODEL_PBAR_TOTAL = len(find_patients(patient_data_file))
        except Exception:
            RUN_MODEL_PBAR_TOTAL = None

        RUN_MODEL_PBAR = tqdm(
            total=RUN_MODEL_PBAR_TOTAL,
            desc="Running Model",
            unit="record",
            leave=True,
            file=sys.stdout,
            delay=0.5,
            disable=not verbose
        )

    if verbose and RUN_MODEL_PBAR is not None:
        RUN_MODEL_PBAR.set_postfix({"patient": patient_id})

    # Load the patient data.
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(patient_data_file, patient_id, session_id)
    features = get_or_create_record_feature_vector(
        record,
        data_folder,
        patient_data,
        csv_path=DEFAULT_CSV_PATH,
        require_physiological_data=False,
    ).reshape(1, -1)

    # Get the model outputs.
    binary_output = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    if verbose and RUN_MODEL_PBAR is not None:
        RUN_MODEL_PBAR.update(1)
        if RUN_MODEL_PBAR_TOTAL is not None and RUN_MODEL_PBAR.n >= RUN_MODEL_PBAR_TOTAL:
            RUN_MODEL_PBAR.close()
            RUN_MODEL_PBAR = None

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def extract_demographic_features(data):
    """
    Extracts the demographic subset used by the current XGBoost model.
    
    Inputs:
        data (dict): A dictionary containing patient metadata (e.g., from a CSV row).
    
    Returns:
        np.array: A feature vector of length 4 with age and sex only.
    """
    age = np.array([load_age(data)])

    sex = load_sex(data)
    sex_vec = np.zeros(3)
    if sex == 'Female': 
        sex_vec[0] = 1
    elif sex == 'Male': 
        sex_vec[1] = 1
    else: 
        sex_vec[2] = 1

    return np.concatenate([age, sex_vec])


# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
