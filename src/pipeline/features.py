import os

import edfio
import joblib
import numpy as np
import pandas as pd

from src.common.channel_utils import normalize_channel_label
from helper_code import HEADERS, PHYSIOLOGICAL_DATA_SUBFOLDER, load_age, load_sex, load_signal_data
from src.ecg_processing import ECG_FEATURE_LENGTH, ECG_FEATURE_NAMES, ECG_KEYWORDS, processECG
from src.eeg_processing import EEG_CHANNEL_SPECS, EEG_FEATURE_LENGTH, EEG_FEATURE_NAMES, processEEG, _get_eeg_aliases
from src.resp_processing import RESP_FEATURE_LENGTH, RESP_FEATURE_NAMES, processResp, _get_resp_alias_groups

from .config import DEFAULT_CSV_PATH, FEATURE_CACHE_FOLDER_NAME, SCRIPT_DIR, TOTAL_PHYSIOLOGICAL_FEATURE_LENGTH


REQUIRED_SIGNAL_ALIASES_CACHE = {}
DEMOGRAPHIC_FEATURE_NAMES = (
    'Age',
    'Sex_Female',
    'Sex_Male',
    'Sex_Unknown',
)
FEATURE_NAME_GROUPS = {
    'demographics': DEMOGRAPHIC_FEATURE_NAMES,
    'resp': tuple(RESP_FEATURE_NAMES),
    'eeg': tuple(EEG_FEATURE_NAMES),
    'ecg': tuple(ECG_FEATURE_NAMES),
}
FEATURE_NAMES = (
    *FEATURE_NAME_GROUPS['demographics'],
    *FEATURE_NAME_GROUPS['resp'],
    *FEATURE_NAME_GROUPS['eeg'],
    *FEATURE_NAME_GROUPS['ecg'],
)


def _get_feature_cache_root(data_folder):
    return os.path.join(SCRIPT_DIR, FEATURE_CACHE_FOLDER_NAME)


def _coerce_feature_vector(features):
    vector = np.asarray(features, dtype=np.float32).reshape(-1)
    vector[~np.isfinite(vector)] = np.nan
    return vector


def _extract_optional_features(extractor, expected_length, *args, **kwargs):
    vector = _coerce_feature_vector(extractor(*args, **kwargs))
    if vector.size != expected_length:
        raise ValueError(
            f"{extractor.__name__} returned {vector.size} features; expected {expected_length}."
        )
    return vector


def _get_physiological_data_file(data_folder, site_id, patient_id, session_id):
    return os.path.join(
        data_folder,
        PHYSIOLOGICAL_DATA_SUBFOLDER,
        site_id,
        f"{patient_id}_ses-{session_id}.edf",
    )
def _get_required_signal_aliases(csv_path):
    normalized_csv_path = os.path.abspath(csv_path)
    required_aliases = REQUIRED_SIGNAL_ALIASES_CACHE.get(normalized_csv_path)
    if required_aliases is not None:
        return required_aliases

    resp_alias_groups = _get_resp_alias_groups(normalized_csv_path)
    eeg_aliases = _get_eeg_aliases(normalized_csv_path)

    required_aliases = set()
    for aliases in resp_alias_groups.values():
        required_aliases.update(aliases)

    for channel_spec in EEG_CHANNEL_SPECS.values():
        required_aliases.update(eeg_aliases.get(normalize_channel_label(channel_spec['direct']), set()))
        required_aliases.update(eeg_aliases.get(normalize_channel_label(channel_spec['positive']), set()))
        required_aliases.update(eeg_aliases.get(normalize_channel_label(channel_spec['reference']), set()))

    REQUIRED_SIGNAL_ALIASES_CACHE[normalized_csv_path] = required_aliases
    return required_aliases


def _load_required_signal_data(edf_path, csv_path):
    required_aliases = _get_required_signal_aliases(csv_path)
    channel_dict = {}
    fs_dict = {}

    try:
        edf = edfio.read_edf(edf_path, lazy_load_data=True)
    except Exception:
        return load_signal_data(edf_path)

    for signal in edf.signals:
        label = signal.label.lower().strip()
        normalized_label = normalize_channel_label(label)
        is_required_signal = normalized_label in required_aliases
        is_ecg_signal = any(keyword in normalized_label for keyword in ECG_KEYWORDS)

        if not is_required_signal and not is_ecg_signal:
            continue

        fs_dict[label] = float(signal.sampling_frequency)
        channel_dict[label] = signal.data

    if channel_dict:
        return channel_dict, fs_dict

    return load_signal_data(edf_path)


def _get_feature_cache_file(data_folder, site_id, patient_id, session_id):
    cache_dir = os.path.join(_get_feature_cache_root(data_folder), site_id)
    return os.path.join(cache_dir, f"{patient_id}_ses-{session_id}.sav")


def get_feature_export_dir(data_folder):
    return os.path.join(_get_feature_cache_root(data_folder), 'exports')


def _get_feature_cache_csv_file(cache_file):
    return f"{os.path.splitext(cache_file)[0]}.csv"


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

    csv_file = _get_feature_cache_csv_file(cache_file)
    temp_csv_file = f"{csv_file}.tmp"
    try:
        pd.DataFrame([payload], columns=get_feature_names()).to_csv(temp_csv_file, index=False)
        os.replace(temp_csv_file, csv_file)
    finally:
        if os.path.exists(temp_csv_file):
            os.remove(temp_csv_file)


def extract_demographic_features(data):
    age_value = data.get(HEADERS['age'])
    try:
        age = float(age_value)
    except (TypeError, ValueError):
        age = np.nan
    if not np.isfinite(age):
        age = np.nan
    age = np.array([age], dtype=np.float32)

    sex = load_sex(data)
    sex_vec = np.zeros(3, dtype=np.float32)
    if sex == 'Female':
        sex_vec[0] = 1
    elif sex == 'Male':
        sex_vec[1] = 1
    else:
        sex_vec[2] = 1

    return np.concatenate([age, sex_vec]).astype(np.float32)


def get_feature_names():
    return FEATURE_NAMES


def get_feature_group_indices(include_demographics=False):
    groups = {}
    start = len(FEATURE_NAME_GROUPS['demographics'])

    if include_demographics:
        demo_indices = np.arange(start, dtype=np.int32)
    else:
        demo_indices = np.array([], dtype=np.int32)

    for group_name in ('resp', 'eeg', 'ecg'):
        group_length = len(FEATURE_NAME_GROUPS[group_name])
        group_indices = np.arange(start, start + group_length, dtype=np.int32)
        if include_demographics:
            groups[group_name] = np.concatenate([demo_indices, group_indices])
        else:
            groups[group_name] = group_indices
        start += group_length

    groups['all'] = np.arange(len(FEATURE_NAMES), dtype=np.int32)
    return groups


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
        resp_features = np.full(RESP_FEATURE_LENGTH, np.nan, dtype=np.float32)

    try:
        eeg_features = _extract_optional_features(
            processEEG,
            EEG_FEATURE_LENGTH,
            physiological_data,
            physiological_fs,
            csv_path=csv_path,
        )
    except Exception:
        eeg_features = np.full(EEG_FEATURE_LENGTH, np.nan, dtype=np.float32)

    try:
        ecg_features = _extract_optional_features(
            processECG,
            ECG_FEATURE_LENGTH,
            physiological_data,
            physiological_fs,
            csv_path=csv_path,
        )
    except Exception:
        ecg_features = np.full(ECG_FEATURE_LENGTH, np.nan, dtype=np.float32)

    return np.hstack([resp_features, eeg_features, ecg_features]).astype(np.float32)


def _compute_record_feature_vector(patient_data, data_folder, site_id, patient_id, session_id, csv_path, require_physiological_data):
    demographic_features = extract_demographic_features(patient_data)
    physiological_data_file = _get_physiological_data_file(
        data_folder,
        site_id,
        patient_id,
        session_id,
    )

    if os.path.exists(physiological_data_file):
        physiological_data, physiological_fs = _load_required_signal_data(physiological_data_file, csv_path)
        physiological_features = extract_extended_physiological_features(
            physiological_data,
            physiological_fs,
            csv_path=csv_path,
        )
    elif require_physiological_data:
        raise FileNotFoundError(f"Missing physiological data for {patient_id}.")
    else:
        physiological_features = np.full(TOTAL_PHYSIOLOGICAL_FEATURE_LENGTH, np.nan, dtype=np.float32)

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