#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

import os
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

try:
    import joblib
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "No module named 'joblib'. This usually means you're running a different Python interpreter than the one you installed packages into.\n\n"
        f"Active interpreter: {sys.executable}\n\n"
        "Fix by installing dependencies into *this* interpreter, e.g.:\n"
        f"  {sys.executable} -m pip install -r requirements.txt\n"
    ) from e

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from helper_code import (
    ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
    DEMOGRAPHICS_FILE,
    HEADERS,
    PHYSIOLOGICAL_DATA_SUBFOLDER,
    derive_bipolar_signal,
    find_patients,
    load_rename_rules,
    load_signal_data,
    standardize_channel_names_rename_only,
)

################################################################################
#
# Configuration
#
################################################################################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, "channel_table.csv")
DEFAULT_RANDOM_STATE = 56
MAX_SIGNAL_SAMPLES = 120_000
SPECTRAL_MIN_SAMPLES = 128
MODEL_THRESHOLD = 0.5
USE_PHYSIOLOGY_FEATURES = True
MAX_CV_SPLITS = 5
MODEL_SELECTION_TOLERANCE = 0.01
ENABLED_PHYSIOLOGY_MODALITIES = ("eeg", "eog", "ecg", "resp", "spo2")

MODALITY_CANDIDATES = (
    ("eeg", ("f3-m2", "f4-m1", "c3-m2", "c4-m1", "o1-m2", "o2-m1")),
    ("eog", ("e1-m2", "e2-m1")),
    ("chin", ("chin1-chin2", "chin")),
    ("leg", ("lat", "rat")),
    ("ecg", ("ecg", "ekg", "hr")),
    ("resp", ("airflow", "ptaf", "abd", "chest")),
    ("spo2", ("spo2", "sao2")),
)

BIPOLAR_CONFIGS = (
    ("f3-m2", "f3", ("m2",)),
    ("f4-m1", "f4", ("m1",)),
    ("c3-m2", "c3", ("m2",)),
    ("c4-m1", "c4", ("m1",)),
    ("o1-m2", "o1", ("m2",)),
    ("o2-m1", "o2", ("m1",)),
    ("e1-m2", "e1", ("m2",)),
    ("e2-m1", "e2", ("m1",)),
    ("chin1-chin2", "chin 1", ("chin 2",)),
    ("lat", "lleg+", ("lleg-",)),
    ("rat", "rleg+", ("rleg-",)),
)

DEMOGRAPHIC_FEATURE_DIM = 15
SIGNAL_SUMMARY_DIM = 15
ACTIVE_MODALITY_CANDIDATES = tuple(
    (name, candidates) for name, candidates in MODALITY_CANDIDATES if name in ENABLED_PHYSIOLOGY_MODALITIES
)
PHYSIOLOGICAL_FEATURE_DIM = len(ACTIVE_MODALITY_CANDIDATES) * SIGNAL_SUMMARY_DIM if USE_PHYSIOLOGY_FEATURES else 0
ALGORITHMIC_FEATURE_DIM = 18


################################################################################
#
# Required functions
#
################################################################################

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    if verbose:
        print("Finding the Challenge data...")

    demographics_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    metadata_df = pd.read_csv(demographics_file)
    patient_metadata_list = find_patients(demographics_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError("No data were provided.")

    metadata_lookup = build_metadata_lookup(metadata_df)

    if verbose:
        print("Extracting features and labels from the data...")

    features = []
    labels = []
    groups = []

    iterator = tqdm(patient_metadata_list, desc="Extracting Features", unit="record", disable=not verbose)
    for record in iterator:
        patient_id = normalize_identifier(record.get(HEADERS["bids_folder"]))
        site_id = normalize_identifier(record.get(HEADERS["site_id"]))
        session_id = normalize_identifier(record.get(HEADERS["session_id"]))

        if verbose:
            iterator.set_postfix({"patient": patient_id})

        try:
            metadata = metadata_lookup.get(make_record_key(site_id, patient_id, session_id), {})
            feature_vector = build_feature_vector(
                metadata=metadata,
                data_folder=data_folder,
                site_id=site_id,
                patient_id=patient_id,
                session_id=session_id,
                csv_path=csv_path,
            )
            label = extract_label(metadata)

            if label in (0, 1):
                features.append(feature_vector)
                labels.append(label)
                groups.append(make_group_id(site_id, patient_id))
        except Exception as exc:
            tqdm.write(f"  !!! Error processing record {patient_id} (session {session_id}): {exc}")
            continue

    if len(features) == 0:
        raise ValueError(
            "No usable training examples were found. "
            "Check that demographics.csv is present and contains valid labels."
        )

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int8)
    groups = np.asarray(groups, dtype=object)

    if verbose:
        print("Selecting and training the model...")

    model_name, model, cv_auc, threshold = select_and_fit_model(features, labels, groups, verbose)

    artifact = {
        "model": model,
        "model_name": model_name,
        "cv_auc": cv_auc,
        "threshold": threshold,
        "feature_dim": int(features.shape[1]),
        "feature_version": 2,
    }

    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, artifact)

    if verbose:
        if cv_auc is None:
            print(f"Selected model: {model_name}")
        else:
            print(f"Selected model: {model_name} (mean CV AUROC {cv_auc:.4f})")
        print(f"Decision threshold: {threshold:.3f}")
        print("Done.")
        print()


def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, "model.sav")
    return joblib.load(model_filename)


def run_model(model, record, data_folder, verbose):
    artifact = model if isinstance(model, dict) else {"model": model, "threshold": MODEL_THRESHOLD}
    estimator = artifact.get("model", artifact)
    threshold = float(artifact.get("threshold", MODEL_THRESHOLD))

    patient_id = normalize_identifier(record.get(HEADERS["bids_folder"]))
    site_id = normalize_identifier(record.get(HEADERS["site_id"]))
    session_id = normalize_identifier(record.get(HEADERS["session_id"]))

    demographics_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    metadata = load_metadata_row(demographics_file, site_id, patient_id, session_id)

    features = build_feature_vector(
        metadata=metadata,
        data_folder=data_folder,
        site_id=site_id,
        patient_id=patient_id,
        session_id=session_id,
    ).reshape(1, -1)

    binary_output, probability_output = predict_binary_and_probability(estimator, features, threshold)
    return binary_output, probability_output


################################################################################
#
# Feature engineering
#
################################################################################

def build_feature_vector(metadata, data_folder, site_id, patient_id, session_id, csv_path=DEFAULT_CSV_PATH):
    demographic_features = extract_demographic_features(metadata)

    if USE_PHYSIOLOGY_FEATURES:
        physiological_data_file = os.path.join(
            data_folder,
            PHYSIOLOGICAL_DATA_SUBFOLDER,
            str(site_id),
            f"{patient_id}_ses-{session_id}.edf",
        )
        if os.path.exists(physiological_data_file):
            physiological_data, physiological_fs = load_signal_data(physiological_data_file)
            physiological_features = extract_physiological_features(
                physiological_data,
                physiological_fs,
                csv_path=csv_path,
            )
        else:
            physiological_features = np.zeros(PHYSIOLOGICAL_FEATURE_DIM, dtype=np.float32)
    else:
        physiological_features = np.zeros(PHYSIOLOGICAL_FEATURE_DIM, dtype=np.float32)

    algorithmic_annotations_file = os.path.join(
        data_folder,
        ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
        str(site_id),
        f"{patient_id}_ses-{session_id}_caisr_annotations.edf",
    )
    if os.path.exists(algorithmic_annotations_file):
        algorithmic_annotations, _ = load_signal_data(algorithmic_annotations_file)
        algorithmic_features = extract_algorithmic_annotations_features(algorithmic_annotations)
    else:
        algorithmic_features = np.zeros(ALGORITHMIC_FEATURE_DIM, dtype=np.float32)

    feature_vector = np.concatenate([demographic_features, physiological_features, algorithmic_features])
    return sanitize_feature_vector(feature_vector)


def extract_demographic_features(data):
    age = safe_float(data.get(HEADERS["age"]))
    bmi = safe_float(data.get(HEADERS["bmi"]))

    sex_vec = np.zeros(3, dtype=np.float32)
    sex = standardize_sex(data.get(HEADERS["sex"]))
    sex_vec[{"female": 0, "male": 1}.get(sex, 2)] = 1.0

    race_vec = np.zeros(5, dtype=np.float32)
    race = standardize_race(data.get(HEADERS["race"]))
    race_vec[{"asian": 0, "black": 1, "other": 2, "unavailable": 3, "white": 4}[race]] = 1.0

    ethnicity_vec = np.zeros(3, dtype=np.float32)
    ethnicity = standardize_ethnicity(data.get(HEADERS["ethnicity"]))
    ethnicity_vec[{"hispanic": 0, "not_hispanic": 1, "unavailable": 2}[ethnicity]] = 1.0

    features = np.array(
        [
            age,
            bmi,
            float(np.isnan(age)),
            float(np.isnan(bmi)),
        ],
        dtype=np.float32,
    )
    features[0] = 0.0 if np.isnan(features[0]) else features[0]
    features[1] = 0.0 if np.isnan(features[1]) else features[1]

    return np.concatenate([features, sex_vec, race_vec, ethnicity_vec]).astype(np.float32)


def extract_physiological_features(physiological_data, physiological_fs, csv_path=DEFAULT_CSV_PATH):
    if not physiological_data:
        return np.zeros(PHYSIOLOGICAL_FEATURE_DIM, dtype=np.float32)

    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    original_labels = list(physiological_data.keys())
    rename_map, cols_to_drop = standardize_channel_names_rename_only(original_labels, rename_rules)

    processed_channels = {}
    processed_fs = {}

    for old_label, signal in physiological_data.items():
        if old_label in cols_to_drop:
            continue

        if old_label not in physiological_fs:
            continue

        new_label = rename_map.get(old_label, old_label.lower())
        processed_channels[new_label] = np.asarray(signal, dtype=np.float64)
        processed_fs[new_label] = float(physiological_fs[old_label])

    for target, pos_label, neg_labels in BIPOLAR_CONFIGS:
        if target in processed_channels or pos_label not in processed_channels:
            continue
        if not all(label in processed_channels for label in neg_labels):
            continue

        fs_values = [processed_fs.get(pos_label)] + [processed_fs.get(label) for label in neg_labels]
        if any(value is None for value in fs_values):
            continue
        if len({float(value) for value in fs_values}) > 1:
            continue

        reference = processed_channels[neg_labels[0]] if len(neg_labels) == 1 else tuple(processed_channels[label] for label in neg_labels)
        derived_signal = derive_bipolar_signal(processed_channels[pos_label], reference)
        if derived_signal is None:
            continue

        processed_channels[target] = np.asarray(derived_signal, dtype=np.float64)
        processed_fs[target] = float(fs_values[0])

    features = []
    for _, candidates in ACTIVE_MODALITY_CANDIDATES:
        summaries = []
        for candidate in candidates:
            if candidate not in processed_channels:
                continue
            summary = summarize_signal(processed_channels[candidate], processed_fs.get(candidate))
            if np.any(summary):
                summaries.append(summary)

        if summaries:
            modality_summary = np.mean(np.vstack(summaries), axis=0)
            availability = min(len(summaries), 3) / 3.0
            features.extend(modality_summary.tolist())
            features.append(float(availability))
        else:
            features.extend([0.0] * (SIGNAL_SUMMARY_DIM - 1))
            features.append(0.0)

    return sanitize_feature_vector(np.asarray(features, dtype=np.float32))


def summarize_signal(signal, fs):
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]

    if values.size < 2:
        return np.zeros(SIGNAL_SUMMARY_DIM - 1, dtype=np.float32)

    step = max(1, int(np.ceil(values.size / MAX_SIGNAL_SAMPLES)))
    sampled = values[::step]
    effective_fs = float(fs) / step if fs not in (None, 0) else 0.0

    median = np.median(sampled)
    centered = sampled - median
    diff = np.diff(centered)

    q05, q25, q50, q75, q95 = np.percentile(sampled, [5, 25, 50, 75, 95])
    activity = np.var(centered)
    diff_var = np.var(diff) if diff.size else 0.0
    diff2 = np.diff(diff)
    diff2_var = np.var(diff2) if diff2.size else 0.0
    mobility = np.sqrt(diff_var / activity) if activity > 0 and diff_var > 0 else 0.0
    complexity = np.sqrt(diff2_var / diff_var) / mobility if diff_var > 0 and mobility > 0 else 0.0

    zcr = np.mean((centered[:-1] * centered[1:]) < 0) if centered.size > 1 else 0.0
    line_length = np.mean(np.abs(diff)) if diff.size else 0.0
    mad = np.median(np.abs(sampled - q50))
    rms = np.sqrt(np.mean(sampled ** 2))

    dominant_freq = 0.0
    spectral_entropy = 0.0
    if effective_fs > 0 and sampled.size >= SPECTRAL_MIN_SAMPLES:
        try:
            freqs, power = welch(centered, fs=effective_fs, nperseg=min(1024, sampled.size))
            power = np.asarray(power, dtype=np.float64)
            power = np.maximum(power, 0.0)
            total_power = power.sum()
            if total_power > 0 and freqs.size > 0:
                dominant_freq = float(freqs[np.argmax(power)])
                normalized_power = power / total_power
                nz = normalized_power > 0
                spectral_entropy = float(
                    -(normalized_power[nz] * np.log(normalized_power[nz])).sum() / np.log(normalized_power.size)
                )
        except Exception:
            dominant_freq = 0.0
            spectral_entropy = 0.0

    summary = np.array(
        [
            np.std(sampled),
            mad,
            q75 - q25,
            np.mean(np.abs(sampled)),
            rms,
            q05,
            q50,
            q95,
            zcr,
            line_length,
            mobility,
            complexity,
            dominant_freq,
            spectral_entropy,
        ],
        dtype=np.float32,
    )
    return sanitize_feature_vector(summary)


def extract_algorithmic_annotations_features(algo_data):
    if not algo_data:
        return np.zeros(ALGORITHMIC_FEATURE_DIM, dtype=np.float32)

    total_seconds = 0
    for key in ("resp_caisr", "arousal_caisr", "limb_caisr", "stage_caisr"):
        if key in algo_data:
            total_seconds = max(total_seconds, len(algo_data[key]))
    total_hours = total_seconds / 3600.0 if total_seconds > 0 else 0.0

    features = []
    for key in ("resp_caisr", "arousal_caisr", "limb_caisr"):
        index_per_hour, mean_duration_minutes = summarize_event_trace(algo_data.get(key), total_hours)
        features.extend([index_per_hour, mean_duration_minutes])

    stages = np.asarray(algo_data.get("stage_caisr", np.array([])), dtype=np.float64)
    valid_stages = stages[np.isfinite(stages) & (stages < 9.0)]
    if valid_stages.size:
        w_pct = np.mean(valid_stages == 5)
        r_pct = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)
        efficiency = np.mean((valid_stages >= 1) & (valid_stages <= 4))
        transitions_per_hour = (
            np.count_nonzero(np.diff(valid_stages)) / total_hours if total_hours > 0 and valid_stages.size > 1 else 0.0
        )
        rem_indices = np.where(valid_stages == 4)[0]
        rem_latency_hours = (float(rem_indices[0]) * 30.0 / 3600.0) if rem_indices.size else 0.0
        stage_distribution = np.array([w_pct, n1_pct, n2_pct, n3_pct, r_pct], dtype=np.float64)
        positive = stage_distribution > 0
        stage_entropy = float(
            -(stage_distribution[positive] * np.log(stage_distribution[positive])).sum() / np.log(stage_distribution.size)
        ) if positive.any() else 0.0
    else:
        w_pct = r_pct = n1_pct = n2_pct = n3_pct = efficiency = transitions_per_hour = rem_latency_hours = stage_entropy = 0.0

    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency, transitions_per_hour, rem_latency_hours, stage_entropy])

    features.extend(
        [
            mean_clean_probability(algo_data.get("caisr_prob_w")),
            mean_clean_probability(algo_data.get("caisr_prob_n3")),
            mean_clean_probability(algo_data.get("caisr_prob_arous")),
        ]
    )

    return sanitize_feature_vector(np.asarray(features, dtype=np.float32))


################################################################################
#
# Model selection and prediction
#
################################################################################

def select_and_fit_model(features, labels, groups, verbose):
    unique_labels, counts = np.unique(labels, return_counts=True)
    if unique_labels.size < 2:
        constant_label = int(unique_labels[0])
        model = DummyClassifier(strategy="constant", constant=constant_label)
        model.fit(features, labels)
        return "constant_dummy", model, None, MODEL_THRESHOLD

    candidate_factories = get_candidate_factories()
    candidate_map = {name: factory for name, factory in candidate_factories}
    best_name = "extra_trees"
    best_score = -np.inf
    best_model = None

    splitters = build_cv_splitters(labels, groups)
    for splitter in splitters:
        splitter_scored = False
        for name, factory in candidate_factories:
            try:
                candidate = factory()
                if splitter["kind"] == "group":
                    scores = evaluate_grouped_auc(candidate, features, labels, groups, splitter["cv"])
                else:
                    scores = cross_val_score(candidate, features, labels, cv=splitter["cv"], scoring="roc_auc", n_jobs=1)
                score = float(np.mean(scores))
                splitter_scored = True
                if verbose:
                    print(f"  {name}: mean CV AUROC {score:.4f} ({splitter['label']})")
            except Exception as exc:
                if verbose:
                    print(f"  {name}: CV failed ({exc})")
                continue

            if is_better_model(name, score, best_name, best_score):
                best_name = name
                best_score = score

        if splitter_scored:
            break

    if best_score == -np.inf:
        best_score = None

    for name, factory in candidate_factories:
        if name == best_name:
            best_model = factory()
            break

    if best_model is None:
        best_model = candidate_factories[0][1]()
        best_name = candidate_factories[0][0]

    threshold = estimate_optimal_threshold(candidate_map[best_name], features, labels, groups, verbose)
    best_model.fit(features, labels)
    return best_name, best_model, best_score, threshold


def get_candidate_factories():
    def extra_trees_base():
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=3,
            min_samples_split=8,
            max_features="sqrt",
            class_weight="balanced",
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=1,
        )

    def random_forest_base():
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=3,
            min_samples_split=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=1,
        )

    def hist_gradient_boosting_base():
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=250,
            max_depth=4,
            max_leaf_nodes=31,
            min_samples_leaf=24,
            l2_regularization=0.5,
            early_stopping=False,
            random_state=DEFAULT_RANDOM_STATE,
        )

    def extra_trees():
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    extra_trees_base(),
                ),
            ]
        )

    def random_forest():
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    random_forest_base(),
                ),
            ]
        )

    def hist_gradient_boosting():
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    hist_gradient_boosting_base(),
                ),
            ]
        )

    def logistic_regression():
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=3000,
                        solver="lbfgs",
                        random_state=DEFAULT_RANDOM_STATE,
                    ),
                ),
            ]
        )

    def voting_ensemble():
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    VotingClassifier(
                        estimators=[
                            ("et", extra_trees_base()),
                            ("rf", random_forest_base()),
                            ("hgb", hist_gradient_boosting_base()),
                            (
                                "lr",
                                Pipeline(
                                    [
                                        ("scaler", StandardScaler()),
                                        (
                                            "model",
                                            LogisticRegression(
                                                C=0.5,
                                                class_weight="balanced",
                                                max_iter=3000,
                                                solver="lbfgs",
                                                random_state=DEFAULT_RANDOM_STATE,
                                            ),
                                        ),
                                    ]
                                ),
                            ),
                        ],
                        voting="soft",
                        weights=[3, 2, 3, 1],
                        n_jobs=1,
                    ),
                ),
            ]
        )

    return [
        ("voting_ensemble", voting_ensemble),
        ("extra_trees", extra_trees),
        ("random_forest", random_forest),
        ("hist_gradient_boosting", hist_gradient_boosting),
        ("logistic_regression", logistic_regression),
    ]


def predict_binary_and_probability(model, features, threshold):
    features = sanitize_feature_vector(np.asarray(features, dtype=np.float32))

    probability_output = 0.0
    classes = getattr(model, "classes_", None)

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
        if probabilities.ndim == 2 and probabilities.shape[0] > 0:
            if probabilities.shape[1] == 1:
                positive_class = int(classes[0]) if classes is not None and len(classes) == 1 else 0
                probability_output = 1.0 if positive_class == 1 else 0.0
            else:
                positive_index = 1
                if classes is not None and 1 in classes:
                    positive_index = list(classes).index(1)
                probability_output = float(probabilities[0, positive_index])
    elif hasattr(model, "decision_function"):
        score = float(np.ravel(model.decision_function(features))[0])
        probability_output = float(1.0 / (1.0 + np.exp(-score)))
    else:
        prediction = np.asarray(model.predict(features)).reshape(-1)
        probability_output = float(prediction[0])

    probability_output = float(np.clip(probability_output, 0.0, 1.0))
    binary_output = bool(probability_output >= threshold)
    return binary_output, probability_output


################################################################################
#
# Metadata and parsing helpers
#
################################################################################

def build_metadata_lookup(metadata_df):
    lookup = {}
    for _, row in metadata_df.iterrows():
        record = row.to_dict()
        site_id = normalize_identifier(record.get(HEADERS["site_id"]))
        patient_id = normalize_identifier(record.get(HEADERS["bids_folder"]))
        session_id = normalize_identifier(record.get(HEADERS["session_id"]))
        lookup[make_record_key(site_id, patient_id, session_id)] = record
    return lookup


def load_metadata_row(demographics_file, site_id, patient_id, session_id):
    metadata_df = pd.read_csv(demographics_file)
    site_series = metadata_df[HEADERS["site_id"]].map(normalize_identifier)
    patient_series = metadata_df[HEADERS["bids_folder"]].map(normalize_identifier)
    session_series = metadata_df[HEADERS["session_id"]].map(normalize_identifier)
    row = metadata_df[
        (site_series == site_id)
        & (patient_series == patient_id)
        & (session_series == session_id)
    ]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def make_record_key(site_id, patient_id, session_id):
    return (normalize_identifier(site_id), normalize_identifier(patient_id), normalize_identifier(session_id))


def make_group_id(site_id, patient_id):
    return f"{normalize_identifier(site_id)}::{normalize_identifier(patient_id)}"


def is_better_model(candidate_name, candidate_score, best_name, best_score):
    if candidate_score > best_score + MODEL_SELECTION_TOLERANCE:
        return True
    if abs(candidate_score - best_score) <= MODEL_SELECTION_TOLERANCE:
        return get_model_priority(candidate_name) < get_model_priority(best_name)
    return False


def get_model_priority(model_name):
    priorities = {
        "logistic_regression": 0,
        "hist_gradient_boosting": 1,
        "voting_ensemble": 2,
        "random_forest": 3,
        "extra_trees": 4,
        "constant_dummy": 5,
    }
    return priorities.get(model_name, 99)


def build_cv_splitters(labels, groups):
    labels = np.asarray(labels)
    groups = np.asarray(groups, dtype=object)

    splitters = []
    unique_groups = np.unique(groups)
    cv_splits = int(min(MAX_CV_SPLITS, np.unique(labels, return_counts=True)[1].min()))

    if unique_groups.size >= 2 and unique_groups.size < labels.size:
        group_splits = int(min(MAX_CV_SPLITS, unique_groups.size))
        if group_splits >= 2:
            splitters.append({
                "kind": "group",
                "cv": GroupKFold(n_splits=group_splits),
                "label": f"grouped {group_splits}-fold CV",
            })

    if cv_splits >= 2:
        splitters.append({
            "kind": "stratified",
            "cv": StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=DEFAULT_RANDOM_STATE),
            "label": f"stratified {cv_splits}-fold CV",
        })

    return splitters


def evaluate_grouped_auc(candidate, features, labels, groups, splitter):
    scores = []
    for train_idx, test_idx in splitter.split(features, labels, groups):
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        if np.unique(train_labels).size < 2 or np.unique(test_labels).size < 2:
            continue

        estimator = clone(candidate)
        estimator.fit(features[train_idx], train_labels)

        if hasattr(estimator, "predict_proba"):
            probabilities = np.asarray(estimator.predict_proba(features[test_idx]), dtype=np.float64)
            if probabilities.ndim != 2 or probabilities.shape[1] < 2:
                continue
            classes = getattr(estimator, "classes_", None)
            positive_index = 1
            if classes is not None and 1 in classes:
                positive_index = list(classes).index(1)
            fold_outputs = probabilities[:, positive_index]
        elif hasattr(estimator, "decision_function"):
            fold_outputs = np.ravel(estimator.decision_function(features[test_idx]))
        else:
            fold_outputs = np.ravel(estimator.predict(features[test_idx]))

        scores.append(float(roc_auc_score(test_labels, fold_outputs)))

    if not scores:
        raise ValueError("Grouped CV produced no valid folds with both classes present.")

    return np.asarray(scores, dtype=np.float64)


def estimate_optimal_threshold(candidate_factory, features, labels, groups, verbose):
    for splitter in build_cv_splitters(labels, groups):
        try:
            candidate = candidate_factory()
            probabilities = generate_oof_probabilities(candidate, features, labels, groups, splitter)
            threshold = choose_decision_threshold(labels, probabilities)
            if verbose:
                print(f"  threshold tuned at {threshold:.3f} using {splitter['label']}")
            return threshold
        except Exception as exc:
            if verbose:
                print(f"  threshold tuning failed ({splitter['label']}: {exc})")

    return MODEL_THRESHOLD


def generate_oof_probabilities(candidate, features, labels, groups, splitter):
    probabilities = np.full(len(labels), np.nan, dtype=np.float64)

    if splitter["kind"] == "group":
        fold_iterator = splitter["cv"].split(features, labels, groups)
    else:
        fold_iterator = splitter["cv"].split(features, labels)

    for train_idx, test_idx in fold_iterator:
        train_labels = labels[train_idx]
        if np.unique(train_labels).size < 2:
            raise ValueError("a training fold has only one class")

        estimator = clone(candidate)
        estimator.fit(features[train_idx], train_labels)
        probabilities[test_idx] = predict_probability_scores(estimator, features[test_idx])

    if np.any(~np.isfinite(probabilities)):
        raise ValueError("OOF probability generation did not cover every sample")

    return probabilities


def choose_decision_threshold(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    quantile_grid = np.quantile(probabilities, np.linspace(0.1, 0.9, 17))
    threshold_grid = np.unique(
        np.clip(
            np.concatenate(([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70], quantile_grid)),
            0.05,
            0.95,
        )
    )

    best_threshold = MODEL_THRESHOLD
    best_score = -np.inf

    for threshold in threshold_grid:
        predictions = (probabilities >= threshold).astype(np.int8)
        score = float(balanced_accuracy_score(labels, predictions))
        if score > best_score + 1e-12:
            best_threshold = float(threshold)
            best_score = score
        elif abs(score - best_score) <= 1e-12 and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5):
            best_threshold = float(threshold)

    return best_threshold


def predict_probability_scores(model, features):
    features = sanitize_feature_vector(np.asarray(features, dtype=np.float32))

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
        if probabilities.ndim == 2 and probabilities.shape[0] > 0:
            if probabilities.shape[1] == 1:
                classes = getattr(model, "classes_", None)
                positive_class = int(classes[0]) if classes is not None and len(classes) == 1 else 0
                score = 1.0 if positive_class == 1 else 0.0
                return np.full(features.shape[0], score, dtype=np.float64)

            classes = getattr(model, "classes_", None)
            positive_index = 1
            if classes is not None and 1 in classes:
                positive_index = list(classes).index(1)
            return np.clip(probabilities[:, positive_index], 0.0, 1.0)

    if hasattr(model, "decision_function"):
        scores = np.ravel(model.decision_function(features))
        return 1.0 / (1.0 + np.exp(-scores))

    predictions = np.ravel(model.predict(features)).astype(np.float64)
    return np.clip(predictions, 0.0, 1.0)


def normalize_identifier(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return str(int(value)) if float(value).is_integer() else str(value)
    return str(value).strip()


def safe_float(value):
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return np.nan
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def extract_label(metadata):
    value = metadata.get(HEADERS["label"])
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "1.0", "t", "y", "yes"}:
            return 1
        if normalized in {"false", "0", "0.0", "f", "n", "no"}:
            return 0
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) in (0, 1) else None
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        value = int(value)
        return value if value in (0, 1) else None
    return None


def standardize_sex(value):
    normalized = str(value).strip().lower()
    if normalized.startswith("f"):
        return "female"
    if normalized.startswith("m"):
        return "male"
    return "unknown"


def standardize_race(value):
    normalized = str(value).strip().lower()
    if any(token in normalized for token in ("white", "caucasian")):
        return "white"
    if any(token in normalized for token in ("black", "african american")):
        return "black"
    if "asian" in normalized:
        return "asian"
    if normalized in {"", "unknown", "unavailable", "declined", "unreported", "nan", "none", "not specified"}:
        return "unavailable"
    return "other"


def standardize_ethnicity(value):
    normalized = str(value).strip().lower()
    if any(token in normalized for token in ("not hispanic", "non-hispanic", "non hispanic", "not latino", "non-latino")):
        return "not_hispanic"
    if "hispanic" in normalized or "latino" in normalized:
        return "hispanic"
    if normalized in {"", "unknown", "unavailable", "declined", "unreported", "nan", "none", "not specified"}:
        return "unavailable"
    return "unavailable"


################################################################################
#
# Annotation and signal utilities
#
################################################################################

def summarize_event_trace(trace, total_hours):
    if trace is None or total_hours <= 0:
        return 0.0, 0.0

    signal = np.asarray(trace, dtype=np.float64).reshape(-1)
    signal = np.where(np.isfinite(signal), signal, 0.0)
    binary = (signal > 0).astype(np.int8)

    if binary.size == 0:
        return 0.0, 0.0

    starts = np.flatnonzero(np.diff(np.pad(binary, (1, 0))) == 1)
    ends = np.flatnonzero(np.diff(np.pad(binary, (0, 1))) == -1)
    durations = (ends - starts).astype(np.float64)

    index_per_hour = float(len(starts) / total_hours)
    mean_duration_minutes = float(np.mean(durations) / 60.0) if durations.size else 0.0
    return index_per_hour, mean_duration_minutes


def mean_clean_probability(values):
    if values is None:
        return 0.0
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr >= 0.0) & (arr <= 1.0)]
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def sanitize_feature_vector(values):
    values = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


################################################################################
#
# Persistence
#
################################################################################

def save_model(model_folder, model_artifact):
    filename = os.path.join(model_folder, "model.sav")
    joblib.dump(model_artifact, filename, protocol=0)
