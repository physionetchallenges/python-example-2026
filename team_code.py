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
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
from tqdm import tqdm

from helper_code import *
from src.resp_processing import processResp
from src.eeg_processing import processEEG
from lib.openECGfunction import openECG
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


def process_training_record(record, data_folder, demographics_cache, diagnosis_cache, csv_path):
    patient_id = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    try:
        patient_data = demographics_cache.get((patient_id, session_id), {})
        demographic_features = extract_demographic_features(patient_data)

        physiological_data_file = os.path.join(
            data_folder,
            PHYSIOLOGICAL_DATA_SUBFOLDER,
            site_id,
            f"{patient_id}_ses-{session_id}.edf"
        )
        if not os.path.exists(physiological_data_file):
            return patient_id, None, None, f"Missing physiological data for {patient_id}. Skipping..."

        physiological_data, physiological_fs = load_signal_data(physiological_data_file)
        physiological_features = extract_physiological_features(
            physiological_data,
            physiological_fs,
            csv_path=csv_path
        )
        resp_features = processResp(
            physiological_data,
            physiological_fs,
            csv_path=csv_path
        )
        eeg_features = processEEG(
            physiological_data,
            physiological_fs,
            csv_path=csv_path
        )
        physiological_features = np.concatenate([physiological_features, resp_features, eeg_features], axis = 1)
        algorithmic_annotations_file = os.path.join(
            data_folder,
            ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
            site_id,
            f"{patient_id}_ses-{session_id}_caisr_annotations.edf"
        )
        algorithmic_annotations, algorithmic_fs = load_signal_data(algorithmic_annotations_file)
        algorithmic_features = extract_algorithmic_annotations_features(algorithmic_annotations)

        label = diagnosis_cache.get(patient_id)

        if label == 0 or label == 1:
            feature_vector = np.hstack([demographic_features, physiological_features, algorithmic_features])
            return patient_id, feature_vector, label, None

        return patient_id, None, None, f"Invalid label for {patient_id}. Skipping..."

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
    labels = np.asarray(labels, dtype=bool)

    # Train the models on the features.
    if verbose:
        print('Training the model on the data...')

    # This very simple model trains a random forest model with very simple features.

    # Define the parameters for the random forest classifier and regressor.
    n_estimators = 12  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    random_state = 56  # Random state; set for reproducibility.

    # Fit the model.
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)

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
    demographic_features = extract_demographic_features(patient_data)

    # Load signal data.
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}.edf")
    if os.path.exists(phys_file):
        phys_data, phys_fs = load_signal_data(phys_file)
        # Ensure csv_path is accessible or defined
        physiological_features = extract_physiological_features(phys_data, phys_fs)
    else:
        # Fallback to zeros if file is missing (length 49)
        physiological_features = np.zeros(49)

    # Load Algorithmic Annotations
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
    if os.path.exists(algo_file):
        algo_data, _ = load_signal_data(algo_file)
        algorithmic_features = extract_algorithmic_annotations_features(algo_data)
    else:
        # Fallback to zeros (length 12)
        algorithmic_features = np.zeros(12)

    features = np.hstack([demographic_features, physiological_features, algorithmic_features]).reshape(1, -1)

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
    Extracts and encodes demographic features from a metadata dictionary.
    
    Inputs:
        data (dict): A dictionary containing patient metadata (e.g., from a CSV row).
    
    Returns:
        np.array: A feature vector of length 11:
            - [0]: Age (Continuous)
            - [1:4]: Sex (One-hot: Female, Male, Other/Unknown)
            - [4:9]: Race (One-hot: Asian, Black, Other, Unavailable, White)
            - [9]: BMI (Continuous)
    """
    # 1. Age Feature (1 dimension)
    # Convert 'Age' to a float; default to 0 if missing
    age = np.array([load_age(data)])

    # 2. Sex One-Hot Encoding (3 dimensions: Female, Male, Other/Unknown)
    # Uses lowercase prefix matching to handle variants like 'F', 'Female', 'M', or 'Male'
    sex = load_sex(data)
    sex_vec = np.zeros(3)
    if sex == 'Female': 
        sex_vec[0] = 1 # Index 0: Female
    elif sex == 'Male': 
        sex_vec[1] = 1 # Index 1: Male
    else: 
        sex_vec[2] = 1 # Index 2: Other/Unknown

    # 3. Race One-Hot Encoding (6 dimensions)
    # Standardizes the raw text into one of six categories using the helper function
    race_category = get_standardized_race(data).lower()
    race_vec = np.zeros(5)
    # Pre-defined mapping for index consistency
    race_mapping = {'asian': 0, 'black': 1, 'others': 2, 'unavailable': 3, 'white': 4}
    race_vec[race_mapping.get(race_category, 2)] = 1

    # 4. Body Mass Index (BMI) Feature (1 dimension)
    # Extracts the pre-calculated mean BMI; handles strings, NaNs, and missing keys
    bmi = np.array([load_bmi(data)])

    # 5. Concatenate all components into a single vector (1 + 3 + 5 + 1 = 10)
    
    return np.concatenate([age, sex_vec, race_vec, bmi])


def extract_physiological_features(physiological_data, physiological_fs, csv_path=DEFAULT_CSV_PATH):
    """
    Standardizes channels and extracts statistical/spectral features.
    """
    original_labels = list(physiological_data.keys())

    # Step 1: Load rules and standardize names
    # Note: Use script-relative path or absolute path for robustness
    rename_rules = get_rename_rules(csv_path)
    rename_map, cols_to_drop = standardize_channel_names_rename_only(original_labels, rename_rules)

    # Step 2: Apply renaming to BOTH signals and their corresponding FS
    processed_channels = {}
    processed_fs = {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        processed_channels[new_label] = data
        # Mapping the sampling rate to the new label
        if old_label in physiological_fs:
            processed_fs[new_label] = physiological_fs[old_label]
        else:
            # Report error and stop if no FS is found for a kept channel
            raise KeyError(f"Sampling frequency (fs) not found for channel '{old_label}' ")
        
    if 'physiological_data' in locals(): del physiological_data

    # Step 3: Construct Bipolar Derivations
    bipolar_configs = [
        ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
        ('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
        ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1']),
        ('e1-m2', 'e1', ['m2']), ('e2-m1', 'e2', ['m1']),
        ('chin1-chin2', 'chin 1', ['chin 2']),
        ('lat', 'lleg+', ['lleg-']), ('rat', 'rleg+', ['rleg-'])
    ]

    for target, pos, neg_list in bipolar_configs:
        # 1. Skip if target already exists or pos channel missing
        if target in processed_channels or pos not in processed_channels:
            continue
        
        # 2. Check all neg channels exist
        if not all(n in processed_channels for n in neg_list):
            continue

        # 3. Check sampling rate consistency
        all_involved = [pos] + neg_list
        fs_values = [processed_fs[ch] for ch in all_involved]
        
        if len(set(fs_values)) > 1:
            raise ValueError(f"Sampling rate mismatch for {target}: {dict(zip(all_involved, fs_values))}")

        # 4. Derive bipolar signal
        ref_sig = processed_channels[neg_list[0]] if len(neg_list) == 1 else tuple(processed_channels[n] for n in neg_list)
        
        derived = derive_bipolar_signal(processed_channels[pos], ref_sig)
        
        if derived is not None:
            processed_channels[target] = derived
            processed_fs[target] = processed_fs[pos]

    leads_to_check = {
        'eeg':  ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1'],
        'eog':  ['e1-m2', 'e2-m1'],
        'chin': ['chin1-chin2', 'chin'],
        'leg':  ['lat', 'rat'],
        'ecg':  ['ecg', 'ekg'],
        'resp': ['airflow', 'ptaf', 'abd', 'chest'],
        'spo2': ['spo2', 'sao2'] # Added sao2 as fallback for spo2
    }
    
    final_features = []
    for lead_type, candidates in leads_to_check.items():
        sig = None
        fs = None
        
        # Identify the first available candidate
        for candidate in candidates:
            if candidate in processed_channels and processed_channels[candidate] is not None:
                sig = processed_channels[candidate]
                fs = processed_fs.get(candidate)
                break 

        # if sig is not None and len(sig) > 0 and fs is not None:
        #     # --- 1. Time Domain Features ---
        #     std_val = np.std(sig)
        #     mav_val = np.mean(np.abs(sig))
        #     energy_val = np.sum(sig**2) / len(sig)
            
        #     # --- 2. Frequency Domain Features (Spectral) ---
        #     n = len(sig)
        #     # Correct spacing for frequency axis based on channel-specific fs
        #     freqs = np.fft.rfftfreq(n, d=1/fs)
            
        #     # Compute Power Spectral Density (PSD)
        #     # Multiplied by 2 for rfft (except DC/Nyquist) and divided by fs for density
        #     fft_res = np.abs(np.fft.rfft(sig))
        #     psd = (fft_res**2) / (n * fs)
            
        #     # Define band masks
        #     delta_mask = (freqs >= 0.5) & (freqs <= 4)
        #     theta_mask = (freqs > 4) & (freqs <= 8)
        #     alpha_mask = (freqs > 8) & (freqs <= 12)
            
        #     # Calculate power in bands using trapezoidal integration for physical accuracy
        #     delta_p = np.trapezoid(psd[delta_mask], freqs[delta_mask]) if np.any(delta_mask) else 0.0
        #     theta_p = np.trapezoid(psd[theta_mask], freqs[theta_mask]) if np.any(theta_mask) else 0.0
        #     alpha_p = np.trapezoid(psd[alpha_mask], freqs[alpha_mask]) if np.any(alpha_mask) else 0.0
            
        #     # Ratio biomarker: Delta/Theta (Indicator of cognitive slowing)
        #     dt_ratio = delta_p / theta_p if theta_p > 0 else 0.0

        #     final_features.extend([std_val, mav_val, energy_val, delta_p, theta_p, alpha_p, dt_ratio])

        if sig is not None and len(sig) > 1:
            # --- Time Domain Features (Very Fast) ---
            std_val = np.std(sig)
            mav_val = np.mean(np.abs(sig))
            
            # Zero Crossing Rate (Proxy for frequency/slowing)
            zcr = np.mean(np.diff(np.sign(sig)) != 0)
            
            # Root Mean Square
            rms = np.sqrt(np.mean(sig**2))
            
            # Signal Activity (Variance)
            activity = np.var(sig)
            
            # Mobility (Hjorth Parameter) - Proxy for mean frequency
            # sqrt(var(diff(sig)) / var(sig))
            diff_sig = np.diff(sig)
            mobility = np.sqrt(np.var(diff_sig) / activity) if activity > 0 else 0.0

            # Complexity (Hjorth Parameter) - Proxy for bandwidth
            diff2_sig = np.diff(diff_sig)
            var_d2 = np.var(diff2_sig)
            var_d1 = np.var(diff_sig)
            complexity = (np.sqrt(var_d2 / var_d1) / mobility) if (var_d1 > 0 and mobility > 0) else 0.0

            final_features.extend([std_val, mav_val, zcr, rms, activity, mobility, complexity])

        else:
            # Padding: 7 features per lead type
            final_features.extend([0.0] * 7)

    if 'processed_channels' in locals(): del processed_channels

    return np.array(final_features)

def extract_algorithmic_annotations_features(algo_data):
    """
    Extracts sleep architecture and event density features from CAISR outputs.
    Output vector length: 12
    """
    if not algo_data:
        return np.zeros(12)

    features = []

    # --- 1. Respiratory & Arousal Event Densities ---
    # Total duration in hours (assuming 1Hz for event traces)
    # If the signal exists, we calculate events per hour (Index)
    total_hours = len(algo_data.get('resp_caisr', [])) / 3600.0
    
    def count_discrete_events(key):
        if key not in algo_data or total_hours <= 0:
            return 0.0
        
        sig = algo_data[key].astype(float)
        # Create a binary mask: 1 if there is an event, 0 if not
        binary_sig = (sig > 0).astype(int)
        
        # Detect rising edges: 0 to 1 transition
        # diff will be 1 at the start of an event, -1 at the end
        diff = np.diff(binary_sig, prepend=0)
        num_events = np.count_nonzero(diff == 1)
        
        return num_events / total_hours
    
    ahi_auto = count_discrete_events('resp_caisr')      # Automated Apnea-Hypopnea Index
    arousal_auto = count_discrete_events('arousal_caisr') # Automated Arousal Index
    limb_auto = count_discrete_events('limb_caisr')    # Automated Limb Movement Index
    
    features.extend([ahi_auto, arousal_auto, limb_auto])

    # --- 2. Sleep Architecture (from stage_caisr) ---
    # Standard labels: 5=W, 4=R, 3=N1, 2=N2, 1=N3 (or similar mapping)
    stages = algo_data.get('stage_caisr', np.array([]))
    # Filter out invalid/background values (like the 9.0 in your sample)
    valid_stages = stages[stages < 9.0]
    
    if len(valid_stages) > 0:
        total_epochs = len(valid_stages)
        # Percentage of each stage
        w_pct = np.mean(valid_stages == 5)
        r_pct = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)
        
        # Sleep Efficiency: (N1+N2+N3+R) / Total
        efficiency = np.mean((valid_stages >= 1) & (valid_stages <= 4))
    else:
        w_pct = n1_pct = n2_pct = n3_pct = r_pct = efficiency = 0.0

    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency])

    # --- 3. Model Confidence / Uncertainty ---
    # Mean probability of Wake and REM (indicators of sleep stability)
    # We use the raw probability traces
    prob_w = np.mean(algo_data.get('caisr_prob_w', [0]))
    prob_n3 = np.mean(algo_data.get('caisr_prob_n3', [0]))
    prob_arous = np.mean(algo_data.get('caisr_prob_arous', [0]))
    
    # Standardize '9.0' or other filler values to 0
    clean_prob = lambda x: x if x < 1.0 else 0.0
    features.extend([clean_prob(prob_w), clean_prob(prob_n3), clean_prob(prob_arous)])

    return np.array(features)

def extract_human_annotations_features(human_data):
    """
    Extracts features from expert-scored human annotations.
    Output vector length: 12 (to match algorithmic feature length)
    """
    # If data is missing (common in hidden test sets), return a zero vector
    if not human_data or 'resp_expert' not in human_data:
        return np.zeros(12)

    features = []

    # --- 1. Human Event Indices (Events per Hour) ---
    # Total duration in hours based on 1Hz signal
    total_seconds = len(human_data.get('resp_expert', []))
    total_hours = total_seconds / 3600.0
    
    def count_discrete_events(key):
        if key not in human_data or total_hours <= 0:
            return 0.0
        sig = (human_data[key] > 0).astype(int)
        # Identify the start of each continuous event block
        diff = np.diff(sig, prepend=0)
        return np.count_nonzero(diff == 1) / total_hours

    ahi_human = count_discrete_events('resp_expert')      # Human AHI
    arousal_human = count_discrete_events('arousal_expert') # Human Arousal Index
    limb_human = count_discrete_events('limb_expert')       # Human PLMI
    
    features.extend([ahi_human, arousal_human, limb_human])

    # --- 2. Human Sleep Architecture ---
    # Standard labels: 0=W, 1=N1, 2=N2, 3=N3, 4=R, 5=Unknown/Movement
    stages = human_data.get('stage_expert', np.array([]))
    
    # Filter out label 5 (often used by experts for movement/unscored)
    valid_mask = (stages < 9.0)
    valid_stages = stages[valid_mask]
    
    if len(valid_stages) > 0:
        w_pct = np.mean(valid_stages == 5)
        r_pct = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)
        efficiency = np.mean(valid_stages > 0)
    else:
        w_pct = n1_pct = n2_pct = n3_pct = r_pct = efficiency = 0.0

    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency])

    # --- 3. Fragmentation & Stability (Replacing Probabilities) ---
    # These metrics quantify how "broken" the sleep is, which is a key marker.
    if len(valid_stages) > 1:
        # Number of stage transitions
        transitions = np.count_nonzero(np.diff(valid_stages)) / total_hours
        # Wake After Sleep Onset (WASO) proxy: non-zero stages followed by zero
        waso_minutes = (np.count_nonzero(valid_stages == 0) * 30) / 60.0
        # REM Latency (epochs until first REM)
        rem_indices = np.where(valid_stages == 4)[0]
        rem_latency = rem_indices[0] if len(rem_indices) > 0 else 0.0
    else:
        transitions = waso_minutes = rem_latency = 0.0

    features.extend([transitions, waso_minutes, rem_latency])

    return np.array(features)


# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
