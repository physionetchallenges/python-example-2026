#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.
# Check the example code to see how to use these functions in your code.

import edfio
import numpy as np
import os
import pandas as pd
import scipy as sp
import sklearn
import sys

from typing import Dict, List, Tuple, Any, Union, Tuple, Optional
from collections import defaultdict

DEMOGRAPHICS_FILE = 'training_set/demographics.csv'
PHYSIOLOGICAL_DATA_SUBFOLDER = 'physiological_data'
ALGORITHMIC_ANNOTATIONS_SUBFOLDER = 'algorithmic_annotations'
HUMAN_ANNOTATIONS_SUBFOLDER = 'human_annotations'

# --- CSV Header Mapping Configuration ---
# Centralized dictionary to manage all CSV column names for the HSP project
HEADERS = {
    # --- Demographics / Metadata Input Columns ---
    'site_id':               'SiteID',
    'patient_id':            'BDSPPatientID',
    'creation_time':         'CreationTime',
    'bids_folder':           'BidsFolder',
    'session_id':            'SessionID',
    'age':                   'Age',
    'sex':                   'Sex',
    'race':                  'Race',
    'ethnicity':             'Ethnicity',
    'bmi':                   'BMI',
    'time_to_event':         'Time_to_Event',
    'label':                 'Cognitive_Impairment',
    'last_visit_date':       'Last_Known_Visit_Date',
    'time_to_last_visit':    'Time_to_Last_Visit',

    # --- Model Prediction Output Columns ---
    'prediction_binary':     'Cognitive_Impairment',
    'prediction_prob':       'Cognitive_Impairment_Probability'
}

### Challenge data I/O functions
def load_rename_rules(csv_path: str) -> Dict[str, List[str]]:
    """
    Loads channel aliases from a CSV file and prepares the renaming rules.
    The CSV should have a 'Channel_Names' column containing string representations 
    of tuples or lists of aliases (e.g., "('C3-M2', 'C3-A2')").

    Returns:
        Dict[str, List[str]]: {Standardized Name (first alias): All Aliases}
    """
    rename_rules = {}
    try:
        channel_table = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Channel table file not found at {csv_path}")
        return rename_rules
    
    if 'Channel_Names' not in channel_table.columns:
        print("Error: CSV file must contain a 'Channel_Names' column.")
        return rename_rules

    for _, row in channel_table.iterrows():
        alias_str_raw = row['Channel_Names']
        
        if pd.isna(alias_str_raw):
            continue

        try:
            # Parse the string representation into a list or tuple
            alias_list = [a.strip().replace("'", "").replace('"', "") 
                for a in str(alias_str_raw).split(';')]
            
            # Remove empty strings
            alias_list = [a for a in alias_list if a]
            
            if alias_list:
                # Use the first alias as the standardized name
                key = alias_list[0].lower() 
                # Store all aliases in lower case for matching
                rename_rules[key] = [str(a) for a in alias_list]

        except (ValueError, SyntaxError, TypeError) as e:
            # print(f"Skipping row due to parsing error: {e} in raw string: {alias_str_raw}")
            continue
            
    return rename_rules


# --- helper function for cleaning channel names ---
def _get_cleaned_name(channel_name: str) -> str:
    """
    Standardizes channel names by converting to lower case and removing common suffixes/separators.
    """
    # convert to lower case
    cleaned = channel_name.lower()
    
    # remove common suffixes
    cleaned = cleaned.replace('_pds', '').replace('_eg', '')
    
    # remove common separators
    cleaned = cleaned.replace(':', '-')
    
    # remove leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned

# --- Main mapping function ---

def map_valid_channels_rename_only(columns_original: List[str], rename_rules: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Finds the first match for each standard name from the rename_rules in the 
    list of original channel names.

    Args:
        columns_original: The raw list of channel labels found in the file.
        rename_rules: Dictionary mapping {Standard Name: [Alias1, Alias2, ...]}

    Returns:
        Dict[str, str]: {Standard Name: Matched Original Channel Name}
    """
    channel_map = {}
    
    # Step 1: Create a map from cleaned column names to original names
    # {cleaned_name: original_name}
    cleaned_to_original_map = {_get_cleaned_name(col): col for col in columns_original}
    
    # Step 2: Iterate through standard names and aliases to find a match
    for std_name, aliases in rename_rules.items():
        for alias in aliases:
            # Clean the alias for comparison
            alias_cleaned = _get_cleaned_name(alias)
            
            if alias_cleaned in cleaned_to_original_map:
                # Found a match! Get the original, uncleaned name
                orig_col = cleaned_to_original_map[alias_cleaned]
                
                # Use the standard name as the key, and the original name as the value
                channel_map[std_name] = orig_col 
                
                # Stop searching for this standard name once the best match is found
                break  
    
    # Note: If needed, you might implement more complex mirror map logic here,
    # but based on your original simplified code, we skip it for now.

    return channel_map

# --- Main standardization function ---

def standardize_channel_names_rename_only(
    columns_original: List[str], 
    rename_rules: Dict[str, List[str]]
) -> Tuple[Dict[str, str], List[str]]:
    """ 
    Standardizes channel names based on rules, identifies duplicates to drop, 
    and handles pulse/pr renaming.

    Args:
        columns_original: The raw list of channel labels (strings) from the file.
        rename_rules: Dictionary mapping {Standard Name: [Alias1, Alias2, ...]}

    Returns:
        Tuple[Dict[str, str], List[str]]: 
            - rename_map: {Original Raw Name: New Standard Name}
            - cols_to_drop: List of original raw names to be dropped.
    """
    
    # Step 1: Find the desired standard name for each matching raw channel
    # Output: {Standard Name: Matched Original Raw Name}
    channel_map = map_valid_channels_rename_only(columns_original, rename_rules)
    
    # Step 2: Reverse map (Raw Name -> Standard Name) for the final rename operation
    # Output: {Original Raw Name: Standard Name}
    rename_map = {orig_raw: std_name for std_name, orig_raw in channel_map.items()}

    # Step 3: Detect and collect duplicate aliases to drop
    cols_to_drop = []

    # Map cleaned names to their original names for quick lookup
    cleaned_to_original_map = {_get_cleaned_name(col): col for col in columns_original}
    
    for std_name, matched_raw in channel_map.items():
        # Get all cleaned aliases corresponding to this standard name
        aliases_cleaned = {_get_cleaned_name(a) for a in rename_rules.get(std_name, [])}

        # Check all existing columns in the file
        for raw_col in columns_original:
            cleaned_col = _get_cleaned_name(raw_col)
            
            # If a column's cleaned name is one of the standard name's aliases
            if cleaned_col in aliases_cleaned:
                
                # AND this column is NOT the one chosen to be kept
                # (We keep 'matched_raw' and drop others that map to the same 'std_name')
                if raw_col != matched_raw:
                    cols_to_drop.append(raw_col)
    
    # Remove duplicates from the drop list
    cols_to_drop = sorted(set(cols_to_drop))

    # Step 4: Handle pulse/pr → hr rename (if not already handled by rename_rules)
    pulse_map = {"pulse": "hr", "pr": "hr"}
    
    # Iterate through the original channels to find 'pulse' or 'pr'
    for orig_ch_raw in columns_original:
        orig_ch_cleaned = _get_cleaned_name(orig_ch_raw)
        
        for orig_ch_alias, new_ch_standard in pulse_map.items():
            if orig_ch_cleaned == orig_ch_alias:
                # Check if this raw channel has NOT been mapped to a standard name yet (to avoid overwriting EEG channel names)
                if orig_ch_raw not in rename_map:
                    # Check if 'hr' is already a standard name in the map values (to avoid duplicate 'hr' output)
                    if new_ch_standard not in rename_map.values():
                        # Add this rename directly to the final map
                        rename_map[orig_ch_raw] = new_ch_standard
    
    return rename_map, cols_to_drop


def derive_bipolar_signal(
    ch_a_signal: np.ndarray, 
    ref_signal: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], 
) -> Optional[np.ndarray]:
    """
    Derives a new bipolar EEG channel by subtracting a reference signal 
    from a primary signal (A - Reference).

    Args:
        ch_a_signal: The primary signal (e.g., C4). Must be in physical units.
        ref_signal: The reference signal(s). Can be a single Series (M1) 
                    or a tuple of two Series (M1, M2) for average referencing.
        scaling_factor: Factor applied to the reference. Use 0.5 for average 
                        mastoid reference (A - 0.5 * (B + C)).

    Returns:
        np.ndarray: The derived bipolar signal, or None if input formats are invalid.
    """
    # Make sure inputs are numPy arrays
    try:
        if isinstance(ref_signal, tuple):
            # Average Reference: A - (B + C) / 2
            sig_b, sig_c = ref_signal
            return ch_a_signal - 0.5 * (sig_b + sig_c)
        else:
            # Simple Bipolar: A - B
            return ch_a_signal - ref_signal
    except Exception as e:
        print(f"Bipolar derivation error: {e}")
        return None

def load_edf_to_nparrays(edf_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Loads an EDF file and returns a dictionary of NumPy arrays.
    Each key is a channel label, and the value is the physical signal (float64).
    and their sampling frequencies (float64).

    Args:
        edf_path: Path to the EDF file.

    Returns:
        Tuple[Dict, Dict]: 
            - channel_dict: {channel_label: physical_signal_array}
            - fs_dict: {channel_label: sampling_frequency}
    """
    try:
        # Lazy loading is set to False to read data into memory immediately
        edf = edfio.read_edf(edf_path, lazy_load_data=False)
        signals = edf.signals
    except Exception as e:
        print(f"Error loading EDF file: {e}")
        return {}, {}

    channel_dict = {}
    fs_dict = {}

    for sig in signals:
        # Extract channel label
        label = sig.label.lower().strip()
        
        # Extract Sampling Frequency (fs)
        fs_dict[label] = float(sig.sampling_frequency)

        # Extract physical signal data as float64 NumPy array
        channel_dict[label] = sig.data

    return channel_dict, fs_dict

# Extracts a list of unique patient identifiers (BIDS folder names) from the metadata CSV
def find_patients(patient_data_file):
    """
    Returns a list of dictionaries, each containing the identifiers 
    needed to locate specific physiological files.
    """
    df = pd.read_csv(patient_data_file)
    # Get the unique combinations of patient, site, and session
    cols = [HEADERS['bids_folder'], HEADERS['site_id'], HEADERS['session_id']]
    patient_info = df[cols].drop_duplicates()
    
    return patient_info.to_dict('records')

# Loads the raw physiological signal data
def load_signal_data(edf_path):
    """
    Wraps the efficient signal loading function to convert EDF files into NumPy arrays.
    """
    return load_edf_to_nparrays(edf_path)

# Find the records in a folder and its subfolders.
# Redo this to load them from the patient file.
def find_records(folder, file_extension='.edf'):
    records = set()
    for root, directories, files in os.walk(folder):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == file_extension:
                record = os.path.relpath(os.path.join(root, file), folder)[:-len(file_extension)]
                records.add(record)
    records = sorted(records)
    return records

# Save updated demographics with predictions
def update_demographics_table(input_file, output_folder, results_dict):
    """
    input_file: original demographics.csv file path
    output_folder: Output folder to save updated demographics.csv
    results_dict: {record_id: (binary_label, probability)}
    """
    # Define column names
    id_bin = HEADERS['prediction_binary']
    id_prob = HEADERS['prediction_prob']
    id_folder = HEADERS['bids_folder']

    # Load the original demographics file
    df = pd.read_csv(input_file)
    
    # Create new columns if they do not exist
    if id_bin not in df.columns: df[id_bin] = None
    if id_prob not in df.columns: df[id_prob] = float('nan')
    
    # Fill in the predictions
    for record_id, (label, prob) in results_dict.items():
        # Match by 'BidsFolder'
        mask = df[id_folder] == record_id
        df.loc[mask, id_bin] = label
        df.loc[mask, id_prob] = prob
    
    # Save to file
    output_file = os.path.join(output_folder, os.path.basename(input_file))
    df.to_csv(output_file, index=False)
    return output_file

# ### Demographic Loading Functions load from demographic CSV Data

# Helper functions to load specific demographic fields from a data dictionary.
def get_header(key):
    return HEADERS.get(key, key)

def load_demographics(metadata_file, patient_id, session_id):
    """
    Loads all demographic data for a patient at once to minimize I/O.
    Returns a dictionary of raw values for the patient_id (BidsFolder).
    """
    df = pd.read_csv(metadata_file)
    # Ensure patient_id matches the 'BidsFolder' column
    mask = (df[HEADERS['bids_folder']] == patient_id) & (df[HEADERS['session_id']] == session_id)
    patient_data = df.loc[mask]
    
    if not patient_data.empty:
        return patient_data.iloc[0].to_dict()
    return {}

def load_site_id(data):
    """Extracts the site identifier from the data dictionary."""
    return data.get(HEADERS['site_id'], 'Unknown')

def load_session(data):
    """Extracts the session identifier from the data dictionary."""
    return data.get(HEADERS['session_id'], 'Unknown')

def load_age(data):
    """Extracts age and converts it to a float."""
    age_val = data.get(HEADERS['age'])
    try:
        return float(age_val) if age_val is not None else 0.0
    except (ValueError, TypeError):
        return 0.0

def load_sex(data):
    """Extracts and standardizes the sex label."""
    sex = str(data.get(HEADERS['sex'], '')).lower()
    if sex.startswith('f'): return 'Female'
    if sex.startswith('m'): return 'Male'
    return 'Unknown'

def load_bmi(data):
    """Extracts BMI and handles invalid or missing values."""
    bmi_val = data.get(HEADERS['bmi'])
    try:
        bmi_float = float(bmi_val)
        return bmi_float if not np.isnan(bmi_float) else 0.0
    except (ValueError, TypeError):
        return 0.0

def load_label(data):
    """Extracts the cognitive impairment label from the dictionary (for training)."""
    val = data.get(HEADERS['label'])
    if isinstance(val, str):
        return 1 if val.upper() == 'TRUE' else 0
    return 1 if val is True else 0

def load_race(data):
    """Extracts the race label."""
    return data.get(HEADERS['race'], 'Unavailable')

def load_ethnicity(data):
    """Extracts the ethnicity label."""
    return data.get(HEADERS['ethnicity'], 'Unavailable')

def get_standardized_race(data):
    """
    Standardizes raw race and ethnicity strings into five canonical categories:
    Asian, Black, Others, Unavailable, White
    """
    # Convert raw inputs to lowercase for case-insensitive matching
    race_raw = str(data.get(HEADERS['race'], '')).lower()
    
    # Check for keywords
    if any(word in race_raw for word in ['white', 'caucasian']):
        return 'White'
    if any(word in race_raw for word in ['black', 'african american']):
        return 'Black'
    if 'asian' in race_raw:
        return 'Asian'
    
    # Handle Unavailable/Unknown
    unavailable_keywords = ['unknown', 'unavailable', 'declined', 'unreported', 'nan',
                            'none', 'not specified', 'prefer not to say']
    if any(word == race_raw or word in race_raw for word in unavailable_keywords):
        return 'Unavailable'
    elif race_raw.strip() == '':
        return 'Unavailable'
    
    # Fallback to Others
    return 'Others'

def get_standardized_ethnicity(data):
    """
    Standardizes ethnicity into three categories: Hispanic, Not Hispanic, Unavailable
    """
    ethnic_raw = str(data.get(HEADERS['ethnicity'], '')).lower().strip()

    # Prioritize Not Hispanic detection
    not_hispanic_keywords = [
        'not hispanic', 'non-hispanic', 'non hispanic', 'not latino', 'non-latino'
    ]
    if any(word in ethnic_raw for word in not_hispanic_keywords):
        return 'Not Hispanic'
    
    # Check for Hispanic/Latino keywords
    if 'hispanic' in ethnic_raw or 'latino' in ethnic_raw:
        return 'Hispanic'
    
    # Handle Unavailable
    unavailable_keywords = ['unknown', 'unavailable', 'declined', 'unreported', 'nan', 
                            'none', 'not specified', 'prefer not to say']
    if any(word == ethnic_raw or word in ethnic_raw for word in unavailable_keywords):
        return 'Unavailable'
    elif ethnic_raw.strip() == '':
        return 'Unavailable'
    
    # Default if no clear indication
    return 'Unavailable'

# Retrieves the cognitive status/diagnosis label
def load_diagnoses(metadata_file, patient_id):
    """
    Fetches the raw 'Cognitive_Impairment' status for a specific patient from the metadata.
    """
    df = pd.read_csv(metadata_file)
    mask = df[HEADERS['bids_folder']] == patient_id
    val = df.loc[mask, HEADERS['label']].values[0]
    return 1 if val else 0

def load_Time_to_Event(data):
    """Extracts Time_to_Event and converts it to a float."""
    tte_val = data.get(HEADERS['time_to_event'])
    try:
        return float(tte_val) if tte_val is not None else -1.0
    except (ValueError, TypeError):
        return -1.0
    
def load_Last_Known_Visit_Date(data):
    """Extracts Last_Known_Visit_Date."""
    return data.get(HEADERS['last_known_visit_date'], 'Unknown')

def load_Time_to_Last_Visit(data):
    """Extracts Time_to_Last_Visit and converts it to a float."""
    ttlv_val = data.get(HEADERS['time_to_last_visit'])
    try:
        return float(ttlv_val) if ttlv_val is not None else -1.0
    except (ValueError, TypeError):
        return -1.0

# ### EDF Handling Functions
def load_edf(record: str):
    """
    Loads a complete EDF file with error handling.
    Adds the .edf extension if it is missing from the record path.
    """
    if not record.endswith('.edf'):
        record += '.edf'
        
    try:
        # Using lazy_load_data=False to fully load signals into memory.
        # Use lazy_load_data=True if you only need the header/metadata.
        return edfio.read_edf(record, lazy_load_data=False)
    except Exception as e:
        print(f"Error: Could not read EDF file {record}: {e}")
        return None

def get_sampling_frequency(signal: edfio.EdfSignal) -> Optional[float]:
    """
    Returns the sampling frequency for a specific EdfSignal object.
    Since EDF files can have mixed sampling rates, call this per channel.
    """
    try:
        return signal.sampling_frequency
    except AttributeError:
        return None

def get_num_samples(signal: edfio.EdfSignal) -> Optional[int]:
    """
    Returns the total number of samples in a specific EdfSignal object.
    """
    try:
        return len(signal.data)
    except (AttributeError, TypeError):
        return None

def get_signal_name(signal: edfio.EdfSignal) -> str:
    """
    Returns the label (channel name) of a specific EdfSignal object.
    """
    try:
        return signal.label.strip()
    except AttributeError:
        return ""

def get_signal_data(signal: edfio.EdfSignal) -> np.ndarray:
    """
    Returns the data array of a specific EdfSignal object.
    """
    try:
        return signal.data
    except AttributeError:
        return np.array([])

def load_signals_as_array(edf_object: edfio.Edf) -> Optional[np.ndarray]:
    """
    Extracts signals into a list of NumPy arrays.
    Each array in the list may have a different length depending on its sampling frequency.
    """
    try:
        # Returns a list of 1D arrays: [array_ch1, array_ch2, ...]
        return [sig.data for sig in edf_object.signals]
    except Exception as e:
        print(f"Error converting signals to array: {e}")
        return None
    
### Evaluation functions

# Compute the Challenge score.
def compute_challenge_score(labels, outputs, fraction_capacity = 0.05, num_permutations = 10**4, seed=12345):
    # Check the data.
    assert len(labels) == len(outputs)
    num_instances = len(labels)
    capacity = int(fraction_capacity * num_instances)

    # Convert the data to NumPy arrays, as needed, for easier indexing.
    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)

    # Permute the labels and outputs so that we can approximate the expected confusion matrix for "tied" probabilities.
    tp = np.zeros(num_permutations)
    fp = np.zeros(num_permutations)
    fn = np.zeros(num_permutations)
    tn = np.zeros(num_permutations)

    if seed is not None:
        np.random.seed(seed)

    for i in range(num_permutations):
        permuted_idx = np.random.permutation(np.arange(num_instances))
        permuted_labels = labels[permuted_idx]
        permuted_outputs = outputs[permuted_idx]

        ordered_idx = np.argsort(permuted_outputs, stable=True)[::-1]
        ordered_labels = permuted_labels[ordered_idx]

        tp[i] = np.sum(ordered_labels[:capacity] == 1)
        fp[i] = np.sum(ordered_labels[:capacity] == 0)
        fn[i] = np.sum(ordered_labels[capacity:] == 1)
        tn[i] = np.sum(ordered_labels[capacity:] == 0)

    tp = np.mean(tp)
    fp = np.mean(fp)
    fn = np.mean(fn)
    tn = np.mean(tn)

    # Compute the true positive rate.
    if tp + fn > 0:
        tpr = tp / (tp + fn)
    else:
        tpr = float('nan')

    return tpr

def compute_auc(labels, outputs):
    import sklearn
    import sklearn.metrics

    auroc = sklearn.metrics.roc_auc_score(labels, outputs, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)
    auprc = sklearn.metrics.average_precision_score(labels, outputs, average='macro', pos_label=1, sample_weight=None)

    return auroc, auprc

# Compute accuracy.
def compute_accuracy(labels, outputs):
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(labels, outputs, normalize=True, sample_weight=None)

    return accuracy

# Compute F-measure.
def compute_f_measure(labels, outputs):
    from sklearn.metrics import f1_score

    f_measure = f1_score(labels, outputs, pos_label=1, average='binary')

    return f_measure

### Other helper functions

# Remove any single or double quotes; parentheses, braces, and brackets (for singleton arrays); and spaces and tabs from a string.
def remove_extra_characters(x):
    x = str(x)
    x = x.replace('"', '').replace("'", "")
    x = x.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '')
    x = x.replace(' ', '').replace('\t', '')
    x = x.strip()
    return x

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Check if a variable is a NaN, i.e., not a number, or represents a NaN.
def is_nan(x):
    if is_number(x):
        return np.isnan(float(x))
    else:
        return False

# Check if a variable is a boolean or represents a boolean.
def is_boolean(x):
    if (is_number(x) and float(x)==0) or (remove_extra_characters(x).casefold() in ('false', 'f', 'no', 'n')):
        return True
    elif (is_number(x) and float(x)==1) or (remove_extra_characters(x).casefold() in ('true', 't', 'yes', 'y')):
        return True
    else:
        return False

# Sanitize integer values.
def sanitize_integer_value(x):
    x = remove_extra_characters(x)
    if is_integer(x):
        return int(float(x))
    else:
        return float('nan')

# Sanitize scalar values.
def sanitize_scalar_value(x):
    x = remove_extra_characters(x)
    if is_number(x):
        return float(x)
    else:
        return float('nan')

# Sanitize boolean values.
def sanitize_boolean_value(x):
    x = remove_extra_characters(x)
    if (is_number(x) and float(x)==0) or (remove_extra_characters(x).casefold() in ('false', 'f', 'no', 'n')):
        return 0
    elif (is_number(x) and float(x)==1) or (remove_extra_characters(x).casefold() in ('true', 't', 'yes', 'y')):
        return 1
    else:
        return float('nan')
