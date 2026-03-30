#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import os
import atexit
import builtins
import re
import sys
from tqdm import tqdm

from helper_code import *
from src.pipeline.config import DEFAULT_CSV_PATH
from src.pipeline.features import _get_feature_cache_file, get_or_create_record_feature_vector
from src.pipeline.training import train_xgb_model
################################################################################
# Path & Constant Configuration (Added for Robustness)
################################################################################

# Progress bar state for run_model (initialized lazily)
RUN_MODEL_PBAR = None
RUN_MODEL_PBAR_TOTAL = None
ORIGINAL_PRINT = builtins.print
PRINT_FILTER_ACTIVE = False
RUN_PROGRESS_LINE_RE = re.compile(r'^-\s+\d+/\d+:\s')
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

    if verbose:
        print('Training the model on the data...')

    model = train_xgb_model(data_folder, verbose, csv_path)

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

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
