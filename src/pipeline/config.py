import os

from src.ecg_processing import ECG_FEATURE_LENGTH
from src.eeg_processing import EEG_FEATURE_LENGTH
from src.resp_processing import RESP_FEATURE_LENGTH


SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')
FEATURE_CACHE_FOLDER_NAME = '.feature_cache'
INCLUDE_SUBJECT_FEATURE_MEAN = False
SEGMENT_AGGREGATION_NAMES = (
    ('Max', 'Min', 'Mean', 'Median', 'Std')
    if INCLUDE_SUBJECT_FEATURE_MEAN
    else ('Max', 'Min', 'Median', 'Std')
)
FEATURE_CORRELATION_THRESHOLD = 0.8
MAX_TRAIN_WORKERS = max(1, min(4, os.cpu_count() or 1))
USE_SITE_GROUPED_CV = False
OPTIMIZE_HYPERPARAMETER_SEARCH = False
RANDOM_CV_N_SPLITS = 5
CV_RANDOM_STATE = 42
CV_SEARCH_ITERATIONS = 20
DEFAULT_CV_HYPERPARAMETERS = {
    'max_depth': 3,
    'min_child_weight': 3,
    'subsample': 0.9,
    'colsample_bytree': 0.6,
    'reg_lambda': 0.5,
    'reg_alpha': 0.1,
}
SEGMENT_DURATION_SECONDS = 5 * 60
SEGMENT_STRIDE_SECONDS = 15 * 60
TOTAL_PHYSIOLOGICAL_FEATURE_LENGTH = (
    RESP_FEATURE_LENGTH * len(SEGMENT_AGGREGATION_NAMES)
    + EEG_FEATURE_LENGTH * len(SEGMENT_AGGREGATION_NAMES)
    + ECG_FEATURE_LENGTH * len(SEGMENT_AGGREGATION_NAMES)
)
