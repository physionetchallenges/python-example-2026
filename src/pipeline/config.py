import os

from src.ecg_processing import ECG_FEATURE_LENGTH
from src.eeg_processing import EEG_FEATURE_LENGTH
from src.resp_processing import RESP_FEATURE_LENGTH


SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')
FEATURE_CACHE_FOLDER_NAME = '.feature_cache'
MAX_TRAIN_WORKERS = max(1, min(4, os.cpu_count() or 1))
SEGMENT_DURATION_SECONDS = 5 * 60
SEGMENT_STRIDE_SECONDS = 15 * 60
TOTAL_PHYSIOLOGICAL_FEATURE_LENGTH = (
    RESP_FEATURE_LENGTH
    + EEG_FEATURE_LENGTH
    + ECG_FEATURE_LENGTH
)
