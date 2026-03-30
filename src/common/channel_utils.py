import os

import pandas as pd


CHANNEL_TABLE_CACHE = {}


def normalize_channel_label(text):
    normalized = ''.join(ch if ch.isalnum() else ' ' for ch in str(text).lower())
    return ' '.join(normalized.split())


def split_channel_aliases(raw_aliases):
    return {normalize_channel_label(alias) for alias in str(raw_aliases).split(';') if alias}


def get_cached_channel_table(csv_path):
    normalized_csv_path = os.path.abspath(csv_path)
    channels = CHANNEL_TABLE_CACHE.get(normalized_csv_path)
    if channels is None:
        channels = pd.read_csv(normalized_csv_path)
        CHANNEL_TABLE_CACHE[normalized_csv_path] = channels
    return channels, normalized_csv_path


def find_matching_label(signal_dict, aliases):
    for label in signal_dict.keys():
        if normalize_channel_label(label) in aliases:
            return label
    return None