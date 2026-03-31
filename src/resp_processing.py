from .lib import resp_features
import numpy as np
from src.common.channel_utils import get_cached_channel_table, normalize_channel_label, split_channel_aliases
from src.common.signal_utils import resample_signal

RESP_CHANNEL_GROUPS = ("Abdomen", "Chest", "Nasal", "Flow")
RESP_FEATURE_NAMES = [
    f"{group}_Peakedness_{metric}"
    for group in RESP_CHANNEL_GROUPS
    for metric in ("Max", "Min", "Mean", "Median", "Std")
] + [
    "SpO2_Max",
    "SpO2_Min",
    "SpO2_Mean",
    "SpO2_Std",
    "CET90",
    "ODI_Mean",
    "ODI_deepness",
]
RESP_FEATURE_LENGTH = len(RESP_FEATURE_NAMES)
RESP_ALIAS_GROUPS_CACHE = {}


def _build_resp_alias_groups(channels):
    resp_rows = channels[channels['Category'].eq('resp')].reset_index(drop=True)
    if len(resp_rows) < 7:
        return {}
    return {
        'Abdomen': split_channel_aliases(resp_rows.iloc[0]['Channel_Names']),
        'Chest': split_channel_aliases(resp_rows.iloc[1]['Channel_Names']),
        'Nasal': split_channel_aliases(resp_rows.iloc[2]['Channel_Names']),
        'Flow': split_channel_aliases(resp_rows.iloc[3]['Channel_Names']),
        'SpO2': split_channel_aliases(resp_rows.iloc[6]['Channel_Names']),
    }


def _get_resp_alias_groups(csv_path):
    channels, normalized_csv_path = get_cached_channel_table(csv_path)
    alias_groups = RESP_ALIAS_GROUPS_CACHE.get(normalized_csv_path)
    if alias_groups is None:
        alias_groups = _build_resp_alias_groups(channels)
        RESP_ALIAS_GROUPS_CACHE[normalized_csv_path] = alias_groups
    return alias_groups


def _find_resp_group(label, alias_groups):
    normalized = normalize_channel_label(label)
    for group_name, aliases in alias_groups.items():
        if normalized in aliases:
            return group_name
    return None


def _compute_resp_quality(used, hat_br):
    used_array = np.asarray(used, dtype=float)
    if used_array.size:
        quality = float(np.nanmean(used_array))
        if np.isfinite(quality):
            return quality
    hat_br = np.asarray(hat_br, dtype=float)
    if hat_br.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(hat_br)))


def _summarize_peakedness(hat_br):
    finite_values = np.asarray(hat_br, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return None
    return {
        'Max': float(np.max(finite_values)),
        'Min': float(np.min(finite_values)),
        'Mean': float(np.mean(finite_values)),
        'Median': float(np.median(finite_values)),
        'Std': float(np.std(finite_values)),
    }


def _summarize_spo2(data, fs):
    if data.size == 0:
        return {}
    working = np.asarray(data, dtype=float).copy()
    if np.nanmax(working) < 2:
        working = np.round((working / 1.055) * 100)

    desaturation_mask = working.copy()
    threshold = 0.7
    for index, value in enumerate(working):
        if value < threshold:
            start = int(max(0, index - fs * 2))
            end = int(min(working.size, index + fs * 2))
            desaturation_mask[start:end] = np.nan

    cet90 = float(np.count_nonzero(desaturation_mask < 90) / max(working.size, 1))
    valid = desaturation_mask[np.isfinite(desaturation_mask)]
    if valid.size == 0:
        return {'CET90': cet90}

    odi_mean, odi_deepness = resp_features.odi_application(desaturation_mask, fs)
    return {
        'SpO2_Max': float(np.max(valid)),
        'SpO2_Min': float(np.min(valid)),
        'SpO2_Mean': float(np.mean(valid)),
        'SpO2_Std': float(np.std(valid)),
        'CET90': cet90,
        'ODI_Mean': float(odi_mean),
        'ODI_deepness': float(odi_deepness),
    }


def processResp(physiological_data, physiological_fs, csv_path):
    alias_groups = _get_resp_alias_groups(csv_path)
    results = {feature_name: np.nan for feature_name in RESP_FEATURE_NAMES}
    best_quality = {group_name: -np.inf for group_name in RESP_CHANNEL_GROUPS}

    for label, signal in physiological_data.items():
        if label not in physiological_fs:
            continue

        group_name = _find_resp_group(label, alias_groups)
        if group_name is None:
            continue

        resampled, fs = resample_signal(signal, physiological_fs[label], 25)
        resampled = np.nan_to_num(resampled, nan=0.0, posinf=0.0, neginf=0.0)

        if group_name == 'SpO2':
            results.update(_summarize_spo2(resampled, fs))
            continue

        try:
            hat_br, _, _, used = resp_features.peakedness_application(
                resampled,
                stage=label,
                subject_id=label,
            )
        except Exception:
            continue

        summary = _summarize_peakedness(hat_br)
        if summary is None:
            continue

        quality = _compute_resp_quality(used, hat_br)
        if quality <= best_quality[group_name]:
            continue

        best_quality[group_name] = quality
        for metric_name, metric_value in summary.items():
            results[f'{group_name}_Peakedness_{metric_name}'] = metric_value

    return np.array([results[name] for name in RESP_FEATURE_NAMES], dtype=np.float32)
