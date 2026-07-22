# features/human.py
# Human annotation feature extraction — not used in the model
#
# Human annotations are ONLY available in the training set.
# They are intentionally excluded from the model feature vector —
# run_model() must work without them on the hidden validation/test sets.
# This module is retained for research and EDA purposes only.
#
# Iteration history:
#   v1 (2026-06-30): Initial — AHI, arousal, limb, stage%, transitions,
#                    WASO, REM latency  [12 features — not in model]

import numpy as np


def extract_human_annotations_features(human_data):
    """
    Extracts features from expert-scored human annotations.
    NOT INCLUDED in the model feature vector — for research/EDA only.
    Returns np.ndarray of length 12.
    """
    if not human_data or 'resp_expert' not in human_data:
        return np.full(12, float('nan'))

    features      = []
    total_seconds = len(human_data.get('resp_expert', []))
    total_hours   = total_seconds / 3600.0

    def count_events(key):
        if key not in human_data or total_hours <= 0:
            return float('nan')
        sig   = (human_data[key] > 0).astype(int)
        edges = np.diff(sig, prepend=0)
        return np.count_nonzero(edges == 1) / total_hours

    features.extend([
        count_events('resp_expert'),
        count_events('arousal_expert'),
        count_events('limb_expert'),
    ])

    stages = human_data.get('stage_expert', np.array([]))
    valid  = stages[stages < 9.0] if len(stages) > 0 else np.array([])
    if len(valid) > 0:
        features.extend([
            float(np.mean(valid == 5)),
            float(np.mean(valid == 4)),
            float(np.mean(valid == 3)),
            float(np.mean(valid == 2)),
            float(np.mean(valid == 1)),
            float(np.mean(valid > 0)),
        ])
    else:
        features.extend([float('nan')] * 6)

    if len(valid) > 1:
        features.extend([
            float(np.count_nonzero(np.diff(valid)) / total_hours),
            float(np.count_nonzero(valid == 0) * 30 / 60.0),
            float(np.where(valid == 4)[0][0]) if np.any(valid == 4) else float('nan'),
        ])
    else:
        features.extend([float('nan')] * 3)

    return np.array(features)