# features/caisr_base.py
# Baseline CAISR annotation features — features 59-70
#
# Iteration history:
#   v1 (2026-06-30): Initial — AHI, arousal idx, limb idx,
#                    stage percentages (W/N1/N2/N3/REM),
#                    sleep efficiency, mean P(Wake), mean P(N3),
#                    mean P(arousal)  [12 features]

import numpy as np


def extract_caisr_base_features(algo_data):
    """
    Extracts baseline CAISR features from annotation channels.

    Returns np.ndarray of length 12:
        [0]  AHI total (events/hr)
        [1]  Arousal index (events/hr)
        [2]  Limb movement index (events/hr)
        [3]  Wake %
        [4]  N1 %
        [5]  N2 %
        [6]  N3 %
        [7]  REM %
        [8]  Sleep efficiency
        [9]  Mean CAISR P(Wake)
        [10] Mean CAISR P(N3)
        [11] Mean CAISR P(arousal)
    """
    if not algo_data:
        return np.full(12, float('nan'))

    resp    = algo_data.get('resp_caisr',    np.array([]))
    arousal = algo_data.get('arousal_caisr', np.array([]))
    limb    = algo_data.get('limb_caisr',    np.array([]))
    stages_raw = algo_data.get('stage_caisr', np.array([]))

    total_hours_resp    = len(resp)    / 3600.0 if len(resp)    > 0 else 0.0
    total_hours_arousal = len(arousal) / 7200.0 if len(arousal) > 0 else 0.0
    total_hours_limb    = len(limb)    / 3600.0 if len(limb)    > 0 else 0.0

    def count_events(sig, t_hours):
        if len(sig) == 0 or t_hours <= 0:
            return float('nan')
        binary = (np.asarray(sig) > 0).astype(int)
        edges  = np.diff(binary, prepend=0)
        return np.count_nonzero(edges == 1) / t_hours

    ahi_auto    = count_events(resp,    total_hours_resp)
    arousal_idx = count_events(arousal, total_hours_arousal)
    limb_idx    = count_events(limb,    total_hours_limb)

    valid_stages = stages_raw[stages_raw < 9.0] if len(stages_raw) > 0 else np.array([])

    if len(valid_stages) > 0:
        w_pct      = float(np.mean(valid_stages == 5))
        n1_pct     = float(np.mean(valid_stages == 3))
        n2_pct     = float(np.mean(valid_stages == 2))
        n3_pct     = float(np.mean(valid_stages == 1))
        rem_pct    = float(np.mean(valid_stages == 4))
        efficiency = float(np.mean((valid_stages >= 1) & (valid_stages <= 4)))
    else:
        w_pct = n1_pct = n2_pct = n3_pct = rem_pct = efficiency = float('nan')

    prob_w    = float(np.mean(algo_data.get('caisr_prob_w',     [float('nan')])))
    prob_n3   = float(np.mean(algo_data.get('caisr_prob_n3',    [float('nan')])))
    prob_arou = float(np.mean(algo_data.get('caisr_prob_arous', [float('nan')])))

    prob_w    = prob_w    if prob_w    <= 1.0 else float('nan')
    prob_n3   = prob_n3   if prob_n3   <= 1.0 else float('nan')
    prob_arou = prob_arou if prob_arou <= 1.0 else float('nan')

    return np.array([
        ahi_auto, arousal_idx, limb_idx,
        w_pct, n1_pct, n2_pct, n3_pct, rem_pct, efficiency,
        prob_w, prob_n3, prob_arou,
    ], dtype=float)