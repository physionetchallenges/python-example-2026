# features/caisr_enriched.py
# Enriched CAISR annotation features — features 71-81
#
# Iteration history:
#   v1 (2026-06-30): First enrichment pass — respiratory breakdown,
#                    stage-conditional AHI, N3 gradient, spontaneous
#                    arousal index, N3 confidence entropy  [11 features]
#
# Planned (not yet implemented):
#   - PLM periodicity index (PLM / isolated LM ratio)        Week 3+
#   - CAISR no-arousal confidence entropy (caisr_prob_no-ar)  Week 3+
#   - Stage transition rate                                   Week 3+
#   - REM latency                                             Week 3+

import numpy as np


def extract_caisr_enriched_features(algo_data):
    """
    Enriched CAISR features requiring temporal cross-referencing
    between annotation channels.

    Returns np.ndarray of length 11:
        [0]  OA rate (events/hr)              — obstructive apnea
        [1]  CA rate (events/hr)              — central apnea
        [2]  HY rate (events/hr)              — hypopnea
        [3]  RERA rate (events/hr)            — effort-related arousals
        [4]  CA / total AHI ratio             — neurological apnea fraction
        [5]  REM-AHI (events/hr)             — apneas during REM
        [6]  NREM-AHI (events/hr)            — apneas during NREM
        [7]  REM-AHI / NREM-AHI ratio        — REM-predominant OSA marker
        [8]  N3 first-half / second-half ratio — temporal N3 gradient
        [9]  Spontaneous arousal index        — non-respiratory arousals/hr
        [10] N3 confidence entropy            — slow-wave staging ambiguity
    """
    if not algo_data:
        return np.full(11, float('nan'))

    resp       = algo_data.get('resp_caisr',    np.array([]))
    arousal    = algo_data.get('arousal_caisr', np.array([]))
    stages_raw = algo_data.get('stage_caisr',   np.array([]))

    total_hours_resp = len(resp) / 3600.0 if len(resp) > 0 else 0.0
    valid_stages     = (stages_raw[stages_raw < 9.0]
                        if len(stages_raw) > 0 else np.array([]))

    # ── Event-type counting helper ────────────────────────────────────────────
    def count_event_type(resp_sig, event_code, t_hours):
        if len(resp_sig) == 0 or t_hours <= 0:
            return float('nan')
        binary = (np.asarray(resp_sig) == event_code).astype(int)
        edges  = np.diff(binary, prepend=0)
        return np.count_nonzero(edges == 1) / t_hours

    def count_events(sig, t_hours):
        if len(sig) == 0 or t_hours <= 0:
            return float('nan')
        binary = (np.asarray(sig) > 0).astype(int)
        edges  = np.diff(binary, prepend=0)
        return np.count_nonzero(edges == 1) / t_hours

    # [0-3] Respiratory event breakdown
    # resp_caisr values: 0=none 1=OA 2=CA 3=MA 4=HY 5=RERA
    oa_rate   = count_event_type(resp, 1, total_hours_resp)
    ca_rate   = count_event_type(resp, 2, total_hours_resp)
    hy_rate   = count_event_type(resp, 4, total_hours_resp)
    rera_rate = count_event_type(resp, 5, total_hours_resp)

    # [4] CA / total AHI ratio
    ahi_total = count_events(resp, total_hours_resp)
    ca_total_ratio = (ca_rate / ahi_total
                      if (not np.isnan(ca_rate) and not np.isnan(ahi_total)
                          and ahi_total > 0)
                      else float('nan'))

    # [5-7] Stage-conditional AHI
    # Requires upsampling stage_caisr (30s epochs → 1s) before cross-referencing
    rem_ahi = nrem_ahi = rem_nrem_ratio = float('nan')
    if len(valid_stages) > 0 and len(resp) > 0:
        stage_1s  = np.repeat(valid_stages, 30)[:len(resp)]
        rem_mask  = (stage_1s == 4)
        nrem_mask = (stage_1s >= 1) & (stage_1s <= 3)

        rem_hours  = float(rem_mask.sum())  / 3600.0
        nrem_hours = float(nrem_mask.sum()) / 3600.0

        if rem_hours > 0:
            resp_rem  = np.where(rem_mask,  resp[:len(rem_mask)],  0)
            edges_rem = np.diff((resp_rem > 0).astype(int), prepend=0)
            rem_ahi   = np.count_nonzero(edges_rem == 1) / rem_hours

        if nrem_hours > 0:
            resp_nrem  = np.where(nrem_mask, resp[:len(nrem_mask)], 0)
            edges_nrem = np.diff((resp_nrem > 0).astype(int), prepend=0)
            nrem_ahi   = np.count_nonzero(edges_nrem == 1) / nrem_hours

        if (not np.isnan(rem_ahi) and not np.isnan(nrem_ahi)
                and nrem_ahi > 0):
            rem_nrem_ratio = rem_ahi / nrem_ahi

    # [8] N3 temporal gradient
    # Healthy sleep front-loads N3 (ratio >> 1).
    # CI-risk pattern: ratio compressed toward 1.
    #
    # Bug fixed (2026-07-01): original code used 1e-6 as denominator guard,
    # producing values of 20,000-150,000 when n3_second == 0 (common in
    # elderly / CI+ patients who have no N3 in the second half of the night).
    # Fix: return NaN when n3_second < 1% (degenerate case — not meaningful
    # as a ratio), and cap at 10.0 for remaining edge cases.
    if len(valid_stages) > 1:
        mid       = len(valid_stages) // 2
        n3_first  = float(np.mean(valid_stages[:mid] == 1))
        n3_second = float(np.mean(valid_stages[mid:]  == 1))
        if n3_second < 0.01:
            # No meaningful N3 in second half — gradient is not defined as ratio.
            # Use a signed indicator instead: positive = any first-half N3 exists.
            n3_gradient = float('nan')
        else:
            n3_gradient = min(n3_first / n3_second, 10.0)  # cap at 10
    else:
        n3_gradient = float('nan')

    # [9] Spontaneous arousal index — REDESIGNED v2 (2026-07-01)
    # Original: computed per total recording hour → confounded by REM amount
    # (CI+ have less REM → fewer arousals regardless of biology).
    # Fix: compute per NREM hour only — removes the REM confound.
    # Spontaneous = arousals not coincident with a respiratory event (±15s).
    spont_arousal_idx = float('nan')
    if len(arousal) > 0 and len(resp) > 0:
        n_1s = min(len(arousal) // 2, len(resp))
        if n_1s > 0:
            ar_reshaped = arousal[:n_1s * 2].reshape(-1, 2)
            arousal_1s  = (ar_reshaped.max(axis=1) > 0).astype(int)
            resp_1s     = (resp[:n_1s] > 0).astype(int)
            edges_ar    = np.diff(arousal_1s, prepend=0)
            ar_starts   = np.where(edges_ar == 1)[0]
            n_resp_coincident = 0
            for s in ar_starts:
                window = resp_1s[max(0, s - 5):min(n_1s, s + 15)]
                if len(window) > 0 and window.any():
                    n_resp_coincident += 1
            n_spont = max(0, len(ar_starts) - n_resp_coincident)

            # Use NREM hours as denominator (not total recording hours)
            # NREM mask from stage_caisr upsampled to 1s resolution
            if len(valid_stages) > 0:
                stage_1s_ar   = np.repeat(valid_stages, 30)[:n_1s]
                nrem_mask_ar  = (stage_1s_ar >= 1) & (stage_1s_ar <= 3)
                nrem_hours_ar = float(nrem_mask_ar.sum()) / 3600.0
                if nrem_hours_ar > 0:
                    spont_arousal_idx = n_spont / nrem_hours_ar
            else:
                # Fallback: use total recording hours if no stage data
                t_hours_1s = n_1s / 3600.0
                if t_hours_1s > 0:
                    spont_arousal_idx = n_spont / t_hours_1s

    # [10] N3 confidence entropy
    # Binary entropy on caisr_prob_n3: higher = more ambiguous staging.
    # Lower entropy = CAISR is confident = cleaner slow waves.
    n3_entropy = float('nan')
    prob_n3_arr = algo_data.get('caisr_prob_n3', np.array([]))
    if len(prob_n3_arr) > 0:
        p = np.clip(np.asarray(prob_n3_arr, dtype=float), 1e-9, 1.0 - 1e-9)
        h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        n3_entropy = float(np.mean(h))

    return np.array([
        oa_rate, ca_rate, hy_rate, rera_rate,
        ca_total_ratio,
        rem_ahi, nrem_ahi, rem_nrem_ratio,
        n3_gradient,
        spont_arousal_idx,
        n3_entropy,
    ], dtype=float)