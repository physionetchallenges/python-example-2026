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


# ── Wake fragmentation — NOT wired into extract_caisr_enriched_features ──
# Standalone, ablation-testable-first, per the same discipline as
# features/spectral.py's candidates. Hypothesis (learning_log.md,
# 2026-07-12): total wake time (W_pct) and arousal rate (Arousal_idx,
# a brief-event construct) don't distinguish "many short wake bouts"
# from "few long wake bouts" at the same total wake time — this
# fragmentation-vs-duration distinction, especially jointly with REM%,
# is hypothesized to carry CI-risk signal not captured by either
# existing feature.
#
# CAISR-annotation-only — no raw physiological EDF needed for THIS
# feature specifically. (Note: a full pipeline ablation against the
# shipped 48-feature baseline still needs phys_data, since
# physiological_ratios.py's block does — the raw-EDF-free speedup only
# applies to computing this feature itself / a standalone univariate
# signal check, not to a full baseline-vs-with-feature model comparison.)

N_WAKE_FRAGMENTATION_FEATURES = 2

# Minimum sleep-period length to trust the bout count/duration estimate —
# same spirit as physiological_ratios.py's _MIN_STAGE_SECONDS, expressed
# in epochs here since this function works entirely in 30s-epoch space
# (no upsampling to signal rate needed, unlike every spectral.py feature).
_MIN_SLEEP_EPOCHS = 4   # 2 minutes


def extract_wake_fragmentation_features(algo_data):
    """
    Wake fragmentation within WASO (wake after sleep onset).

    WASO window: from the first sleep-stage epoch (any of N1/N2/N3/REM)
    through the last sleep-stage epoch, inclusive. This excludes initial
    sleep-onset latency and any trailing wake after the final sleep
    epoch — standard WASO convention, chosen specifically to avoid
    ambiguity about whether to count a long "I'm done sleeping" wake
    tail at the end of the recording.

    Returns np.ndarray of length N_WAKE_FRAGMENTATION_FEATURES (2):
        [0]  wake_bout_count — number of discrete contiguous wake
             episodes within the WASO window. Uses the same rising-edge
             counting technique as count_events() above, applied to the
             wake-stage binary mask instead of a respiratory/arousal
             signal.
        [1]  mean_wake_bout_duration — total WASO seconds / wake_bout_count.
             The feature the underlying hypothesis is actually about:
             long total wake time with LOW bout count means few, long
             consolidated wake episodes rather than fragmented brief
             awakenings, at the same total wake time.

    NaN fallback:
        - Missing CAISR annotations → both NaN
        - No sleep-stage epochs found at all → both NaN
        - WASO window shorter than _MIN_SLEEP_EPOCHS → both NaN
        - Zero wake bouts in the WASO window (fully consolidated sleep,
          a real and valid outcome) → wake_bout_count=0 (a real value,
          not NaN), mean_wake_bout_duration=NaN (0/0 is genuinely
          undefined, not zero)
    """
    if not algo_data:
        return np.full(N_WAKE_FRAGMENTATION_FEATURES, float('nan'))

    stages_raw   = algo_data.get('stage_caisr', np.array([]))
    valid_stages = stages_raw[stages_raw < 9.0] if len(stages_raw) > 0 else np.array([])

    if len(valid_stages) == 0:
        return np.full(N_WAKE_FRAGMENTATION_FEATURES, float('nan'))

    sleep_mask = (valid_stages >= 1) & (valid_stages <= 4)
    if not sleep_mask.any():
        return np.full(N_WAKE_FRAGMENTATION_FEATURES, float('nan'))

    sleep_indices = np.where(sleep_mask)[0]
    first_sleep, last_sleep = sleep_indices[0], sleep_indices[-1]

    waso_window = valid_stages[first_sleep:last_sleep + 1]
    if len(waso_window) < _MIN_SLEEP_EPOCHS:
        return np.full(N_WAKE_FRAGMENTATION_FEATURES, float('nan'))

    wake_binary = (waso_window == 5).astype(int)
    edges = np.diff(wake_binary, prepend=0)
    wake_bout_count = int(np.count_nonzero(edges == 1))

    total_waso_epochs = int(wake_binary.sum())
    total_waso_seconds = total_waso_epochs * 30.0

    if wake_bout_count > 0:
        mean_wake_bout_duration = total_waso_seconds / wake_bout_count
    else:
        mean_wake_bout_duration = float('nan')

    return np.array([wake_bout_count, mean_wake_bout_duration], dtype=float)


# ── REM latency — NOT wired into extract_caisr_enriched_features ─────────
# Planned since Week 2-3 (FEATURES.md, "Always" gate — not scale-gated
# like the spectral candidates), never implemented until now. Classic
# clinical sleep-architecture marker. Motivated by the 2026-07-11 MCI/age
# finding: MCI is the earlier, subtler-signal subtype, and REM latency
# shifts are a well-established early marker in the cognitive-decline
# literature — a stronger a priori case than any of the three raw-EDF
# spectral candidates tested 2026-07-12, none of which survived the full
# pipeline.

N_REM_LATENCY_FEATURES = 1


def extract_rem_latency_features(algo_data):
    """
    REM latency: time from sleep onset (first epoch in any sleep stage,
    N1/N2/N3/REM) to the first REM epoch, in minutes.

    Returns np.ndarray of length N_REM_LATENCY_FEATURES (1):
        [0]  REM latency (minutes)

    NaN fallback:
        - Missing CAISR annotations → NaN
        - No sleep at all → NaN (latency undefined without a sleep onset)
        - No REM occurs anywhere in the recording → NaN (a real clinical
          possibility — some CI+ patients may have zero REM — but
          "latency to an event that never happens" is undefined, not
          zero or infinite; do not silently encode it as either)
    """
    if not algo_data:
        return np.full(N_REM_LATENCY_FEATURES, float('nan'))

    stages_raw   = algo_data.get('stage_caisr', np.array([]))
    valid_stages = stages_raw[stages_raw < 9.0] if len(stages_raw) > 0 else np.array([])

    if len(valid_stages) == 0:
        return np.full(N_REM_LATENCY_FEATURES, float('nan'))

    sleep_mask = (valid_stages >= 1) & (valid_stages <= 4)
    if not sleep_mask.any():
        return np.full(N_REM_LATENCY_FEATURES, float('nan'))

    first_sleep_idx = np.where(sleep_mask)[0][0]

    rem_mask = (valid_stages == 4)
    if not rem_mask.any():
        return np.full(N_REM_LATENCY_FEATURES, float('nan'))

    first_rem_idx = np.where(rem_mask)[0][0]

    if first_rem_idx < first_sleep_idx:
        # Should not happen (REM before any sleep stage is detected is
        # incoherent given REM is itself a sleep stage) — guard against
        # a data/annotation anomaly rather than returning a negative
        # latency.
        return np.full(N_REM_LATENCY_FEATURES, float('nan'))

    latency_epochs = first_rem_idx - first_sleep_idx
    latency_minutes = latency_epochs * 30.0 / 60.0

    return np.array([latency_minutes], dtype=float)


# ── No-arousal confidence entropy — NOT wired in ──────────────────────────
# Planned since Week 2-3 (FEATURES.md, "Always" gate), never implemented.
# Mechanically identical to the existing, already-shipped N3_entropy
# feature (caisr_enriched.py index 10) — same binary-entropy calculation,
# applied to caisr_prob_no-ar instead of caisr_prob_n3. Cheapest possible
# thing to test since it reuses proven, working logic verbatim.

N_NO_AROUSAL_ENTROPY_FEATURES = 1


def extract_no_arousal_entropy_features(algo_data):
    """
    Binary entropy on caisr_prob_no-ar: higher = more ambiguous
    arousal/no-arousal staging confidence. Identical calculation to the
    shipped N3_entropy feature, different source channel.

    Returns np.ndarray of length N_NO_AROUSAL_ENTROPY_FEATURES (1).

    NaN fallback: missing CAISR annotations, or the caisr_prob_no-ar
    channel not present in this dataset (key name per FEATURES.md's
    original plan — not yet confirmed against real data; if this key
    doesn't match what's actually in the CAISR annotation EDFs, this
    will silently return NaN for every patient, which the NaN-rate
    check in the EDA script will surface immediately).
    """
    if not algo_data:
        return np.full(N_NO_AROUSAL_ENTROPY_FEATURES, float('nan'))

    prob_no_ar_arr = algo_data.get('caisr_prob_no-ar', np.array([]))
    if len(prob_no_ar_arr) == 0:
        return np.full(N_NO_AROUSAL_ENTROPY_FEATURES, float('nan'))

    p = np.clip(np.asarray(prob_no_ar_arr, dtype=float), 1e-9, 1.0 - 1e-9)
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

    return np.array([float(np.mean(h))], dtype=float)

# ── Gating design decision (2026-07-14) ───────────────────────────────────
# The 2026-07-12 learning_log.md entry describes a duration guard
# (_MIN_RECORDING_HOURS / _recording_too_short()) as added to this file.
# Confirmed against the actual repo: it was never implemented — this was
# a genuinely undecided design question at the time (exclude short
# recordings outright, vs. keep specific components that may still carry
# usable signal even when the whole-night measure doesn't), not a lost
# commit. Resolved here, but NOT by resurrecting a blanket duration cutoff.
#
# For NF1 specifically, duration is the wrong thing to gate on. The real
# constraint is estimability of the transition matrix itself: a short
# recording (e.g. ~12 minutes, ~24 epochs) yields ~23 transitions to
# populate a 5x5 matrix — most cells land at 0 or 1 count. The stationary
# distribution, solved via eigendecomposition of that matrix, isn't just
# "less confident" at that point; a sparse/disconnected empirical chain
# can produce a numerically unstable or meaningless eigenvector at
# eigenvalue 1. That's a different failure mode than "weak signal," and
# gating on wall-clock duration is a proxy for it, not the thing itself —
# a technically-short recording with a clean, well-sampled stage sequence
# would be discarded for no real reason under a duration cutoff.
#
# Fix: gate NF1 on total valid TRANSITION COUNT directly (_MIN_TRANSITIONS
# below), the same way NF2 already gates on arousal EVENT COUNT
# (_MIN_AROUSAL_EVENTS) rather than duration. This targets whether the
# component is estimable, not how long the file happens to be.
# _MIN_TRANSITIONS=40 is a starting value, not empirically validated —
# worth a quick synthetic check (does the stationary-distribution
# eigensolve behave reasonably at exactly 40 transitions?) before
# treating it as final.
#
# Given the original duration audit found effect sizes already near zero
# and only ~1.3% of small-set patients affected either way, this is a
# correctness fix, not expected to move the ablation result — but it's
# the right way to gate these two specific features regardless.

_MIN_TRANSITIONS = 40


# ── NF1: Markov stage-transition features — NOT wired in ─────────────────
# Per TEST_PLAN.md's dependency graph: RB1 came back null (2026-07-13,
# rem_latency at C=1.0, max |z|=0.13sigma across folds), which rules out
# the regularization-budget explanation for the 3-for-3 CAISR null
# pattern. Team decision: pursue branch (a) — test NF1+NF2 as a COMBINED
# feature set rather than one-at-a-time, since single-feature ablation
# is structurally blind to interaction effects. This deliberately breaks
# the project's one-variable-per-test discipline, per TEST_PLAN.md's
# explicit allowance for this branch.
#
# CAISR-annotation-only. Stage codes per this project's established
# convention (see extract_rem_latency_features, extract_wake_fragmentation
# _features above): Wake=5, N1=3, N2=2, N3=1, REM=4.

N_MARKOV_FEATURES = 9

# State code -> index mapping, fixed order: Wake, N1, N2, N3, REM.
_STATE_CODES = [5, 3, 2, 1, 4]
_STATE_LABELS = ['Wake', 'N1', 'N2', 'N3', 'REM']
_STATE_IDX = {code: i for i, code in enumerate(_STATE_CODES)}


def extract_markov_transition_features(algo_data):
    """
    Stage-transition dynamics via a first-order Markov chain fitted to
    the epoch-level stage sequence. Static stage percentages (already
    shipped: Wake_pct, N1_pct, N2_pct, N3_pct, REM_pct) don't capture
    HOW a patient moves between stages — this does.

    Returns np.ndarray of length N_MARKOV_FEATURES (9):
        [0] transition_entropy — Shannon entropy (nats) of the
            flattened joint transition-probability distribution
            P(s_t, s_t+1) over all 25 (5x5) state pairs. This is the
            entropy of the WHOLE transition matrix, not a per-row
            average — a deliberate choice: per-row entropy averaged by
            visit frequency would double-count the marginal stage
            distribution already captured by [4:9] below, whereas the
            joint-pair entropy captures how predictable/chaotic the
            SEQUENCE of transitions is, which is the actually-new
            information this feature is meant to add.
        [1] n3_to_wake_rate (events/hr) — reported empirically, NOT
            assumed rare or pathological a priori (per TEST_PLAN.md's
            explicit caution against over-framing this transition).
        [2] rem_to_n3_rate (events/hr) — same caution applies.
        [3] escalating_rate (events/hr) — rate of the exact 3-step
            sequence N3->N2->N1->Wake (4 consecutive epochs in that
            order), the "smooth de-escalation" pattern contrasted
            against the abrupt transitions above.
        [4:9] stationary distribution — the fitted Markov chain's
            stationary distribution over [Wake, N1, N2, N3, REM],
            summing to 1. Solved via the row-normalized transition
            matrix's left eigenvector at eigenvalue 1 (pi P = pi),
            NOT simply the raw marginal stage-epoch proportions — those
            are only equal in the infinite-sample/perfectly-ergodic
            limit. Falls back to raw marginal proportions if the
            eigen-solve fails (e.g. an absorbing/disconnected
            empirical chain from a short or unusual recording).

    REQUIRED before ablation-testing [4:9] as independent: check
    correlation against the existing Wake_pct/N1_pct/N2_pct/N3_pct/
    REM_pct features on the small set (per TEST_PLAN.md's NF1 internal
    dependency). Not computed here — this function only extracts the
    feature; the correlation check belongs in the ablation/EDA script
    so it's visible per-run, not buried in extraction code.

    NaN fallback (all 9 values):
        - Missing CAISR annotations
        - Fewer than 2 valid stage epochs (can't form even 1 transition)
        - Fewer than _MIN_TRANSITIONS (40) total valid transitions —
          gated on estimability of the transition matrix itself, not
          wall-clock duration (see module-level "Gating design decision"
          note above for why duration was rejected as the gate here).
    """
    if not algo_data:
        return np.full(N_MARKOV_FEATURES, float('nan'))

    stages_raw = algo_data.get('stage_caisr', np.array([]))
    valid_stages = stages_raw[stages_raw < 9.0] if len(stages_raw) > 0 else np.array([])

    mask = np.isin(valid_stages, _STATE_CODES)
    seq = valid_stages[mask]

    if len(seq) < 2:
        return np.full(N_MARKOV_FEATURES, float('nan'))

    n_transitions = len(seq) - 1
    if n_transitions < _MIN_TRANSITIONS:
        return np.full(N_MARKOV_FEATURES, float('nan'))

    total_hours = len(seq) * 30.0 / 3600.0

    cur_idx = np.array([_STATE_IDX[c] for c in seq[:-1]])
    nxt_idx = np.array([_STATE_IDX[c] for c in seq[1:]])

    trans_counts = np.zeros((5, 5))
    for i in range(len(cur_idx)):
        trans_counts[cur_idx[i], nxt_idx[i]] += 1

    total_trans = trans_counts.sum()
    if total_trans == 0:
        return np.full(N_MARKOV_FEATURES, float('nan'))

    # [0] Transition entropy — joint distribution over all (i,j) pairs.
    p_joint = trans_counts / total_trans
    flat_nonzero = p_joint.flatten()
    flat_nonzero = flat_nonzero[flat_nonzero > 0]
    transition_entropy = float(-np.sum(flat_nonzero * np.log(flat_nonzero)))

    # [1-2] Abrupt transition rates — reported as-is, no a priori framing.
    n3_to_wake_rate = trans_counts[_STATE_IDX[1], _STATE_IDX[5]] / total_hours
    rem_to_n3_rate  = trans_counts[_STATE_IDX[4], _STATE_IDX[1]] / total_hours

    # [3] Escalating rate — exact 4-epoch N3->N2->N1->Wake sequence.
    esc_count = 0
    for i in range(len(seq) - 3):
        if seq[i] == 1 and seq[i+1] == 2 and seq[i+2] == 3 and seq[i+3] == 5:
            esc_count += 1
    escalating_rate = esc_count / total_hours

    # [4-8] Stationary distribution via left eigenvector at eigenvalue 1.
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    p_rownorm = trans_counts / row_sums_safe

    stationary = None
    try:
        eigvals, eigvecs = np.linalg.eig(p_rownorm.T)
        best_idx = int(np.argmin(np.abs(eigvals - 1)))
        vec = np.real(eigvecs[:, best_idx])
        vec = np.clip(vec, 0, None)
        if vec.sum() > 1e-9:
            stationary = vec / vec.sum()
    except np.linalg.LinAlgError:
        stationary = None

    if stationary is None:
        # Fallback: raw marginal proportion, same state order.
        marginal_counts = np.array(
            [np.sum(seq == code) for code in _STATE_CODES], dtype=float)
        stationary = marginal_counts / marginal_counts.sum()

    return np.array([
        transition_entropy, n3_to_wake_rate, rem_to_n3_rate, escalating_rate,
        *stationary,
    ], dtype=float)


# ── NF2: Micro-arousal temporal clustering — NOT wired in ────────────────
# Same sequencing note as NF1: built for the combined-feature test
# (branch (a) after RB1's null), not a standalone single-feature
# ablation candidate.
#
# Gated on event count only (_MIN_AROUSAL_EVENTS), not duration — this
# was already the right design even before the 2026-07-14 gating
# correction above; CV and burst index need enough EVENTS to be
# meaningful, not enough wall-clock time. The duration-guard call this
# patch originally (incorrectly) included has been removed — the
# event-count gate below is sufficient on its own.
#
# ASSUMPTION FLAGGED: arousal_caisr's sample rate is inferred as 2Hz
# (0.5s/sample) from the existing spont_arousal_idx code's
# reshape(-1, 2) pairing against 1s-resolution resp, in
# extract_caisr_enriched_features above. This has NOT been independently
# confirmed against real data / a header spec. It affects burst_index's
# absolute "<3 minute" threshold; it does NOT affect the CV feature
# (unitless ratio, invariant to a constant sample-rate scaling error).
# Confirm before trusting burst_index specifically.

N_AROUSAL_CLUSTERING_FEATURES = 2
_MIN_AROUSAL_EVENTS = 5
_AROUSAL_SAMPLE_HZ = 2.0  # SEE ASSUMPTION FLAG ABOVE — confirm before trusting.


def extract_arousal_clustering_features(algo_data):
    """
    Temporal clustering of arousal events — a different statistical
    property of the arousal-event sequence than the rate-only
    Arousal_idx/Spontaneous_arousal_idx already shipped. Tightly
    clustered arousals (thalamocortical gating failure, hypothesized)
    vs. evenly-spaced arousals (benign) could carry signal a rate
    measure can't see.

    Returns np.ndarray of length N_AROUSAL_CLUSTERING_FEATURES (2):
        [0] cv_inter_arousal_interval — coefficient of variation
            (std/mean) of the time gaps between consecutive arousal
            event onsets. Unitless — robust to the sample-rate
            assumption above.
        [1] burst_index — fraction of inter-arousal intervals under
            3 minutes (180s). Depends on the sample-rate assumption
            above for its absolute threshold.

    REQUIRED before ablation-testing as independent: check correlation
    against Arousal_idx / Spontaneous_arousal_idx on the small set (per
    TEST_PLAN.md's NF2 internal dependency) — not computed here, same
    reasoning as NF1's collinearity note above.

    NaN fallback (both values):
        - Missing CAISR annotations
        - Fewer than _MIN_AROUSAL_EVENTS (5) arousal events detected —
          the sole gate for this feature; CV and burst fraction aren't
          meaningful from 1-4 events regardless of recording length.
    """
    if not algo_data:
        return np.full(N_AROUSAL_CLUSTERING_FEATURES, float('nan'))

    arousal = algo_data.get('arousal_caisr', np.array([]))
    if len(arousal) == 0:
        return np.full(N_AROUSAL_CLUSTERING_FEATURES, float('nan'))

    binary = (np.asarray(arousal) > 0).astype(int)
    edges = np.diff(binary, prepend=0)
    start_indices = np.where(edges == 1)[0]

    if len(start_indices) < _MIN_AROUSAL_EVENTS:
        return np.full(N_AROUSAL_CLUSTERING_FEATURES, float('nan'))

    times_sec = start_indices / _AROUSAL_SAMPLE_HZ
    intervals = np.diff(times_sec)

    if len(intervals) == 0:
        return np.full(N_AROUSAL_CLUSTERING_FEATURES, float('nan'))

    mu = float(np.mean(intervals))
    sigma = float(np.std(intervals))
    cv = sigma / mu if mu > 0 else float('nan')
    burst_index = float(np.mean(intervals < 180.0))

    return np.array([cv, burst_index], dtype=float)