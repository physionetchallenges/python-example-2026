# features/sample_weighting.py
# Team Narnia — PhysioNet Challenge 2026
#
# Entry 6 (2026-07-16/19): compute_value_weighted_sample_weights(), promoted
# from tools/reg_sweep.py (where it was developed and validated on Kaggle —
# see learning_log.md / learning_log_3, 2026-07-16 entries) into a shared
# features/ module, same precedent as AgeResidualizer (features/age_residuals.py):
# one definition, imported by both team_code.py (the real submission path)
# and tools/reg_sweep.py (the validation harness it was proven in), so the
# two can never silently diverge the way loso_cv.py's hand-copied Pipeline
# once did (see features/pipeline.py header for that incident).
#
# Validated result this function is responsible for (alpha=1.0, on top of
# C=0.001): +12.9% relative reward vs. C=0.001 alone, AUROC flat (max 0.21σ
# across all 3 folds). Reproduced bit-for-bit on an independent Kaggle run
# (learning_log_3, 2026-07-16 final entry). Combined with the C=0.001 change:
# +52% relative reward vs. the original shipped Entry 5 baseline, AUROC flat
# throughout the entire chain.

import numpy as np

from evaluate_model import compute_prevalence


def _lookup_prevalence_train(age_to_prevalence, age):
    """Same nearest-key fallback logic as tools/test_age_banded_threshold.py's
    _lookup_prevalence — kept as a local duplicate rather than a cross-file
    import, matching reg_sweep.py's original standalone-on-Kaggle discipline."""
    key = round(age)
    if key in age_to_prevalence:
        return age_to_prevalence[key]
    nearest = min(age_to_prevalence.keys(), key=lambda k: abs(k - age))
    return age_to_prevalence[nearest]


def compute_value_weighted_sample_weights(y_train, age_train, alpha):
    """
    Added 2026-07-16 (tools/reg_sweep.py), promoted here 2026-07-19 for
    Entry 6. Builds a per-training-sample weight array that layers a
    reward-VALUE-aware boost on top of whatever class_weight='balanced'
    already does (LOGREG_PARAMS — confirmed already active in every logreg
    config tested this project; this is an ADDITION, not a replacement).
    sklearn composes class_weight and an explicit sample_weight array
    multiplicatively, so both apply together.

    ONLY positive-class samples get boosted — negatives keep weight 1.0
    regardless of age, matching the actual intent (prioritize getting
    high-value positives right, not reweighting the whole population by
    age indiscriminately).

    CRITICAL LEAKAGE DISCIPLINE: age_to_prevalence here is fit using ONLY
    y_train/age_train. At LOSO-validation time (reg_sweep.py) that means
    only the patients in THIS fold's training set, never the held-out test
    fold. At real submission time (team_code.py) there is no fold rotation
    at all — train_model() trains once on whatever training set the
    organizers provide, so passing that full training set here is itself
    the correct, leakage-safe usage; there is no held-out slice to leak
    from in production. This is deliberately different from how
    compute_prevalence is used elsewhere in this project: reward SCORING
    correctly uses the FULL population as its reference (confirmed
    2026-07-09 finding), because that's evaluating an already-fixed
    decision against reality. This is different — it directly shapes what
    the MODEL LEARNS from these exact training patients, so using anything
    beyond the training set's own data here would leak test-correlated
    information into training, the same class of mistake AgeResidualizer's
    train-fold-only fitting discipline exists to prevent.

    alpha=0.0 reproduces IDENTICAL behavior to no weighting at all (returns
    all-ones) — backward compatible, matches every previously logged result
    exactly when alpha isn't explicitly set.
    """
    if alpha == 0.0:
        return np.ones(len(y_train))

    age_to_prevalence_train = compute_prevalence(age_train, y_train, age_train, gap=2)
    prevalences = np.array([_lookup_prevalence_train(age_to_prevalence_train, a)
                             for a in age_train])
    value = (1.0 / prevalences) - 1.0
    # Normalize using this training set's own min/max — not any global
    # reference — same train-only discipline as the prevalence fit above.
    value_norm = (value - value.min()) / (value.max() - value.min() + 1e-12)

    weights = np.ones(len(y_train))
    pos_mask = (y_train == 1)
    weights[pos_mask] = 1.0 + alpha * value_norm[pos_mask]
    return weights