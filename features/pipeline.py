# features/pipeline.py
# Shared model pipeline construction — single source of truth for
# team_code.py (the shipped submission) AND loso_cv.py (the validation
# harness).
#
# Root cause this file fixes (2026-07-03): loso_cv.py had its own
# hand-copied Pipeline() construction, predating AgeResidualizer (Entry 4).
# When AgeResidualizer was added to team_code.py, loso_cv.py's copy was
# never updated — LOSO silently kept validating the OLD 48-feature
# pipeline while team_code.py had already moved to 50. Caught because a
# post-"Entry 4" LOSO run matched the logged Entry 3 numbers to 4 decimal
# places across every metric, including per-fold top-feature importance —
# if AgeResidualizer had actually been active, at least some numerical
# drift would be expected (different feature count into the imputer,
# different fitted trees).
#
# Fix: both team_code.py and loso_cv.py now import build_pipeline() from
# here instead of each constructing their own Pipeline inline. There is
# now exactly ONE place that defines "what the model is" — changing it
# here changes it everywhere that matters, and it is structurally
# impossible for the validation harness and the submission to silently
# diverge again the way they just did.
#
# Lives under features/ (not repo root) deliberately: check_submission_files.py
# only resolves `from features.X import ...` / `from features import ...`
# into required-file paths. A root-level shared module would instead be
# misclassified as a third-party pip package by that script's import parser
# and flagged as "missing from requirements.txt" — a confusing false
# warning for something that isn't a package at all.

# features/pipeline.py
# Shared model pipeline construction — single source of truth for
# team_code.py AND loso_cv.py / reg_sweep.py.
#
# 2026-07-08 addition: build_logreg_pipeline(), for the Ridge/Lasso/
# ElasticNet regularization sweep. Reuses the same AgeResidualizer ->
# SimpleImputer -> [CalibratedClassifierCV] skeleton as build_pipeline(),
# swapping XGBClassifier for LogisticRegression, with two changes that
# matter specifically for a regularization sweep:
#
#   1. Added StandardScaler between the imputer and the classifier. This
#      was NOT verified to already exist in whatever build_logreg_pipeline()
#      produced the 0.6002 large-set LOSO result (learning_log.md,
#      2026-07-07) — that code wasn't available when this was written.
#      A C sweep is not meaningfully comparable across features without
#      scaling first: Age (~0-100), BMI (~15-50), event-rate features
#      (~0-30/hr), and ratio features (~0-3) sit on wildly different
#      scales, so an unscaled C controls regularization strength
#      inconsistently across coefficients. IF a scaler already existed in
#      the original build_logreg_pipeline(), the 0.6002 baseline was
#      already fit this way and nothing changes. If it did NOT exist,
#      this sweep's results are not directly comparable to that 0.6002
#      number at face value — confirm which case you're in before treating
#      a sweep win as a clean improvement over the existing baseline.
#   2. class_weight='balanced' instead of an XGBoost-style scale_pos_weight
#      — sklearn's LogisticRegression equivalent, same "derive from actual
#      data, don't hardcode" principle already established for XGBoost
#      (learning_log.md, 2026-06-30).
#
# solver='saga' is required for L1 and ElasticNet penalties; used
# uniformly (including for L2) so penalty type alone varies across the
# sweep, not solver + penalty together.
# features/pipeline.py
# Shared model pipeline construction — single source of truth for
# team_code.py (the shipped submission) AND loso_cv.py (the validation
# harness).
#
# Root cause this file fixes (2026-07-03): loso_cv.py had its own
# hand-copied Pipeline() construction, predating AgeResidualizer (Entry 4).
# When AgeResidualizer was added to team_code.py, loso_cv.py's copy was
# never updated — LOSO silently kept validating the OLD 48-feature
# pipeline while team_code.py had already moved to 50. Caught because a
# post-"Entry 4" LOSO run matched the logged Entry 3 numbers to 4 decimal
# places across every metric, including per-fold top-feature importance —
# if AgeResidualizer had actually been active, at least some numerical
# drift would be expected (different feature count into the imputer,
# different fitted trees).
#
# Fix: both team_code.py and loso_cv.py now import build_pipeline() from
# here instead of each constructing their own Pipeline inline. There is
# now exactly ONE place that defines "what the model is" — changing it
# here changes it everywhere that matters, and it is structurally
# impossible for the validation harness and the submission to silently
# diverge again the way they just did.
#
# Lives under features/ (not repo root) deliberately: check_submission_files.py
# only resolves `from features.X import ...` / `from features import ...`
# into required-file paths. A root-level shared module would instead be
# misclassified as a third-party pip package by that script's import parser
# and flagged as "missing from requirements.txt" — a confusing false
# warning for something that isn't a package at all.

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Compatibility shim: cv='prefit' was removed in scikit-learn 1.6+ in favor
# of wrapping the fitted estimator in FrozenEstimator. This repo's pinned
# requirements.txt uses scikit-learn==1.5.2 (pre-FrozenEstimator), but
# loso_cv.py is also run in environments with newer scikit-learn (e.g. the
# Kaggle Notebook used for the large-set run, which had 1.8.0 and rejected
# cv='prefit' outright). Try the modern import; fall back to the legacy
# string if it's unavailable, so PrefitCalibratedModel works either way
# without needing to know which environment it's running in.
try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN_ESTIMATOR = True
except ImportError:
    _HAS_FROZEN_ESTIMATOR = False

from features.age_residuals import AgeResidualizer
from features import IDX_AGE, IDX_CA_RATE, IDX_EEG_VAR_REM_WAKE

# XGBoost hyperparameters, frozen since Entry 2 (depth 4->3, added L1/L2 +
# min_child_weight, tightened subsample/colsample after LOSO ablation showed
# the Entry 1 config overfit to within-site patterns). Do not tune these
# without a LOSO regression check — see learning_log.md, 2026-07-01.
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_alpha=0.1,
    reg_lambda=2.0,
    min_child_weight=5,
    random_state=42,
    eval_metric='auc',
    verbosity=0,
)


def _scale_pos_weight(y_train):
    y_train = np.asarray(y_train)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    return n_neg / n_pos if n_pos > 0 else 1.0


def build_pipeline(y_train, calibrated=True, use_age_residual=True,
                    calibration_ensemble=True, xgb_overrides=None):
    """
    Builds the full Narnia_ML model pipeline:

        [AgeResidualizer ->] SimpleImputer(median) -> XGBoost [+ CalibratedClassifierCV]

    y_train is needed up front (not deferred to .fit() time) only to
    compute scale_pos_weight, matching every prior entry's convention of
    deriving it from the actual training split rather than hardcoding it.

    calibrated=True (default — matches team_code.py / Entry 3+): wraps
        XGBoost in CalibratedClassifierCV(method='sigmoid', cv=5).
    calibrated=False: uncalibrated single XGBoost. Used by loso_cv.py's
        run_ablation() and any fast/uncalibrated fold metrics, where the
        extra ~5x training cost of calibration buys nothing.

    calibration_ensemble=True (default — matches shipped team_code.py):
        CalibratedClassifierCV's own `ensemble=True` behavior — 5 models
        trained on different folds, their 5 calibrated outputs averaged
        at predict time.
    calibration_ensemble=False (2026-07-06): passes `ensemble=False` to
        CalibratedClassifierCV. RULED OUT as a fix for the large-set
        reward collapse (2026-07-06 entry — identical raw scores, no
        measurable reward/AUROC change vs ensemble=True). Kept available
        for completeness, not because it's still a live hypothesis.

    use_age_residual=True (default — matches team_code.py / Entry 4):
        includes the AgeResidualizer step (2 extra features, 48 -> 50).
    use_age_residual=False (2026-07-06): builds the Entry-3-EQUIVALENT
        pipeline. Also largely ruled out as a driver of the large-set
        reward collapse (2026-07-06 attribution entry: negligible effect
        at both scales) — kept for completeness/future feature work, not
        because it's still the leading hypothesis for that specific problem.

    xgb_overrides=None (default): XGBoost uses XGB_PARAMS unchanged, the
        config validated via LOSO at Entry 2 (small-scale n). Pass a dict
        to override specific keys, e.g. {'reg_lambda': 10.0,
        'min_child_weight': 25} — added 2026-07-06 for loso_cv.py's
        --tune-hyperparams sweep, testing whether XGB_PARAMS (never
        re-validated above ~900-patient folds) is under-regularized at
        large-scale fold sizes (~1,000-5,500 patients). See
        learning_log.md, 2026-07-06 (raw score inflation entry) for the
        finding motivating this: the I0006 holdout fold's raw
        (pre-calibration) XGBoost output shifted from median 0.12 (small
        scale) to median 0.69 (large scale) — a base-model phenomenon,
        confirmed independent of calibration architecture. team_code.py
        never passes this — it always wants the validated XGB_PARAMS —
        this exists for loso_cv.py's comparison runs only.

    Returns an UNFITTED sklearn Pipeline. Caller is responsible for
    calling .fit(X_train, y_train) — this function never fits anything,
    so the same discipline (fit on train, freeze for inference/LOSO-test)
    that AgeResidualizer and SimpleImputer already enforce internally is
    also enforced at the call-site level: nothing in this function can
    accidentally see test data.
    """
    spw = _scale_pos_weight(y_train)
    params = {**XGB_PARAMS, **(xgb_overrides or {})}
    xgb = XGBClassifier(scale_pos_weight=spw, **params)

    classifier = (
        CalibratedClassifierCV(xgb, method='sigmoid', cv=5, ensemble=calibration_ensemble)
        if calibrated else xgb
    )

    steps = []
    if use_age_residual:
        steps.append(('age_residual', AgeResidualizer(
            age_idx=IDX_AGE,
            ca_rate_idx=IDX_CA_RATE,
            eeg_var_rem_wake_idx=IDX_EEG_VAR_REM_WAKE,
        )))
    steps.append(('imputer', SimpleImputer(strategy='median')))
    steps.append(('classifier', classifier))

    return Pipeline(steps)


# Logistic regression hyperparameters. class_weight='balanced' stands in
# for XGB_PARAMS' scale_pos_weight -- same purpose (correct for the ~7%
# positive rate), different mechanism; LogisticRegression has no
# scale_pos_weight argument. C is passed separately per-call (see
# build_logreg_pipeline) since it's the one parameter under active
# investigation, not frozen the way XGB_PARAMS is.
LOGREG_PARAMS = dict(
    penalty='l2',
    class_weight='balanced',
    max_iter=2000,
    random_state=42,
)


def build_logreg_pipeline(y_train, calibrated=True, use_age_residual=True,
                           calibration_ensemble=True, C=0.01,
                           penalty=None, l1_ratio=None, max_iter=None):
    """
    Logistic regression alternative to build_pipeline()'s XGBoost model.
    Added 2026-07-06 after the model-family diagnostic (loso_cv.py
    --diagnostic-models) showed logistic regression beating XGBoost's
    age-conditioned AUROC on BOTH large-set LOSO folds:

        Model              I0006 AUROC   S0001 AUROC   Mean
        xgboost_baseline   0.549         0.539         0.544
        logreg_strong_l2   0.615         0.585         0.600  (C=0.01)

    Mean 0.600 is the first result in the whole large-set investigation
    to clear the "real, not noise" bar established from the leaderboard's
    own noise-floor analysis (~north of 0.57-0.58). It also showed LESS
    severe raw-score inflation on the I0006 fold specifically (median
    0.47 vs XGBoost's 0.69) — see learning_log.md, 2026-07-06 (model
    diagnostic entry) for the full comparison. The hyperparameter sweep
    that ran alongside this diagnostic ruled out "XGBoost just needs more
    regularization" — every more-conservative XGBoost candidate made the
    I0006 inflation WORSE, not better — so this isn't a same-model tweak,
    it's a different model family that empirically behaves better on the
    exact problem under investigation.

    Same [AgeResidualizer ->] Imputer -> ... -> Classifier [+ Calibration]
    shape as build_pipeline(), with two necessary differences:
      - Adds a StandardScaler step after the imputer. Unlike tree-based
        XGBoost, a linear model is sensitive to the wildly different
        scales across this feature vector (e.g. Age in years vs. Hjorth
        ratios near 1.0) and won't behave sensibly without it.
      - class_weight='balanced' (see LOGREG_PARAMS) stands in for
        XGBoost's scale_pos_weight.

    C=0.01 (default) matches the diagnostic's best-performing candidate.
    Exposed as a parameter for further tuning — not yet swept the way
    XGB_PARAMS was; this is the first calibrated test of this model
    family, not a converged config.

    penalty/l1_ratio/max_iter (added 2026-07-09, for the Ridge/Lasso/
    ElasticNet regularization sweep): NEW — LOGREG_PARAMS hardcodes
    penalty='l2', so l1/elasticnet were never tried before this. Left as
    None by default so an un-parameterized call reproduces the EXACT
    original tested config (penalty='l2' from LOGREG_PARAMS, default
    solver, max_iter=2000) byte-for-byte — passing any of these three
    overrides that default. l1/elasticnet require solver='saga'
    (LOGREG_PARAMS' implicit lbfgs-default doesn't support them), so the
    solver is switched automatically only when a non-l2 penalty is
    requested — the confirmed-working l2 path never changes solver.

    calibrated / use_age_residual / calibration_ensemble: same meaning
    and defaults as build_pipeline(). Calibration mode was ruled out as
    the driver of the ORIGINAL XGBoost reward problem, but that finding
    doesn't automatically transfer to a different base model — worth
    re-checking cv5 vs cv5-single-curve on THIS model too if it looks
    promising, rather than assuming the earlier conclusion still applies.

    Returns an UNFITTED sklearn Pipeline. Caller is responsible for
    calling .fit(X_train, y_train) — same contract as build_pipeline().
    team_code.py does not use this function; it remains on build_pipeline()
    (XGBoost) unless/until a decision is logged in learning_log.md to
    switch, per this project's standing discipline.
    """
    steps = []
    if use_age_residual:
        steps.append(('age_residual', AgeResidualizer(
            age_idx=IDX_AGE,
            ca_rate_idx=IDX_CA_RATE,
            eeg_var_rem_wake_idx=IDX_EEG_VAR_REM_WAKE,
        )))
    steps.append(('imputer', SimpleImputer(strategy='median')))
    steps.append(('scaler', StandardScaler()))

    lr_params = dict(LOGREG_PARAMS)
    if max_iter is not None:
        lr_params['max_iter'] = max_iter
    if penalty is not None:
        lr_params['penalty'] = penalty
    if lr_params['penalty'] in ('l1', 'elasticnet'):
        lr_params['solver'] = 'saga'  # LOGREG_PARAMS' l2 default (lbfgs) can't do l1/elasticnet
        if lr_params['penalty'] == 'elasticnet':
            if l1_ratio is None:
                raise ValueError("penalty='elasticnet' requires l1_ratio in [0, 1].")
            lr_params['l1_ratio'] = l1_ratio

    logreg = LogisticRegression(C=C, **lr_params)
    classifier = (
        CalibratedClassifierCV(logreg, method='sigmoid', cv=5, ensemble=calibration_ensemble)
        if calibrated else logreg
    )
    steps.append(('classifier', classifier))

    return Pipeline(steps)


class PrefitCalibratedModel:
    """
    Alternative to build_pipeline(calibrated=True)'s CalibratedClassifierCV
    (..., cv=5) ensembling. Added 2026-07-06 after two independent pieces
    of evidence that cv=5's per-fold calibration curves may not transfer
    stably across training-set scale:

      1. Real leaderboard, Entry 2 -> Entry 3 (calibration added, nothing
         else changed): standard AUROC rose (0.746 -> 0.772) while
         age-conditioned AUROC and reward both fell, and accuracy dropped
         sharply (0.866 -> 0.803) with F-measure barely moving -- more
         patients being called positive, at a cost.
      2. Large-set LOSO (Entry 4 config): reward at the identical
         THRESHOLD=0.12 dropped from 0.1148 (small set) to 0.0726 (large
         set), traced to pooled specificity collapsing from ~93% to ~68%
         -- age-conditioned AUROC barely moved, so this is a calibrated-
         PROBABILITY-magnitude problem, not a ranking problem.

    Mechanism under test: cv=5 (without cv='prefit') fits 5 separate
    XGBoost models on different 80% training slices and averages their
    independently-fit Platt curves. Each of those 5 curves is itself
    sensitive to whatever slice it happened to see, on top of the whole
    ensemble's sensitivity to the overall training set's size/composition
    (911 patients at small scale vs several thousand at large scale, per
    LOSO fold). THRESHOLD=0.12 is applied AFTER this mapping, so if the
    mapping itself shifts between training runs, the same fixed number
    stops meaning the same thing -- this is the leading explanation for
    both observations above.

    This class fits ONE XGBoost model on a training split, then fits ONE
    Platt curve (cv='prefit') on a SEPARATE held-out calibration split --
    removing the 5-curves-averaged-together source of instability
    entirely, so a comparison against build_pipeline(calibrated=True)
    isolates whether cv=5's ensembling specifically is the cause.

    NOT a sklearn Pipeline: cv='prefit' calibration inherently needs two
    different data slices (one to fit the base estimator, a separate one
    to fit the calibrator), which doesn't fit Pipeline's contract of one
    X flowing through every step via a single .fit(X, y) call. Exposes
    .fit(X, y) / .predict_proba(X) with the same signatures a fitted
    Pipeline would, so it's a drop-in replacement at loso_cv.py's call
    sites without those call sites needing to know which mode is active.

    Caveat: splitting off a separate calibration slice costs real training
    data, which matters more at small scale. At the large training set's
    per-fold sizes (~1,000-5,500), a calib_fraction=0.2 split still leaves
    a healthy number of positives on both sides. At the SMALL set's sizes
    (~250-900 per fold, 28-64 positives), the same split leaves considerably
    less on each side -- if comparing this against cv=5 on the small set
    specifically, read the result with that in mind rather than treating
    it as a clean apples-to-apples comparison of calibration mode alone.
    """

    def __init__(self, use_age_residual=True, calib_fraction=0.2, random_state=42):
        self.use_age_residual = use_age_residual
        self.calib_fraction = calib_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        idx_train, idx_calib = train_test_split(
            np.arange(len(y)), test_size=self.calib_fraction,
            stratify=y, random_state=self.random_state)

        X_train_raw, y_train = X[idx_train], y[idx_train]
        X_calib_raw, y_calib = X[idx_calib], y[idx_calib]

        # Preprocessing (age residual + imputer) fit ONLY on the training
        # slice -- same train-only-fit discipline as everywhere else in
        # this pipeline. The calibration slice only ever gets .transform()'d.
        steps = []
        if self.use_age_residual:
            steps.append(('age_residual', AgeResidualizer(
                age_idx=IDX_AGE,
                ca_rate_idx=IDX_CA_RATE,
                eeg_var_rem_wake_idx=IDX_EEG_VAR_REM_WAKE,
            )))
        steps.append(('imputer', SimpleImputer(strategy='median')))
        self.preprocessing_ = Pipeline(steps)

        X_train = self.preprocessing_.fit_transform(X_train_raw)
        X_calib = self.preprocessing_.transform(X_calib_raw)

        spw = _scale_pos_weight(y_train)
        xgb = XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
        xgb.fit(X_train, y_train)
        self.base_estimator_ = xgb  # kept for feature_importances_ access

        if _HAS_FROZEN_ESTIMATOR:
            self.calibrated_ = CalibratedClassifierCV(FrozenEstimator(xgb), method='sigmoid')
        else:
            self.calibrated_ = CalibratedClassifierCV(xgb, method='sigmoid', cv='prefit')
        self.calibrated_.fit(X_calib, y_calib)
        return self

    def predict_proba(self, X):
        X_t = self.preprocessing_.transform(np.asarray(X, dtype=float))
        return self.calibrated_.predict_proba(X_t)

    @property
    def feature_importances_(self):
        return self.base_estimator_.feature_importances_


def extract_fitted_coefficients(fitted_pipeline):
    """
    ADDED 2026-07-09, not part of the original file — pulls linear
    coefficients back out of a fitted build_logreg_pipeline() Pipeline,
    for the coefficient-stability-across-folds check. Handles both
    calibrated=True (CalibratedClassifierCV wraps N cloned+refit estimators
    — averages their coefficients) and calibrated=False.

    Returns a 1D np.array in STANDARDIZED-feature space (coefficients on
    scaled features) — comparable across folds/features for a stability
    check, but not directly interpretable as raw-unit effect sizes.
    """
    clf = fitted_pipeline.named_steps['classifier']
    if isinstance(clf, CalibratedClassifierCV):
        coefs = np.stack([cc.estimator.coef_[0] for cc in clf.calibrated_classifiers_])
        return coefs.mean(axis=0)
    return clf.coef_[0]