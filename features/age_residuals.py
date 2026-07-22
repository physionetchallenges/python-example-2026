# features/age_residuals.py
# Age-residualized features — Entry 4
#
# Origin: 2026-07-02 age-residualized EDA (see learning_log.md). CA_rate and
# EEG_var_REM_Wake were previously written off as uninformative in the raw
# Phase 1 EDA (AUROC ~0.52, BH-FDR p_adj > 0.7 both) — but regressing Age out
# of every feature and re-testing the residual showed both carry real signal
# that raw testing couldn't see:
#
#     Feature            Raw AUROC / p_adj      Residual AUROC / p_adj
#     CA_rate             0.520 / 0.805           0.622 / 0.0075
#     EEG_var_REM_Wake    0.520 / 0.808           0.618 / 0.0098
#
# Age is available at real inference time (unlike the ICD-subtype features
# ruled out earlier for exactly that reason) — this is feature engineering,
# not leakage.
#
# Fitting discipline: identical to SimpleImputer. The linear age-regression
# coefficient for each source feature is fit ONCE on training data (inside
# AgeResidualizer.fit(), called by Pipeline.fit()) and frozen. transform()
# only ever applies the already-fitted coefficients — it must never re-fit,
# including at inference time on a single patient.
#
# Source features (already present earlier in the vector — see
# features/__init__.py for the index constants used to locate them):
#   Age                 index 0                 (features/demographic.py)
#   CA_rate             IDX_ENRICHED_START + 1  (features/caisr_enriched.py)
#   EEG_var_REM_Wake    IDX_RATIO_START + 11    (features/physiological_ratios.py)
#
# These 2 new features are APPENDED at the end of the vector (indices 48-49)
# rather than inserted mid-vector, so the existing demo/base/enriched/ratio
# index constants don't shift. Total feature count: 48 -> 50.

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

N_AGE_RESIDUAL_FEATURES = 2
_MIN_FIT_ROWS = 10


class AgeResidualizer(BaseEstimator, TransformerMixin):
    def __init__(self, age_idx, ca_rate_idx, eeg_var_rem_wake_idx):
        self.age_idx = age_idx
        self.ca_rate_idx = ca_rate_idx
        self.eeg_var_rem_wake_idx = eeg_var_rem_wake_idx

    @staticmethod
    def _fit_one(age, feature):
        mask = ~(np.isnan(age) | np.isnan(feature))
        if int(mask.sum()) < _MIN_FIT_ROWS:
            return 0.0, 0.0
        slope, intercept = np.polyfit(age[mask], feature[mask], 1)
        return float(slope), float(intercept)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        age = X[:, self.age_idx]
        self.ca_rate_slope_, self.ca_rate_intercept_ = self._fit_one(
            age, X[:, self.ca_rate_idx])
        self.eeg_slope_, self.eeg_intercept_ = self._fit_one(
            age, X[:, self.eeg_var_rem_wake_idx])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        age = X[:, self.age_idx]
        ca_rate_resid = X[:, self.ca_rate_idx] - (
            self.ca_rate_intercept_ + self.ca_rate_slope_ * age)
        eeg_resid = X[:, self.eeg_var_rem_wake_idx] - (
            self.eeg_intercept_ + self.eeg_slope_ * age)
        return np.hstack([X, ca_rate_resid.reshape(-1, 1), eeg_resid.reshape(-1, 1)])