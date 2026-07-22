# features/feature_selection.py
# Team Narnia — PhysioNet Challenge 2026
#
# Entry 8: drops 11 SIGN-FLIPPING features (coefficients that change
# direction depending on which 2 training sites fit the model — direct,
# measured evidence of site-specific overfitting rather than genuine
# biological signal; see the cross-fold coefficient analysis, 2026-07-19,
# and learning_log.md for the full derivation).
#
# This is "Stage 1" of the two-stage drop tested in tools/ablation_drop_
# signflip_features.py. Validated: +8.2% relative reward, mean age-
# conditioned AUROC 0.6459 -> 0.6711 (+0.0252), ALL THREE LOSO folds
# improved simultaneously (not a mixed result), cross-fold spread shrank
# 0.1264 -> 0.1145 (more stable, not just better on average — the
# specific signature the sign-flip theory predicted). Reproduced bit-
# for-bit across an independent Kaggle kernel restart.
#
# Stage 2 (CA_rate_age_residual, Race_Black/White/Asian) is deliberately
# NOT included here. CA_rate_age_residual carries separate validation
# history (Entry 4, BH-FDR significance on residual AUROC — a different
# question than raw-coefficient sign stability). The Race_* features
# carry real, documented demographic signal (dementia diagnosis odds
# genuinely differ by race in the published literature) that a numerically
# unstable one-hot coefficient does not necessarily invalidate. Stage 2
# showed a further, real improvement in LOSO (clears the pre-committed
# 1.0-sigma gate outright, vs. Stage 1's 0.97 just under it) but was held
# back given the extra scrutiny both warrant and the proximity of the
# deadline — not promoted to production as of Entry 8. Revisit only with
# a specific, separate justification for each, not just because Stage 1
# worked.
#
# Promoted into this shared features/ module (rather than living only in
# the ablation script) so team_code.py and any future validation run
# share exactly the same drop list — same precedent as AgeResidualizer
# and compute_value_weighted_sample_weights.

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from features import FEATURE_NAMES_50

STAGE1_DROP_FEATURES = [
    'BMI', 'EEG_mob_REM_Wake', 'RERA_rate', 'N1_pct', 'Spont_arousal_idx',
    'EEG_std_REM_Wake', 'EEG_cplx_REM_Wake', 'REM_NREM_AHI_ratio',
    'CA_rate', 'CA_total_ratio', 'EEG_zcr_REM_Wake',
]


class FeatureDropper(BaseEstimator, TransformerMixin):
    """
    Drops named columns from the 50-length (post-AgeResidualizer) vector.
    MUST be inserted immediately after the 'age_residual' pipeline step
    and before 'imputer' — AgeResidualizer's own source features (e.g.
    CA_rate, used to compute CA_rate_age_residual, which IS kept in
    Stage 1) need to still be present when AgeResidualizer runs, even
    though the raw CA_rate column itself is dropped from what the
    classifier ultimately sees. Dropping it any earlier in the pipeline
    would break that computation.
    """
    def __init__(self, drop_feature_names):
        self.drop_feature_names = list(drop_feature_names)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        drop_indices = [FEATURE_NAMES_50.index(name) for name in self.drop_feature_names]
        keep = [i for i in range(np.asarray(X).shape[1]) if i not in drop_indices]
        return np.asarray(X)[:, keep]