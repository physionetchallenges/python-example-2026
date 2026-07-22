# features/__init__.py
# Shared constants for the Narnia_ML feature pipeline.
# Update these whenever a feature module adds or removes features.
#
# Entry 2 (2026-07-01): Dropped absolute Hjorth (site-confounded per LOSO).
# Added within-recording ratio features (site-stable by construction).
# Feature count: 82 → 48.
#
# Entry 4 (2026-07-02): Added 2 age-residualized features (CA_rate,
# EEG_var_REM_Wake) via features/age_residuals.py — see that module's
# header for the age-residualized EDA rationale. These are appended by an
# AgeResidualizer pipeline step AFTER extraction/hstack, not by a feature
# module called from team_code.py's extraction block, so they live at the
# end of the vector rather than inside the demo/base/enriched/ratio blocks.
# Feature count: 48 → 50.
#
# NEW candidate, not yet validated (2026-07-20): Added 3 stage-conditional
# limb movement features (Limb_REM, Limb_NREM, Limb_REM_NREM_ratio) via
# features/caisr_enriched.py — same enrichment pattern already proven for
# AHI (REM_AHI/NREM_AHI/REM_NREM_AHI_ratio), applied to Limb_idx instead.
# Appended at the END of the enriched block (after N3_entropy, before the
# ratio block starts) — inserted, not appended to the whole vector, so
# downstream index constants that are defined relative to IDX_RATIO_START
# (e.g. IDX_EEG_VAR_REM_WAKE below) auto-update correctly; nothing already
# shipped needs a manual index bump. Feature count: 50 → 53 (raw 48 → 51,
# plus the same 2 age-residual features as before). Requires regenerating
# any --features-cache built before this change — the raw hstack length
# changed from 48 to 51.


N_DEMOGRAPHIC_FEATURES     = 10   # features/demographic.py
N_CAISR_BASE_FEATURES      = 12   # features/caisr_base.py
N_CAISR_ENRICHED_FEATURES  = 14   # features/caisr_enriched.py (was 11 — +3 limb features, 2026-07-20)
N_RATIO_FEATURES           = 15   # features/physiological_ratios.py
N_AGE_RESIDUAL_FEATURES    = 2    # features/age_residuals.py (Entry 4, pipeline-appended)

N_PHYSIOLOGICAL_FEATURES  = 0
N_ALGORITHMIC_FEATURES    = N_CAISR_BASE_FEATURES + N_CAISR_ENRICHED_FEATURES  # 23

N_TOTAL_FEATURES = (
    N_DEMOGRAPHIC_FEATURES +
    N_CAISR_BASE_FEATURES +
    N_CAISR_ENRICHED_FEATURES +
    N_RATIO_FEATURES +
    N_AGE_RESIDUAL_FEATURES
)  # 50

IDX_DEMO_START      = 0
IDX_DEMO_END        = N_DEMOGRAPHIC_FEATURES                              # 10
IDX_BASE_START      = IDX_DEMO_END
IDX_BASE_END        = IDX_BASE_START + N_CAISR_BASE_FEATURES             # 22
IDX_ENRICHED_START  = IDX_BASE_END
IDX_ENRICHED_END    = IDX_ENRICHED_START + N_CAISR_ENRICHED_FEATURES     # 33
IDX_RATIO_START     = IDX_ENRICHED_END
IDX_RATIO_END       = IDX_RATIO_START + N_RATIO_FEATURES                 # 48
IDX_AGE_RESID_START = IDX_RATIO_END
IDX_AGE_RESID_END   = IDX_AGE_RESID_START + N_AGE_RESIDUAL_FEATURES      # 50

IDX_AGE              = IDX_DEMO_START               # Age is demo feature 0
IDX_CA_RATE          = IDX_ENRICHED_START + 1        # CA_rate is caisr_enriched feature 1
IDX_EEG_VAR_REM_WAKE = IDX_RATIO_START + 11          # EEG_var_REM_Wake is ratio feature 11

# Human-readable names for the 48-length PRE-AgeResidualizer vector, index-aligned.
# Pulled directly from FEATURES.md's Entry 2 (48-feature) index table — real
# names, not placeholders. Used by reg_sweep.py to label coefficients / top
# features so sweep output is directly readable against your own registry.
FEATURE_NAMES_48 = [
    'Age', 'Sex_F', 'Sex_M', 'Sex_Unk', 'Race_Asian', 'Race_Black',
    'Race_Other', 'Race_Unavailable', 'Race_White', 'BMI',
    'AHI_total', 'Arousal_idx', 'Limb_idx', 'Wake_pct', 'N1_pct', 'N2_pct',
    'N3_pct', 'REM_pct', 'Sleep_eff', 'Prob_W', 'Prob_N3', 'Prob_arous',
    'OA_rate', 'CA_rate', 'HY_rate', 'RERA_rate', 'CA_total_ratio',
    'REM_AHI', 'NREM_AHI', 'REM_NREM_AHI_ratio', 'N3_gradient',
    'Spont_arousal_idx', 'N3_entropy',
    'Limb_REM', 'Limb_NREM', 'Limb_REM_NREM_ratio',
    'EEG_std_N3_Wake', 'EEG_mav_N3_Wake', 'EEG_zcr_N3_Wake', 'EEG_rms_N3_Wake',
    'EEG_var_N3_Wake', 'EEG_mob_N3_Wake', 'EEG_cplx_N3_Wake',
    'EEG_std_REM_Wake', 'EEG_mav_REM_Wake', 'EEG_zcr_REM_Wake', 'EEG_rms_REM_Wake',
    'EEG_var_REM_Wake', 'EEG_mob_REM_Wake', 'EEG_cplx_REM_Wake',
    'Chin_mav_REM_NREM_atonia',
]
assert len(FEATURE_NAMES_48) == IDX_RATIO_END, \
    f"FEATURE_NAMES_48 length {len(FEATURE_NAMES_48)} != {IDX_RATIO_END}"
assert FEATURE_NAMES_48[IDX_CA_RATE] == 'CA_rate'
assert FEATURE_NAMES_48[IDX_EEG_VAR_REM_WAKE] == 'EEG_var_REM_Wake'

FEATURE_NAMES_50 = FEATURE_NAMES_48 + ['CA_rate_age_residual', 'EEG_var_REM_Wake_age_residual']