# features/__init__.py
# Shared constants for the Narnia_ML feature pipeline.
# Update these whenever a feature module adds or removes features.
#
# Entry 2 (2026-07-01): Dropped absolute Hjorth (site-confounded per LOSO).
# Added within-recording ratio features (site-stable by construction).
# Feature count: 82 → 48.

# ── Per-module counts ─────────────────────────────────────────────────────────
N_DEMOGRAPHIC_FEATURES    = 10   # features/demographic.py
N_CAISR_BASE_FEATURES     = 12   # features/caisr_base.py
N_CAISR_ENRICHED_FEATURES = 11   # features/caisr_enriched.py
N_RATIO_FEATURES          = 15   # features/physiological_ratios.py

# Legacy — kept for backward compatibility with loso_cv.py / phase1_eda.py
N_PHYSIOLOGICAL_FEATURES  = 0    # absolute Hjorth REMOVED in entry 2
N_ALGORITHMIC_FEATURES    = N_CAISR_BASE_FEATURES + N_CAISR_ENRICHED_FEATURES  # 23

# ── Total ─────────────────────────────────────────────────────────────────────
N_TOTAL_FEATURES = (
    N_DEMOGRAPHIC_FEATURES +
    N_CAISR_BASE_FEATURES +
    N_CAISR_ENRICHED_FEATURES +
    N_RATIO_FEATURES
)  # 48

# ── Index ranges (entry 2 vector order) ──────────────────────────────────────
# demo [0:10] → caisr_base [10:22] → caisr_enriched [22:33] → ratios [33:48]
IDX_DEMO_START      = 0
IDX_DEMO_END        = N_DEMOGRAPHIC_FEATURES                              # 10
IDX_BASE_START      = IDX_DEMO_END
IDX_BASE_END        = IDX_BASE_START + N_CAISR_BASE_FEATURES             # 22
IDX_ENRICHED_START  = IDX_BASE_END
IDX_ENRICHED_END    = IDX_ENRICHED_START + N_CAISR_ENRICHED_FEATURES     # 33
IDX_RATIO_START     = IDX_ENRICHED_END
IDX_RATIO_END       = IDX_RATIO_START + N_RATIO_FEATURES                 # 48