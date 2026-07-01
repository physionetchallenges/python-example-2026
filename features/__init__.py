# features/__init__.py
# Shared constants for the Narnia_ML feature pipeline.
# Update these whenever a feature module adds or removes features.

N_DEMOGRAPHIC_FEATURES   = 10   # features/demographic.py
N_PHYSIOLOGICAL_FEATURES = 49   # features/physiological.py
N_CAISR_BASE_FEATURES    = 12   # features/caisr_base.py
N_CAISR_ENRICHED_FEATURES = 11  # features/caisr_enriched.py
N_ALGORITHMIC_FEATURES   = N_CAISR_BASE_FEATURES + N_CAISR_ENRICHED_FEATURES  # 23

N_TOTAL_FEATURES = (
    N_DEMOGRAPHIC_FEATURES +
    N_PHYSIOLOGICAL_FEATURES +
    N_ALGORITHMIC_FEATURES
)  # 82

# Index ranges for each group — useful for slicing and feature importance analysis
IDX_DEMO_START   = 0
IDX_DEMO_END     = N_DEMOGRAPHIC_FEATURES                         # 0:10
IDX_PHYS_START   = IDX_DEMO_END
IDX_PHYS_END     = IDX_PHYS_START + N_PHYSIOLOGICAL_FEATURES      # 10:59
IDX_ALGO_START   = IDX_PHYS_END
IDX_ALGO_END     = IDX_ALGO_START + N_ALGORITHMIC_FEATURES        # 59:82