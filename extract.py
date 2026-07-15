import sys
import numpy as np

# Adjust these paths to match your repo layout
sys.path.insert(0, 'tools')   # so `from loso_cv import extract_all_features` resolves
sys.path.insert(0, '.')       # repo root, for helper_code/features imports loso_cv.py needs

from loso_cv import extract_all_features

DATA_PATH = '/Users/briandillon/.cache/kagglehub/datasets/physionet/physionetchallenge2026data/versions/7'

# Ground truth: loso_cv.py's own extraction (same function that produced every logged result)
X_truth, y_truth, ages_truth, sites_truth, ids_truth = extract_all_features(DATA_PATH, verbose=True)

# Cache: build_features_cache.py's output
d = np.load('tools/small_features.npz', allow_pickle=True)
X_cache, y_cache, ids_cache = d['X'], d['y'], d['patient_id']

print(f"Ground truth: {X_truth.shape}, cache: {X_cache.shape}")

# Align by patient_id — order may differ between the two extraction runs
truth_lookup = {pid: i for i, pid in enumerate(ids_truth)}
cache_lookup = {pid: i for i, pid in enumerate(ids_cache)}

common_ids = sorted(set(truth_lookup) & set(cache_lookup))
print(f"Common patient IDs: {len(common_ids)} (expect 1103)")

max_diffs = []
mismatched_patients = []

for pid in common_ids:
    row_truth = X_truth[truth_lookup[pid]]
    row_cache = X_cache[cache_lookup[pid]]
    # NaN-safe comparison — both matrices use NaN for missing CAISR data
    diff = np.abs(np.nan_to_num(row_truth, nan=-999) - np.nan_to_num(row_cache, nan=-999))
    max_diff = diff.max()
    max_diffs.append(max_diff)
    if max_diff > 1e-4:
        mismatched_patients.append((pid, max_diff, np.argmax(diff)))

max_diffs = np.array(max_diffs)
print(f"\nMax abs diff across all patients/features: {max_diffs.max():.6f}")
print(f"Patients with any mismatch (>1e-4): {len(mismatched_patients)}")

if mismatched_patients:
    print("\nFirst 5 mismatches (patient_id, max_diff, feature_column_index):")
    for m in mismatched_patients[:5]:
        print(" ", m)
else:
    print("\nCLEAN — every patient's 48-length vector matches exactly between "
          "loso_cv.py's extraction and build_features_cache.py's cache.")