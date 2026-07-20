# features/subtype_weighting.py
# Team Narnia — PhysioNet Challenge 2026
#
# Entry 7: MCI-targeted sample weighting, layered on top of the Entry 6
# age-based value weighting (features/sample_weighting.py). Promoted from
# eda/build_positive_patient_report.py's CODE_FAMILY/classify_family/
# dominant_family (originally interpretation-only, never touched by
# team_code.py) into a shared features/ module, because this is now a
# TRAINING-TIME input to the real submission path — same "single source
# of truth, not duplicated" precedent as AgeResidualizer and
# compute_value_weighted_sample_weights.
#
# CODE_FAMILY here includes 3 codes added 2026-07-19, after auditing every
# unique code in the real ICD_codes_CI.csv against the official
# ICD_codes.csv reference table (moody-challenge.physionet.org/2026/
# ICD_codes.csv): G31.85 (Corticobasal degeneration), 331.11 (Pick's
# disease, ICD-9), F10.27 (Alcohol-induced persisting dementia) — all
# three were present in the real data but missing from the original
# eda-only dict. Verified zero practical impact on any 2026-07-19 finding
# (0 of 498 patients had their dominant_family() classification changed
# by this gap — every affected row co-occurred with an already-correctly-
# classified code for the same patient), but added here for completeness
# given this is now a training-time dependency, not just interpretation.
#
# Note: the official reference table only recognizes 3 top-level
# categories (MCI, Alzheimer's disease, Dementia) — the finer split used
# here (Lewy body / vascular / frontotemporal / unspecified / etc., all
# "Dementia" officially) is this project's own clinically-motivated
# breakdown, not the challenge's. Only the "Mild cognitive impairment"
# category is actually load-bearing for compute_mci_boosted_weights below
# — the finer dementia sub-buckets exist for interpretation/EDA
# elsewhere (site_subtype_breakdown.py), not for anything in this module.
#
# Validated candidate (2026-07-19 ablation, tools/ablation_mci_sample_
# weight.py): beta_mci=0.75 on top of alpha_age=1.0 — reward +5.6%
# relative vs. alpha_age=1.0 alone, S0001 AUROC flat (+0.01 sigma), MCI
# sensitivity 76.8% -> 85.9%. A fine sweep (0.60-0.90) confirmed 0.75 as a
# genuine local optimum — a smooth, broad plateau, not a single-point
# fluke. See learning_log.md, 2026-07-19, for full methodology.
#
# LEAKAGE DISCIPLINE: ICD/subtype data is used ONLY to shape a training-
# time sample weight, exactly like the age-based weighting it extends —
# never a model input, never touched by run_model()/inference. Official
# ICD_codes_CI.csv is guaranteed present in both training_set_small and
# training_set_large (confirmed against the official challenge data-
# access documentation, moody-challenge.physionet.org/2026, 2026-07-19),
# at the same directory level as demographics.csv.

import os
from collections import defaultdict

import numpy as np
import pandas as pd

from features.sample_weighting import compute_value_weighted_sample_weights

ICD_CODES_FILENAME = 'ICD_codes_CI.csv'

CODE_FAMILY = {
    # ICD-10
    "G30": "Alzheimer disease",
    "G31.84": "Mild cognitive impairment",
    "G31.83": "Dementia with Lewy bodies",
    "G31.0": "Frontotemporal dementia",
    "G31.85": "Dementia in other disease",   # added 2026-07-19 (Corticobasal degeneration)
    "F01": "Vascular dementia",
    "F02": "Dementia in other disease",
    "F03": "Unspecified dementia",
    "F10.27": "Dementia in other disease",   # added 2026-07-19 (alcohol-induced persisting dementia)
    # ICD-9
    "331.0": "Alzheimer disease",
    "331.83": "Mild cognitive impairment",
    "331.82": "Dementia with Lewy bodies",
    "331.19": "Frontotemporal dementia",
    "331.11": "Dementia in other disease",   # added 2026-07-19 (Pick's disease)
    "290": "Senile/presenile dementia",
    "294": "Dementia in conditions classified elsewhere",
}

_MCI_LABEL = "Mild cognitive impairment"

_PRIORITY = [
    "Dementia with Lewy bodies", "Frontotemporal dementia", "Alzheimer disease",
    "Vascular dementia", _MCI_LABEL, "Dementia in other disease",
    "Unspecified dementia", "Senile/presenile dementia",
    "Dementia in conditions classified elsewhere",
]


def _clean_icd_value(val):
    """CSV cells missing a value come back as float NaN even under
    dtype=str (pandas' missing-value representation doesn't respect the
    dtype hint for empty cells). NaN is truthy in Python, so a naive
    `row.get('ICD10','') or row.get('ICD9','')` silently returns NaN
    instead of '' whenever ICD10 is missing — classify_family() then
    calls .startswith() on a float and crashes. This is the normal case,
    not an edge case: most patients have only one of ICD10/ICD9 filled.
    Same fix as eda/site_subtype_breakdown.py's identical bug, caught
    2026-07-19 testing this module against the real ICD_codes_CI.csv."""
    if pd.isna(val):
        return ''
    return str(val)


def classify_family(icd10, icd9):
    code = icd10 or icd9
    if not code:
        return "unknown"
    for prefix, family in CODE_FAMILY.items():
        if code.startswith(prefix):
            return family
    return f"other ({code})"


def dominant_family(families):
    for p in _PRIORITY:
        if p in families:
            return p
    return sorted(families)[0] if families else "unknown"


def load_icd_subtype_lookup(data_folder, verbose=False):
    """
    Loads ICD_codes_CI.csv from the training data folder and returns a
    dict: BDSPPatientID (str) -> dominant subtype (str).

    Graceful fallback, not a crash: returns an empty dict if the file is
    missing. compute_mci_boosted_weights() treats any patient absent from
    this lookup as "not MCI" (weight multiplier 1.0) — same behavior as
    if the file were present but that patient simply had no ICD entries.
    This means a missing file silently degrades to alpha-only weighting
    (Entry 6 behavior) rather than crashing train_model() — deliberate,
    since the ICD file is training-time-only interpretive data, not core
    to the pipeline the way demographics.csv is.
    """
    icd_path = os.path.join(data_folder, ICD_CODES_FILENAME)
    if not os.path.exists(icd_path):
        if verbose:
            print(f'  ! {ICD_CODES_FILENAME} not found in {data_folder} — '
                  f'MCI-boosted weighting will fall back to alpha-only (Entry 6 behavior).')
        return {}
    df = pd.read_csv(icd_path, dtype=str)
    by_patient = defaultdict(set)
    for _, row in df.iterrows():
        pid = row['BDSPPatientID']
        fam = classify_family(_clean_icd_value(row.get('ICD10')), _clean_icd_value(row.get('ICD9')))
        by_patient[pid].add(fam)
    return {pid: dominant_family(fams) for pid, fams in by_patient.items()}


def compute_mci_boosted_weights(y_train, age_train, patient_id_train, subtype_lookup,
                                 alpha_age=1.0, beta_mci=0.75):
    """
    Combines the Entry 6 age-based value weighting with an MCI-specific
    boost. beta_mci=0.0 reproduces compute_value_weighted_sample_weights
    exactly, unchanged — this is a pure additive extension, not a
    replacement.

    Only positive training samples are ever boosted; negatives always get
    weight 1.0, same convention as the age-based weighting it extends.
    patient_id_train entries absent from subtype_lookup (including the
    case where subtype_lookup is empty, e.g. ICD_codes_CI.csv missing)
    are treated as "not MCI" — no boost, silent fallback to alpha-only
    behavior for those patients.
    """
    w = compute_value_weighted_sample_weights(y_train, age_train, alpha=alpha_age)
    if beta_mci == 0.0:
        return w
    w = w.copy()
    for i in range(len(y_train)):
        if y_train[i] == 1:
            pid = str(patient_id_train[i])
            if subtype_lookup.get(pid) == _MCI_LABEL:
                w[i] *= (1.0 + beta_mci)
    return w