"""
verify_label_integrity.py
Team Narnia — PhysioNet Challenge 2026

Checks whether a training set's demographics.csv is pre-curated to contain
ONLY clean Positive/Negative patients, per the official 3-way definition
on the Challenge scoring page, or whether it silently mixes "Other"
(ambiguous/excluded) patients into the Cognitive_Impairment=False bucket.

Background: the small training set was verified clean (every negative has
>=6yr Time_to_Last_Visit, no single-code or wrong-timing patients hiding
in the negatives). There is no guarantee the large training set received
the same curation before release — this script checks that assumption
explicitly rather than trusting it.

Official definition (physionetchallenges.org/2026/#scoring):
    Positive: >=2 CI-related ICD codes, first one 1-6 years after the PSG,
              >=7 days between first and last code.
    Negative: zero CI-related ICD codes ever, AND >=6 years of encounters
              (Time_to_Last_Visit) after the PSG.
    Other:    everything else (first code <1yr or >6yr out, exactly one
              code, or zero codes with <6yr follow-up). EXCLUDED from
              official scoring.

Usage:
    python verify_label_integrity.py \
        --demographics /path/to/training_set_large/demographics.csv \
        --icd /path/to/training_set_large/ICD_codes_CI.csv \
        --out large_set_label_audit.csv

Exit behavior:
    Prints a summary to the console either way. If any "Other" patients
    are found hiding in the current True/False label column, writes their
    IDs to --out so you can decide whether to exclude them from training
    (recommended, to match what real scoring will never include as a
    negative) before committing to a full training run on the large set.
"""

import argparse
import csv
from collections import defaultdict
from datetime import datetime

LOWER_BOUND_DAYS = 365.25 * 1   # 1 year
UPPER_BOUND_DAYS = 365.25 * 6   # 6 years
MIN_SPAN_DAYS = 7               # first-to-last code confirmation
MIN_FOLLOWUP_DAYS = 365.25 * 6  # negative follow-up requirement


def parse_date(s):
    return datetime.strptime(s[:10], "%Y-%m-%d")


def load_icd_dates(icd_path):
    """Returns dict: BDSPPatientID -> sorted list of ICD code dates (str)."""
    by_patient = defaultdict(list)
    with open(icd_path) as f:
        for row in csv.DictReader(f):
            by_patient[row["BDSPPatientID"]].append(row["ICDDate"])
    return by_patient


def classify_official(psg_date, icd_dates, time_to_last_visit_days):
    """
    Reconstructs the official Positive/Negative/Other category from raw
    ICD dates and follow-up duration, independent of whatever label the
    local create_labels.py may have already assigned.

    Returns (category, reason) where category is one of
    'Positive', 'Negative', 'Other', and reason explains Other cases.
    """
    n_codes = len(icd_dates)

    if n_codes == 0:
        if time_to_last_visit_days is not None and time_to_last_visit_days >= MIN_FOLLOWUP_DAYS:
            return "Negative", None
        return "Other", "zero_codes_insufficient_followup"

    if n_codes == 1:
        return "Other", "exactly_one_code"

    first = parse_date(min(icd_dates))
    last = parse_date(max(icd_dates))
    span_days = (last - first).days
    gap_days = (first - psg_date).days

    if span_days < MIN_SPAN_DAYS:
        return "Other", "span_under_7days"
    if gap_days < LOWER_BOUND_DAYS:
        return "Other", "diagnosed_too_early"
    if gap_days > UPPER_BOUND_DAYS:
        return "Other", "diagnosed_too_late"

    return "Positive", None


def run(demographics_path, icd_path, out_path):
    icd_by_patient = load_icd_dates(icd_path)

    total = 0
    official_counts = defaultdict(int)
    reason_counts = defaultdict(int)
    mismatches = []  # patients whose current True/False label disagrees with official category

    with open(demographics_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            pid = row["BDSPPatientID"]
            site = row.get("SiteID", "")
            psg_date = parse_date(row["CreationTime"])
            icd_dates = icd_by_patient.get(pid, [])

            ttlv_raw = row.get("Time_to_Last_Visit", "")
            try:
                ttlv_days = float(ttlv_raw) if ttlv_raw not in ("", "nan") else None
            except ValueError:
                ttlv_days = None

            category, reason = classify_official(psg_date, icd_dates, ttlv_days)
            official_counts[category] += 1
            if reason:
                reason_counts[reason] += 1

            current_label = row.get("Cognitive_Impairment", "")
            current_is_true = current_label == "True"

            # Flag any disagreement between what create_labels.py assigned
            # and what the official 3-way rule would assign.
            expected_true = (category == "Positive")
            if current_is_true != expected_true or category == "Other":
                mismatches.append({
                    "BDSPPatientID": pid,
                    "SiteID": site,
                    "current_label": current_label,
                    "official_category": category,
                    "reason": reason or "",
                    "n_icd_codes": len(icd_dates),
                    "time_to_last_visit_days": ttlv_days if ttlv_days is not None else "",
                })

    # ── Report ────────────────────────────────────────────────────────────
    print(f"Total patients checked: {total}")
    print(f"\nOfficial category breakdown:")
    for cat in ("Positive", "Negative", "Other"):
        n = official_counts.get(cat, 0)
        pct = 100 * n / total if total else 0
        print(f"  {cat:<10} {n:>6}  ({pct:.1f}%)")

    if reason_counts:
        print(f"\n'Other' breakdown:")
        for reason, n in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<35} {n}")

    current_true = sum(1 for _ in open(demographics_path)) - 1  # not used directly; see below
    n_other = official_counts.get("Other", 0)

    print(f"\n{'='*60}")
    if n_other == 0:
        print("CLEAN — no 'Other' patients found. This dataset appears to be")
        print("pre-curated the same way the small training set was. Safe to")
        print("train on the existing True/False labels as-is.")
    else:
        pct_of_total = 100 * n_other / total if total else 0
        print(f"NOT CLEAN — {n_other} patients ({pct_of_total:.1f}% of the dataset)")
        print("are 'Other' by the official definition but are currently")
        print("labeled False (Negative) in this demographics.csv.")
        print(f"\nRecommendation: exclude these {n_other} patients from training")
        print("entirely (do not train on them as either class) so your training")
        print("distribution matches what real scoring will actually contain.")
        print(f"\nFull list written to: {out_path}")
    print(f"{'='*60}")

    if mismatches:
        fieldnames = ["BDSPPatientID", "SiteID", "current_label", "official_category",
                      "reason", "n_icd_codes", "time_to_last_visit_days"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mismatches)

    return official_counts, mismatches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demographics", required=True)
    parser.add_argument("--icd", required=True)
    parser.add_argument("--out", default="label_audit_mismatches.csv")
    args = parser.parse_args()

    run(args.demographics, args.icd, args.out)