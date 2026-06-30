#!/usr/bin/env python3
"""
build_dev_subset.py — Team Narnia

Builds a small, stratified development subset from the full Moody's
small training set so you can iterate on team_code.py without reading
all 1,103 patients on every test run.

Usage:
    python build_dev_subset.py --source /path/to/training_set_small \
                                 --dest /path/to/dev_subset \
                                 --n-per-site 10

Stratification:
    - Pulls patients from all three sites (S0001, I0002, I0006)
    - Within each site, balances positive (CI=TRUE) and negative (CI=FALSE)
      labels as close to 50/50 as the site's data allows
    - Copies demographics.csv (filtered to subset), physiological EDFs,
      CAISR annotation EDFs, and human annotation EDFs for the sampled
      patients only

Run this once after the full small dataset is downloaded. Re-run with
a different --seed to get a different (but still stratified) sample.
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Build a stratified dev subset of the Moody's training data.")
    p.add_argument("--source", required=True, help="Path to the full training_set_small directory")
    p.add_argument("--dest", required=True, help="Path to write the dev subset directory")
    p.add_argument("--n-per-site", type=int, default=10,
                   help="Number of patients to sample per site (default: 10)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    p.add_argument("--include-human-annotations", action="store_true", default=True,
                   help="Copy human expert annotations too (training only, default: True)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be copied without copying any files")
    return p.parse_args()


def load_demographics(source: Path) -> pd.DataFrame:
    demo_path = source / "demographics.csv"
    if not demo_path.exists():
        sys.exit(f"ERROR: demographics.csv not found at {demo_path}")
    df = pd.read_csv(demo_path)
    required_cols = {"SiteID", "BidsFolder", "SessionID", "Cognitive_Impairment"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"ERROR: demographics.csv missing expected columns: {missing}")
    return df


def stratified_sample(df: pd.DataFrame, n_per_site: int, seed: int) -> pd.DataFrame:
    """
    For each site, sample up to n_per_site patients, balancing
    Cognitive_Impairment TRUE/FALSE as evenly as the site allows.
    Patients with missing/ambiguous labels (e.g. 'other') are excluded
    from the dev subset by default — this keeps the subset clean for
    quick binary-classification testing.
    """
    samples = []
    for site_id, site_df in df.groupby("SiteID"):
        site_df = site_df[site_df["Cognitive_Impairment"].isin(["TRUE", "FALSE", True, False])]
        pos = site_df[site_df["Cognitive_Impairment"].isin(["TRUE", True])]
        neg = site_df[site_df["Cognitive_Impairment"].isin(["FALSE", False])]

        n_pos_target = n_per_site // 2
        n_neg_target = n_per_site - n_pos_target

        n_pos_actual = min(n_pos_target, len(pos))
        n_neg_actual = min(n_neg_target, len(neg))

        # If one class is short, top up from the other class so the
        # site still contributes n_per_site patients where possible
        shortfall = n_per_site - (n_pos_actual + n_neg_actual)
        if shortfall > 0:
            if len(pos) - n_pos_actual > 0:
                extra = min(shortfall, len(pos) - n_pos_actual)
                n_pos_actual += extra
                shortfall -= extra
            if shortfall > 0 and len(neg) - n_neg_actual > 0:
                extra = min(shortfall, len(neg) - n_neg_actual)
                n_neg_actual += extra

        pos_sample = pos.sample(n=n_pos_actual, random_state=seed) if n_pos_actual > 0 else pos.iloc[0:0]
        neg_sample = neg.sample(n=n_neg_actual, random_state=seed) if n_neg_actual > 0 else neg.iloc[0:0]

        site_sample = pd.concat([pos_sample, neg_sample])
        samples.append(site_sample)

        print(f"  {site_id}: sampled {len(site_sample)} "
              f"({n_pos_actual} positive, {n_neg_actual} negative) "
              f"from {len(site_df)} eligible patients")

    return pd.concat(samples).reset_index(drop=True)


def file_path_for(patient_id: str, session_id: int, kind: str) -> str:
    """
    Mirrors the naming convention used by the challenge data.
    kind: 'physio' | 'caisr' | 'human'
    """
    if kind == "physio":
        return f"{patient_id}_ses-{session_id}.edf"
    if kind == "caisr":
        return f"{patient_id}_ses-{session_id}_caisr_annotations.edf"
    if kind == "human":
        return f"{patient_id}_ses-{session_id}_expert_annotations.edf"
    raise ValueError(f"Unknown kind: {kind}")


def copy_patient_files(row, source: Path, dest: Path, include_human: bool, dry_run: bool):
    site_id = row["SiteID"]
    patient_id = row["BidsFolder"]
    session_id = row["SessionID"]

    asset_dirs = [
        ("physiological_data", "physio"),
        ("algorithmic_annotations", "caisr"),
    ]
    if include_human:
        asset_dirs.append(("human_annotations", "human"))

    copied = []
    missing = []

    for subdir, kind in asset_dirs:
        fname = file_path_for(patient_id, session_id, kind)
        src_file = source / subdir / site_id / fname
        dst_file = dest / subdir / site_id / fname

        if not src_file.exists():
            missing.append(str(src_file))
            continue

        if dry_run:
            copied.append(str(dst_file))
            continue

        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied.append(str(dst_file))

    return copied, missing


def main():
    args = parse_args()
    source = Path(args.source)
    dest = Path(args.dest)

    if not source.exists():
        sys.exit(f"ERROR: source path does not exist: {source}")

    print(f"Loading demographics from {source}/demographics.csv ...")
    df = load_demographics(source)
    print(f"Total patients in source: {len(df)}")
    print(f"Sites: {sorted(df['SiteID'].unique())}")
    print()

    print(f"Stratified sampling (n_per_site={args.n_per_site}, seed={args.seed}):")
    sampled = stratified_sample(df, args.n_per_site, args.seed)
    print()
    print(f"Total sampled: {len(sampled)} patients")
    print()

    if args.dry_run:
        print("DRY RUN — no files will be copied.\n")

    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    total_missing = []
    total_copied = 0

    print("Copying patient files...")
    for _, row in sampled.iterrows():
        copied, missing = copy_patient_files(row, source, dest, args.include_human_annotations, args.dry_run)
        total_copied += len(copied)
        total_missing.extend(missing)

    # Write the filtered demographics.csv for the subset
    subset_demo_path = dest / "demographics.csv"
    if not args.dry_run:
        sampled.to_csv(subset_demo_path, index=False)
        print(f"\nWrote filtered demographics.csv: {subset_demo_path}")
    else:
        print(f"\n[dry run] would write: {subset_demo_path}")

    # Copy ICD codes file if present (used by create_labels.py / label inspection)
    icd_src = source / "ICD_codes_CI.csv"
    if icd_src.exists() and not args.dry_run:
        shutil.copy2(icd_src, dest / "ICD_codes_CI.csv")
        print(f"Copied ICD_codes_CI.csv")

    print()
    print("=" * 60)
    print(f"DEV SUBSET SUMMARY")
    print("=" * 60)
    print(f"Patients sampled:     {len(sampled)}")
    print(f"Files copied:         {total_copied}")
    print(f"Files missing:        {len(total_missing)}")
    if total_missing:
        print("\nMissing files (first 10):")
        for m in total_missing[:10]:
            print(f"  {m}")
        if len(total_missing) > 10:
            print(f"  ... and {len(total_missing) - 10} more")
    print()
    print(f"Dev subset written to: {dest}")
    print()
    print("Next step — test the pipeline against this subset:")
    print(f"  python train_model.py -d {dest} -m ./dev_model -v")


if __name__ == "__main__":
    main()
