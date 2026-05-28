#!/usr/bin/env bash
set -euo pipefail

# ============================================
# Create smoke training dataset
# ============================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# IMPORTANT:
# Each team member can modify this path to
# match their local dataset location.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
FULL_DATA_PATH="${FULL_DATA_PATH:-data/training_set}"  # Override with env var if needed

SMOKE_PATH="data/training_smoke"
N_RECORDS="${N_RECORDS:-20}"

echo "Creating smoke dataset..."
echo "Source: ${FULL_DATA_PATH}"
echo "Destination: ${SMOKE_PATH}"

rm -rf "${SMOKE_PATH}"
mkdir -p "${SMOKE_PATH}"

selected_records_file="$(mktemp)"
trap 'rm -f "${selected_records_file}"' EXIT

# Select first N EDF files
while IFS= read -r file_path; do
    rel_path="${file_path#${FULL_DATA_PATH}/}"
    target_path="${SMOKE_PATH}/${rel_path}"
    mkdir -p "$(dirname "${target_path}")"
    cp "${file_path}" "${target_path}"
    stem="$(basename "${file_path}" .edf)"
    patient_part="${stem%_ses-*}"
    session_part="${stem##*_ses-}"
    site_id="$(basename "$(dirname "${file_path}")")"
    printf '%s,%s,%s\n' "${site_id}" "${patient_part}" "${session_part}" >> "${selected_records_file}"
done < <(
    find "${FULL_DATA_PATH}/physiological_data" -type f -name "*.edf" | sort | head -n "${N_RECORDS}"
)

# Copy only annotation EDFs for the selected smoke records.
while IFS=',' read -r site_id patient_part session_part; do
    algo_source="${FULL_DATA_PATH}/algorithmic_annotations/${site_id}/${patient_part}_ses-${session_part}_caisr_annotations.edf"
    algo_target="${SMOKE_PATH}/algorithmic_annotations/${site_id}/${patient_part}_ses-${session_part}_caisr_annotations.edf"
    if [[ -f "${algo_source}" ]]; then
        mkdir -p "$(dirname "${algo_target}")"
        cp "${algo_source}" "${algo_target}"
    fi

    human_source="${FULL_DATA_PATH}/human_annotations/${site_id}/${patient_part}_ses-${session_part}_expert_annotations.edf"
    human_target="${SMOKE_PATH}/human_annotations/${site_id}/${patient_part}_ses-${session_part}_expert_annotations.edf"
    if [[ -f "${human_source}" ]]; then
        mkdir -p "$(dirname "${human_target}")"
        cp "${human_source}" "${human_target}"
    fi
done < "${selected_records_file}"

# Filter demographics to the copied smoke records.
python - <<'PY'
import csv
from pathlib import Path

full_data = Path("data/training_set")
smoke_path = Path("data/training_smoke")

source_csv = full_data / "demographics.csv"
target_csv = smoke_path / "demographics.csv"
phys_root = smoke_path / "physiological_data"

selected_records = set()
for edf_path in phys_root.rglob("*.edf"):
    site_id = edf_path.parent.name
    stem = edf_path.stem
    patient_part, session_part = stem.rsplit("_ses-", 1)
    selected_records.add((site_id, patient_part, session_part))

with source_csv.open("r", newline="", encoding="utf-8") as source_file:
    reader = csv.DictReader(source_file)
    rows = [
        row for row in reader
        if (row["SiteID"], row["BidsFolder"], str(row["SessionID"])) in selected_records
    ]
    fieldnames = reader.fieldnames

with target_csv.open("w", newline="", encoding="utf-8") as target_file:
    writer = csv.DictWriter(target_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY

echo "Smoke dataset created successfully."
