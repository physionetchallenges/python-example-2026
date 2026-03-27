# ============================================
# Create smoke training dataset
# ============================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# IMPORTANT:
# Each team member must modify this path to
# match their local dataset location.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
$FULL_DATA_PATH = "data/training_set"  # <-- CHANGE THIS IF NEEDED

$SMOKE_PATH = "data/training_smoke"
$N_RECORDS = 5

Write-Host "Creating smoke dataset..."
Write-Host "Source: $FULL_DATA_PATH"
Write-Host "Destination: $SMOKE_PATH"

Remove-Item -Recurse -Force $SMOKE_PATH -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $SMOKE_PATH | Out-Null

$selectedRecords = New-Object System.Collections.Generic.List[object]

# Select first N EDF files
$edfs = Get-ChildItem "$FULL_DATA_PATH/physiological_data" -Recurse -Filter *.edf |
        Sort-Object FullName |
        Select-Object -First $N_RECORDS

foreach ($f in $edfs) {
    $rel = $f.FullName.Substring((Resolve-Path $FULL_DATA_PATH).Path.Length).TrimStart('\')
    $target = Join-Path $SMOKE_PATH $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    Copy-Item $f.FullName $target

    $stem = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
    $parts = $stem -split '_ses-'
    $selectedRecords.Add([pscustomobject]@{
        SiteID = $f.Directory.Name
        Patient = $parts[0]
        Session = $parts[1]
    }) | Out-Null
}

# Copy only annotation EDFs for the selected smoke records.
foreach ($record in $selectedRecords) {
    $algoSource = Join-Path $FULL_DATA_PATH "algorithmic_annotations/$($record.SiteID)/$($record.Patient)_ses-$($record.Session)_caisr_annotations.edf"
    $algoTarget = Join-Path $SMOKE_PATH "algorithmic_annotations/$($record.SiteID)/$($record.Patient)_ses-$($record.Session)_caisr_annotations.edf"
    if (Test-Path $algoSource) {
        New-Item -ItemType Directory -Force -Path (Split-Path $algoTarget) | Out-Null
        Copy-Item $algoSource $algoTarget
    }

    $humanSource = Join-Path $FULL_DATA_PATH "human_annotations/$($record.SiteID)/$($record.Patient)_ses-$($record.Session)_expert_annotations.edf"
    $humanTarget = Join-Path $SMOKE_PATH "human_annotations/$($record.SiteID)/$($record.Patient)_ses-$($record.Session)_expert_annotations.edf"
    if (Test-Path $humanSource) {
        New-Item -ItemType Directory -Force -Path (Split-Path $humanTarget) | Out-Null
        Copy-Item $humanSource $humanTarget
    }
}

# Filter demographics to the copied smoke records.
$env:SMOKE_FULL_DATA_PATH = (Resolve-Path $FULL_DATA_PATH).Path
$env:SMOKE_PATH = (Resolve-Path $SMOKE_PATH).Path
python -c @"
import csv
import os
from pathlib import Path

full_data = Path(os.environ['SMOKE_FULL_DATA_PATH'])
smoke_path = Path(os.environ['SMOKE_PATH'])

source_csv = full_data / 'demographics.csv'
target_csv = smoke_path / 'demographics.csv'
phys_root = smoke_path / 'physiological_data'

selected_records = set()
for edf_path in phys_root.rglob('*.edf'):
    site_id = edf_path.parent.name
    patient_part, session_part = edf_path.stem.rsplit('_ses-', 1)
    selected_records.add((site_id, patient_part, session_part))

with source_csv.open('r', newline='', encoding='utf-8') as source_file:
    reader = csv.DictReader(source_file)
    rows = [
        row for row in reader
        if (row['SiteID'], row['BidsFolder'], str(row['SessionID'])) in selected_records
    ]
    fieldnames = reader.fieldnames

with target_csv.open('w', newline='', encoding='utf-8') as target_file:
    writer = csv.DictWriter(target_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
"@
Remove-Item Env:SMOKE_FULL_DATA_PATH -ErrorAction SilentlyContinue
Remove-Item Env:SMOKE_PATH -ErrorAction SilentlyContinue

Write-Host "Smoke dataset created successfully."