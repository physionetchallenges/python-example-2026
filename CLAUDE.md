# Narnia_ML — PhysioNet Challenge 2026

## Project
Team Narnia submission for the George B. Moody PhysioNet Challenge 2026.
Task: predict future cognitive impairment (CI) diagnosis from a single overnight PSG recording.
Submission deadline: Late August 2026. Wild card entry deadline: 31 July 2026.

## Local Environment (venv) — Fast Iteration Loop

The Dockerfile pins `python:3.10.1-buster`. Match this locally so the gap between
"works in venv" and "works in Docker" stays small.

```bash
# One-time setup
cd Narnia_ML
python3.10 -m venv .venv
source .venv/bin/activate          # macOS/Linux
pip install -r requirements.txt

# Every new session
source .venv/bin/activate
```

**Discipline rules:**
- Never `pip install` anything without immediately adding it to `requirements.txt`.
  This is the most common cause of "works locally, fails in Docker" — a package
  installed ad hoc in the venv that was never declared for the container build.
- Add `.venv/` to `.gitignore`. It must never be copied into the Docker image
  (the Dockerfile's `COPY ./ /challenge` would drag it in otherwise).
- Use loose version specs in `requirements.txt` while iterating early. Pin exact
  versions (e.g. `xgboost==2.0.3`) about a week before any real submission, so a
  future `pip install` doesn't silently pull a newer package with different behavior.
- The venv is for fast debugging only. It is NOT the source of truth for what
  gets submitted — Docker is. See "Docker Verification" below for the required
  pre-submission check.

## Run Commands

```bash
# Train model on a data folder
python train_model.py -d /path/to/training_data -m /path/to/model_folder -v

# Run inference
python run_model.py -d /path/to/data -m /path/to/model_folder -o /path/to/outputs -v

# Evaluate (score against demographics.csv labels)
python evaluate_model.py -d /path/to/data -o /path/to/outputs

# Build Docker container (required before submission)
docker build -t physionet2026 .

# Train inside Docker
docker run -v /path/to/data:/challenge/data \
           -v /path/to/model:/challenge/model \
           physionet2026 python train_model.py -d /challenge/data -m /challenge/model -v

# Run inference inside Docker
docker run -v /path/to/data:/challenge/data \
           -v /path/to/model:/challenge/model \
           -v /path/to/outputs:/challenge/outputs \
           physionet2026 python run_model.py -d /challenge/data -m /challenge/model -o /challenge/outputs -v
```

## Critical Constraints — Read Before Editing Anything

**Only edit `team_code.py`.** The organizers run their own unedited copies of:
- `train_model.py` — do not edit
- `run_model.py` — do not edit
- `helper_code.py` — do not edit
- `evaluate_model.py` — do not edit

Editing any of these will cause the submission to fail silently on the hidden validation set.

To add Python packages: edit `requirements.txt` only.
To add system packages: edit the `Dockerfile` only (apt install line).

## File Structure

```
Narnia_ML/
├── team_code.py          ← ONLY FILE TO EDIT (model + feature extraction)
├── helper_code.py        ← data loading API — read, never edit
├── train_model.py        ← entry point — never edit
├── run_model.py          ← entry point — never edit
├── evaluate_model.py     ← scoring — never edit
├── channel_table.csv     ← channel alias dictionary — never edit
├── requirements.txt      ← add packages here
├── Dockerfile            ← add apt installs here
└── create_labels.py      ← utility for understanding labels, not used at runtime
```

## Data Directory Structure

```
training_set_small/                          # 214 GiB
├── demographics.csv                         # 1,103 rows — primary iteration file
├── ICD_codes_CI.csv                         # CI diagnosis codes
├── physiological_data/
│   ├── S0001/  (857 EDFs)                   # BIDMC
│   ├── I0002/  ( 54 EDFs)                   # Emory — too small for cross-site testing
│   └── I0006/  (192 EDFs)                   # Kaiser Permanente
├── algorithmic_annotations/
│   ├── S0001/ | I0002/ | I0006/
└── human_annotations/                       # training only — NOT in validation/test
    ├── S0001/ | I0002/ | I0006/
```

File naming pattern:
- Physiological:  `{patient_id}_ses-{session_id}.edf`
- CAISR:          `{patient_id}_ses-{session_id}_caisr_annotations.edf`
- Human:          `{patient_id}_ses-{session_id}_expert_annotations.edf`

## Data Loading API (helper_code.py)

```python
# Iterate over patients
patients = find_patients(os.path.join(data_folder, 'demographics.csv'))
# Returns: [{'BidsFolder': str, 'SiteID': str, 'SessionID': int}, ...]

# Load demographics (returns dict of all CSV columns)
data = load_demographics(demographics_file, patient_id, session_id)
# Key fields: Age, Sex, Race, Ethnicity, BMI
# Training only: Time_to_Event, Cognitive_Impairment, Time_to_Last_Visit

# Load any EDF (physiological or annotation)
channel_dict, fs_dict = load_signal_data(edf_path)
# channel_dict: {lowercase_label: np.ndarray float64}
# fs_dict:      {lowercase_label: float (Hz)}
# Mixed sampling rates across channels are normal
```

## CAISR Annotation Channels

When `load_signal_data` is called on a `_caisr_annotations.edf` file:

| Channel key | Resolution | Values |
|---|---|---|
| `stage_caisr` | 30s epochs | 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=unavailable |
| `caisr_prob_n3/n2/n1/r/w` | 30s epochs | 0.0–1.0 softmax probabilities |
| `arousal_caisr` | 0.5s | 0=none, 1=arousal |
| `caisr_prob_arousal` | 0.5s | 0.0–1.0 |
| `resp_caisr` | 1s | 0=none, 1=OA, 2=CA, 3=MA, 4=HY, 5=RERA |
| `limb_caisr` | 1s | 0=none, 1=isolated, 2=periodic |

## Current Feature Vector (71 features)

```
[0]       Age
[1-3]     Sex one-hot (Female, Male, Unknown)
[4-8]     Race one-hot (Asian, Black, Others, Unavailable, White)
[9]       BMI
[10-58]   Physiological: 7 Hjorth-style features × 7 lead types
            Lead order: EEG, EOG, ChinEMG, LegEMG, ECG, Resp, SpO2
            Feature order per lead: std, MAV, ZCR, RMS, variance, mobility, complexity
[59]      AHI (automated events/hour)
[60]      Arousal index (automated)
[61]      Limb movement index (automated)
[62-66]   Stage %: Wake, N1, N2, N3, REM
[67]      Sleep efficiency
[68]      Mean CAISR Wake probability
[69]      Mean CAISR N3 probability
[70]      Mean CAISR arousal probability
```

Human annotation features are extracted but intentionally NOT used in inference
(human annotations are unavailable in the hidden validation and test sets).

## Known Bugs to Fix

1. **Wrong arousal key** in `extract_algorithmic_annotations_features`:
   ```python
   # Wrong (always returns NaN):
   prob_arous = np.mean(algo_data.get('caisr_prob_arous', [float('nan')]))
   # Correct:
   prob_arous = np.mean(algo_data.get('caisr_prob_arousal', [float('nan')]))
   ```

2. **Docstring error** in `extract_demographic_features`: says "length 11" but produces 10.
   Code is correct; fix the comment only.

## Immediate Next Steps (Priority Order)

1. Fix the `caisr_prob_arousal` bug
2. Swap `RandomForestClassifier` → `XGBoost` (add `xgboost` to requirements.txt)
3. Enrich CAISR respiratory features: break AHI into OA/CA/HY/RERA rates + CA/total ratio + REM-AHI vs NREM-AHI
4. Add N3 temporal gradient: N3% first half vs second half of recording
5. Run Phase 1 EDA: BH-FDR corrected univariate tests on all features vs CI label

## Scoring Metrics

- **Age-conditioned AUROC**: primary metric. Probability model ranks a positive patient above a negative patient of approximately the same age (±2 years).
- **Prevalence-based reward**: secondary. Inversely weights by age-local CI prevalence. The baseline example scores -0.001.

Current top leaderboard: AUROC 0.748 (GT NeuroSignals Lab), Reward 0.159 (CLJ Ulm).
Target for wild card consideration: AUROC > 0.60, Reward > 0.015.

## Cross-Site Rules

- **S0001 (BIDMC)**: 857 records — primary training and stability test site
- **I0006 (Kaiser)**: 192 records — secondary stability test site
- **I0002 (Emory)**: 54 records — held-out sanity check only, too small for testing
- Validation set: I0004 (hidden, different institution entirely)
- Test set: I0007 (hidden, evaluated once after challenge)

Never evaluate cross-site stability on I0002. Always leave-one-site-out CV using S0001 and I0006.

## Docker Verification — Required Before Any Submission

Local venv testing is fast but is NOT a guarantee that code works on the
organizers' infrastructure. They build and run your Dockerfile, not your venv.
Before treating any code change as submission-ready, run the full cycle inside
Docker against at minimum the dev subset, ideally the full small set:

```bash
# Rebuild the image whenever requirements.txt or Dockerfile changes
docker build -t physionet2026 .

# Train inside the container
docker run -v /path/to/dev_subset:/challenge/data \
           -v /path/to/dev_model:/challenge/model \
           physionet2026 \
           python train_model.py -d /challenge/data -m /challenge/model -v

# Run inference inside the container
docker run -v /path/to/dev_subset:/challenge/data \
           -v /path/to/dev_model:/challenge/model \
           -v /path/to/dev_outputs:/challenge/outputs \
           physionet2026 \
           python run_model.py -d /challenge/data -m /challenge/model -o /challenge/outputs -v

# Score locally before ever submitting
python evaluate_model.py -d /path/to/dev_subset -o /path/to/dev_outputs
```

**Cadence:** iterate fast in the venv while writing and debugging features.
Before finalizing any of the 10 scored leaderboard entries, run this full
Docker cycle at least once. Catching a missing dependency or path issue here
costs minutes; catching it after submitting costs one of your 10 entries and
up to 72 hours of feedback latency.

## Submission

- Register at: https://forms.gle/hQPUQ8w4hb1MmGHa7 (Team Narnia)
- Submit code at: https://forms.gle/8wivWQqmwTf7nTYo8
- Max 10 scored entries in the official phase
- Feedback takes up to 72 hours per submission
- Submit at least 5 days before any deadline
