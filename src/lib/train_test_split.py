import csv
import os
import shutil


def create_test_set_from_ids(test_ids_file, demographics_file, source_dir, dest_dir):
    """
    Create a test set by moving EDF files based on patient IDs.

    Args:
        test_ids_file (str): Path to the file containing test patient IDs (one per line).
        demographics_file (str): Path to the demographics CSV file.
        source_dir (str): Path to the source directory containing physiological data folders.
        dest_dir (str): Path to the destination directory for the test set.
    """
    # Read test IDs
    with open(test_ids_file, 'r') as f:
        test_ids = set(line.strip() for line in f if line.strip())

    print(f"Test IDs: {len(test_ids)}")

    # Parse demographics to get mapping
    id_to_bids = {}
    with open(demographics_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row['BDSPPatientID']
            bids_folder = row['BidsFolder']
            if patient_id in test_ids:
                if patient_id not in id_to_bids:
                    id_to_bids[patient_id] = []
                id_to_bids[patient_id].append(bids_folder)

    print(f"Found mappings for {len(id_to_bids)} patients")

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    moved_files = []
    # Find and move files
    for patient_id, bids_list in id_to_bids.items():
        for bids in bids_list:
            # bids is like sub-I0002150000686
            # file is sub-I0002150000686_ses-X.edf
            # folder is I0002, I0006, S0001
            site = bids.split('-')[1][:5]  # I0002 or I0006 or S0001
            folder = os.path.join(source_dir, site)
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.startswith(bids + '_') and file.lower().endswith('.edf'):
                        src = os.path.join(folder, file)
                        dst = os.path.join(dest_dir, file)
                        shutil.move(src, dst)
                        moved_files.append(dst)
                        print(f"Moved {src} to {dst}")

    print(f"Done. Moved {len(moved_files)} EDF files")
    return moved_files


def get_selected_records_from_test_set(test_set_dir):
    selected_records = set()
    for root, _, files in os.walk(test_set_dir):
        for file in files:
            if file.lower().endswith('.edf'):
                stem = os.path.splitext(file)[0]
                if '_ses-' not in stem:
                    continue
                bids_folder, session_id = stem.rsplit('_ses-', 1)
                site_id = bids_folder.split('-')[1][:5] if '-' in bids_folder else ''
                selected_records.add((site_id, bids_folder, session_id))
    return selected_records


def sync_demographics_between_train_and_test(source_demographics_file, test_set_dir, test_demographics_file=None):
    """
    Create a filtered demographics CSV in the test set and remove those rows from the training demographics file.

    Args:
        source_demographics_file (str): Path to the source training_set demographics CSV.
        test_set_dir (str): Directory containing the moved EDF files for the test set.
        test_demographics_file (str, optional): Output path for the test set demographics CSV.
            Defaults to '<test_set_dir>/demographics.csv'.
    """
    if test_demographics_file is None:
        test_demographics_file = os.path.join(test_set_dir, 'demographics.csv')

    selected_records = get_selected_records_from_test_set(test_set_dir)
    print(f"Found {len(selected_records)} EDF records in {test_set_dir}")

    with open(source_demographics_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        source_rows = list(reader)

    rows_for_test = []
    remaining_rows = []
    for row in source_rows:
        key = (row.get('SiteID'), row.get('BidsFolder'), str(row.get('SessionID')))
        if key in selected_records:
            rows_for_test.append(row)
        else:
            remaining_rows.append(row)

    os.makedirs(os.path.dirname(test_demographics_file), exist_ok=True)
    with open(test_demographics_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_for_test)

    with open(source_demographics_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(remaining_rows)

    print(f"Wrote {len(rows_for_test)} rows to {test_demographics_file}")
    print(f"Removed {len(source_rows) - len(remaining_rows)} rows from {source_demographics_file}")


if __name__ == "__main__":
    # Example usage
    moved_files = create_test_set_from_ids(
        test_ids_file='testids_lista.txt',
        demographics_file='data/training_set/demographics.csv',
        source_dir='data/training_set/physiological_data',
        dest_dir='data/test_set'
    )
    if moved_files:
        sync_demographics_between_train_and_test(
            source_demographics_file='data/training_set/demographics.csv',
            test_set_dir='data/test_set'
        )