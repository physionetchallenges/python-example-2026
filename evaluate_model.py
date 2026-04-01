#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for evaluating models for the Challenge. You can run it as follows:
#
#   python evaluate_model.py -d labels.csv -o predictions.csv -s scores.csv
#
# where 'labels.csv' is a CSV file containing the labels, 'predictions.csv' is a CSV file containing containing the predictions, and
# 'scores.csv' (optional) is a collection of scores for the predictions.
#
# The Challenge webpage describes the file formats and scoring functions.

import argparse
import numpy as np
import os
import os.path
import pandas as pd
import sys

id_patients = 'BDSPPatientID'
id_labels = 'Cognitive_Impairment'
id_binary_predictions = 'Cognitive_Impairment'
id_probability_predictions = 'Cognitive_Impairment_Probability'

# Parse arguments.
def get_parser():
    description = 'Evaluate the Challenge model.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--labels_folder', type=str, required=True)
    parser.add_argument('-o', '--predictions_folder', type=str, required=True)
    parser.add_argument('-s', '--score_file', type=str, required=False)
    return parser

# Compute AUC.
def compute_auc(labels, predictions):
    from sklearn.metrics import roc_auc_score, average_precision_score
    auroc = roc_auc_score(labels, predictions, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)
    auprc = average_precision_score(labels, predictions, average='macro', pos_label=1, sample_weight=None)
    return auroc, auprc

# Compute accuracy.
def compute_accuracy(labels, predictions):
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels, predictions, normalize=True, sample_weight=None)
    return accuracy

# Compute F-measure.
def compute_f_measure(labels, predictions):
    from sklearn.metrics import f1_score
    f_measure = f1_score(labels, predictions, pos_label=1, average='binary')
    return f_measure

# Evaluate the models.
def evaluate_model(labels_file, predictions_file):
    # Load the labels and predictions.
    df_labels = pd.read_csv(labels_file)
    df_labels.set_index(id_patients, inplace=True)
    df_predictions = pd.read_csv(predictions_file)
    df_predictions.set_index(id_patients, inplace=True)

    def standardize_bool(val):
            s = str(val).strip().upper()
            if s in ['TRUE', '1', '1.0', 'T', 'Y', 'YES']: return 1.0
            if s in ['FALSE', '0', '0.0', 'F', 'N', 'NO']: return 0.0
            return np.nan
    
    # Standardize the labels and predictions to be 0/1.
    df_labels[id_labels] = df_labels[id_labels].apply(standardize_bool)
    df_predictions[id_binary_predictions] = df_predictions[id_binary_predictions].apply(standardize_bool)

    # Only consider patients with positive or negative labels.
    df_labels = df_labels[(df_labels[id_labels] == 0) | (df_labels[id_labels] == 1)]
    patients = df_labels.index
    num_patients = len(patients)

    # Extract the labels and predictions.
    labels = np.zeros(num_patients)
    binary_predictions = np.zeros(num_patients)
    probability_predictions = np.zeros(num_patients)

    for i, patient in enumerate(patients):
        label = df_labels.loc[patient, id_labels]
        labels[i] = label
        if patient in df_predictions.index:   # Set missing predictions to 0.
            binary_prediction = float(np.real(df_predictions.loc[patient, id_binary_predictions]))
            if binary_prediction == 0 or binary_prediction == 1:   # Set invalid binary predictions to 0.
                binary_predictions[i] = binary_prediction
            probability_prediction = float(np.real(df_predictions.loc[patient, id_probability_predictions]))
            if np.isfinite(probability_prediction):   # Set invalid probability predictions to 0.
                probability_predictions[i] = probability_prediction

    # Evaluate the predictions.
    auroc, auprc = compute_auc(labels, probability_predictions)
    accuracy = compute_accuracy(labels, binary_predictions)
    f_measure = compute_f_measure(labels, binary_predictions)

    return auroc, auprc, accuracy, f_measure

# Run the code.
def run(args):
    # Compute the scores for the model predictions.
    auroc, auprc, accuracy, f_measure = evaluate_model(args.labels_folder, args.predictions_folder)

    output_string = \
        f'AUROC: {auroc:.3f}\n' \
        f'AUPRC: {auprc:.3f}\n' + \
        f'Accuracy: {accuracy:.3f}\n' \
        f'F-measure: {f_measure:.3f}\n'

    # Output the scores to screen and/or a file.
    if args.score_file:
        with open(args.score_file, 'w') as f:
            f.write(output_string)
    else:
        print(output_string)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
