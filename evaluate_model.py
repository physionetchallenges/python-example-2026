#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for evaluating models for the Challenge. You can run it as follows:
#
#   python evaluate_model.py -d labels.csv -o predictions.csv -p prevalence.csv -s scores.csv -t table.csv
#
# where 'labels.csv' is one or more CSV files containing the labels, 'predictions.csv' is one or more CSV files containing the
# predictions, 'prevalence.csv' is one or more CSV files containing labels to define the prevalence of the positive class at
# different ages, 'scores.csv' (optional) is a collection of scores for the predictions, and 'table.csv' (optional) is a table
# summary of the predictions at different ages.
#
# The Challenge webpage describes the file formats and scoring functions for this script.

# Import packages.
import argparse
import numpy as np
import os
import os.path
import pandas as pd
import sys

from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score

# Define headers.
id_site = 'SiteID'
id_patient = 'BDSPPatientID'
id_label = 'Cognitive_Impairment'
id_age = 'Age'
id_sex = 'Sex'
id_binary_prediction = 'Cognitive_Impairment'
id_probability_prediction = 'Cognitive_Impairment_Probability'

# Parse arguments.
def get_parser():
    description = 'Evaluate the Challenge model.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--labels_files', type=str, required=True, nargs='*')
    parser.add_argument('-o', '--predictions_files', type=str, required=True, nargs='*')
    parser.add_argument('-p', '--prevalence_files', type=str, required=True, nargs='*')
    parser.add_argument('-s', '--score_file', type=str, required=False)
    parser.add_argument('-t', '--table_file', type=str, required=False)
    return parser

# Compute the prevalence of the positive class at different ages.
def compute_prevalence(ages, prevalence_labels, prevalence_ages, gap=0):
    m = len(ages)
    n = len(prevalence_labels)
    o = len(prevalence_ages)
    assert(n == o)

    unique_ages = np.unique(ages[np.isfinite(ages)])
    age_to_labels = defaultdict(list)
    for age in unique_ages:
        for i in range(n):
            if abs(age-prevalence_ages[i]) <= gap:
                age_to_labels[age].append(prevalence_labels[i])

    age_to_prevalence = dict()
    for age in age_to_labels:
        age_to_prevalence[age] = sum(age_to_labels[age])/len(age_to_labels[age])

    return age_to_prevalence

# Compute a prevalence-based reward metric.
def compute_reward(labels, predictions, ages, age_to_prevalence, min_prevalence=1e-6, max_prevalence=1-1e-6):
    m = len(labels)
    n = len(predictions)
    o = len(ages)
    assert(m == n == o)

    scores = np.zeros(n)
    num_scores = 0

    for i in range(n):
        if np.isfinite(ages[i]):
            p = age_to_prevalence[ages[i]]
            p = min(max(p, min_prevalence), max_prevalence) # Ensure p is strictly greater than 0 and strictly less than 1.

            if labels[i] == 1 and predictions[i] == 1:
                scores[i] = 1/p - 1
            elif labels[i] == 0 and predictions[i] == 1:
                scores[i] = -1
            if labels[i] == 1 and predictions[i] == 0:
                scores[i] = -1
            elif labels[i] == 0 and predictions[i] == 0:
                scores[i] = 1/(1-p) - 1
            num_scores += 1

    score = np.sum(scores)/num_scores

    return score

# Compute the area under the receiver-operating characteristic curve conditioned on age.
def compute_auroc_age(labels, predictions, ages, gap=0):
    m = len(labels)
    n = len(predictions)
    o = len(ages)
    assert(m == n == o)

    idx_pos = [i for i in range(m) if labels[i] == 1]
    idx_neg = [i for i in range(m) if labels[i] == 0]
    num_pos = len(idx_pos)
    num_neg = len(idx_neg)
    
    numer = 0
    denom = 0
    for i in range(num_pos):
        for j in range(num_neg):
            if abs(ages[idx_pos[i]] - ages[idx_neg[j]]) <= gap:
                if predictions[idx_pos[i]] > predictions[idx_neg[j]]:
                    numer += 1
                elif predictions[idx_pos[i]] == predictions[idx_neg[j]]:
                    numer += 0.5
                denom += 1
    return numer/denom

# Compute the age-weighted mean of the area under the receiver-operating characteristic curve across ages.
def compute_auroc_weighted(labels, predictions, ages, gap=0):
    m = len(labels)
    n = len(predictions)
    o = len(ages)
    assert(m == n == o)

    finite_ages = ages[np.isfinite(ages)]
    range_ages = np.arange(np.min(finite_ages) - gap, np.max(finite_ages) + gap + 1)
    p = len(finite_ages)
    q = len(range_ages)

    idx_pos = [i for i in range(m) if labels[i] == 1]
    idx_neg = [i for i in range(m) if labels[i] == 0]
    num_pos = len(idx_pos)
    num_neg = len(idx_neg)

    numer = np.zeros(q)
    denom = np.zeros(q)
    for k in range(q):
        for i in range(num_pos):
            for j in range(num_neg):
                if abs(ages[idx_pos[i]] - range_ages[k]) <= gap and abs(ages[idx_neg[j]] - range_ages[k]) <= gap:
                    if predictions[idx_pos[i]] > predictions[idx_neg[j]]:
                        numer[k] += 1
                    elif predictions[idx_pos[i]] == predictions[idx_neg[j]]:
                        numer[k] += 0.5
                    denom[k] += 1              

    weights = np.zeros(q)
    for k in range(q):
        weights[k] = sum(1 for i in range(p) if abs(finite_ages[i] - range_ages[k]) <= gap)

    for k in range(q):
        if denom[k] == 0: # Avoid NaN propagation
            weights[k] = 0
            numer[k] = 0
            denom[k] = 1

    weights = weights / np.sum(weights)

    return np.sum(weights * (numer / denom))

# Compute the area under the receiver-operating characteristic curve.
def compute_auroc(labels, predictions):
    auroc = roc_auc_score(labels, predictions, sample_weight=None, max_fpr=None, multi_class='raise', labels=None)
    return auroc

# Compute the area under the precision-recall curve.
def compute_auprc(labels, predictions):
    auprc = average_precision_score(labels, predictions, pos_label=1, sample_weight=None)
    return auprc

# Compute a confusion matrix.
def compute_confusion_matrix(labels, predictions):
    n = np.size(labels)
    tp = fp = fn = tn = 0
    for i in range(n):
        if labels[i] == 1 and predictions[i] == 1:
            tp += 1
        elif labels[i] == 0 and predictions[i] == 1:
            fp += 1
        elif labels[i] == 1 and predictions[i] == 0:
            fn += 1
        elif labels[i] == 0 and predictions[i] == 0:
            tn += 1
    return tp, fp, fn, tn

# Compute accuracy.
def compute_accuracy(labels, predictions):
    tp, fp, fn, tn = compute_confusion_matrix(labels, predictions)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return accuracy

# Compute the F-measure.
def compute_f_measure(labels, predictions):
    tp, fp, fn, tn = compute_confusion_matrix(labels, predictions)
    f_measure =  (2*tp) / (2*tp + fp + fn)
    return f_measure

# Evaluate the models.
def evaluate_model(labels_files, predictions_files, prevalence_files):
    # Load the labels, predictions, and prevalence data.
    id_site_patient = id_site + '_' + id_patient

    df_labels = [pd.read_csv(labels_file) for labels_file in labels_files]
    df_labels = pd.concat(df_labels)
    df_labels.drop_duplicates(inplace=True)
    df_labels[id_site_patient] = df_labels[id_site].astype(str) + '_' + df_labels[id_patient].astype(str)
    df_labels.set_index(id_site_patient, inplace=True)

    df_predictions = [pd.read_csv(predictions_file) for predictions_file in predictions_files]
    df_predictions = pd.concat(df_predictions)
    df_predictions.drop_duplicates(inplace=True)
    df_predictions[id_site_patient] = df_predictions[id_site].astype(str) + '_' + df_predictions[id_patient].astype(str)
    df_predictions.set_index(id_site_patient, inplace=True)

    df_prevalence = [pd.read_csv(prevalence_file) for prevalence_file in prevalence_files]
    df_prevalence = pd.concat(df_prevalence)
    df_prevalence.drop_duplicates(inplace=True)
    df_prevalence[id_site_patient] = df_prevalence[id_site].astype(str) + '_' + df_prevalence[id_patient].astype(str)
    df_prevalence.set_index(id_site_patient, inplace=True)

    # Only consider patients with positive or negative labels.
    df_labels = df_labels[(df_labels[id_label] == 0) | (df_labels[id_label] == 1)]
    patients = df_labels.index
    num_patients = len(patients)

    # Extract the labels, predictions, and ages.
    labels = np.zeros(num_patients)
    binary_predictions = np.zeros(num_patients)
    probability_predictions = np.zeros(num_patients)
    ages = np.zeros(num_patients)

    for i, patient in enumerate(patients):
        label = df_labels.loc[patient, id_label]
        labels[i] = label
        if patient in df_predictions.index:   # Set missing predictions to 0.
            binary_prediction = float(df_predictions.loc[patient, id_binary_prediction])
            if binary_prediction == 0 or binary_prediction == 1:   # Set invalid binary predictions to 0.
                binary_predictions[i] = binary_prediction
            probability_prediction = float(df_predictions.loc[patient, id_probability_prediction])
            if np.isfinite(probability_prediction):   # Set invalid probability predictions to 0.
                probability_predictions[i] = probability_prediction
        age = df_labels.loc[patient, id_age]
        ages[i] = age

    # Validate the labels, predictions, and ages.
    assert(labels.ndim == binary_predictions.ndim == probability_predictions.ndim == ages.ndim == 1)
    assert(np.size(labels) == np.size(binary_predictions) == np.size(probability_predictions) == np.size(ages))
    assert(np.size(labels) > 0)
    assert(np.all((labels==0) | (labels==1)))
    assert(np.all((binary_predictions==0) | (binary_predictions==1)))
    assert(np.all(~np.isnan(probability_predictions)))

    # Extract the prevalence of the positive class at different ages.
    df_prevalence = df_prevalence[(df_prevalence[id_label] == 0) | (df_prevalence[id_label] == 1)]
    prevalence_patients = df_prevalence.index

    num_prevalence_patients = len(prevalence_patients)
    prevalence_labels = np.zeros(num_prevalence_patients)
    prevalence_ages = np.zeros(num_prevalence_patients)  
    
    for i, patient in enumerate(prevalence_patients):
        label = df_prevalence.loc[patient, id_label]
        prevalence_labels[i] = label
        age = df_prevalence.loc[patient, id_age]
        prevalence_ages[i] = age

    age_to_prevalence = compute_prevalence(ages, prevalence_labels, prevalence_ages, gap=1)

    # Evaluate the predictions.
    reward = compute_reward(labels, binary_predictions, ages, age_to_prevalence)
    auroc_age = compute_auroc_age(labels, probability_predictions, ages, gap=1)
    auroc_weighted = compute_auroc_weighted(labels, probability_predictions, ages, gap=1)
    auroc = compute_auroc(labels, probability_predictions)
    auprc = compute_auprc(labels, probability_predictions)
    accuracy = compute_accuracy(labels, binary_predictions)
    f_measure = compute_f_measure(labels, binary_predictions)
    
    table = list()
    
    header = ['Age', 'Prevalence (prevalence data)', '# positive labels (prevalence data)', '# negative labels (prevalence data)', \
              '# positive labels', '# positive labels', '# positive predictions', '# negative predictions', \
              '# true positives', '# false positives', '# false negatives', '# true negatives']
    table.append(header)

    m = len(prevalence_labels)
    n = len(labels)
    for age in np.unique(np.concatenate((ages, list(age_to_prevalence)))):
        prevalence = age_to_prevalence[age] if age in age_to_prevalence else float('nan')
        pa = sum(1 for i in range(m) if prevalence_ages[i] == age and prevalence_labels[i] == 1)
        na = sum(1 for i in range(m) if prevalence_ages[i] == age and prevalence_labels[i] == 0)
        pb = sum(1 for i in range(n) if ages[i] == age and labels[i] == 1)
        nb = sum(1 for i in range(n) if ages[i] == age and labels[i] == 0)
        pc = sum(1 for i in range(n) if ages[i] == age and binary_predictions[i] == 1)
        nc = sum(1 for i in range(n) if ages[i] == age and binary_predictions[i] == 0)                  
        tp = sum(1 for i in range(n) if ages[i] == age and labels[i] == 1 and binary_predictions[i] == 1)
        fp = sum(1 for i in range(n) if ages[i] == age and labels[i] == 0 and binary_predictions[i] == 1)
        fn = sum(1 for i in range(n) if ages[i] == age and labels[i] == 1 and binary_predictions[i] == 0)
        tn = sum(1 for i in range(n) if ages[i] == age and labels[i] == 0 and binary_predictions[i] == 0)
        row = [age, age_to_prevalence[age], pa, na, pb, nb, pc, nc, tp, fp, fn, tn]
        table.append(row)

    return reward, auroc_age, auroc_weighted, auroc, auprc, accuracy, f_measure, table

# Run the code.
def run(args):
    # Compute the scores for the model predictions.
    reward, auroc_age, auroc_weighted, auroc, auprc, accuracy, f_measure, table = evaluate_model(args.labels_files, args.predictions_files, args.prevalence_files)

    output_string = \
        f'Reward: {reward:.3f}\n' + \
        f'Age-conditioned AUROC : {auroc_age:.3f}\n' + \
        f'Age-weighted AUROC : {auroc_weighted:.3f}\n' + \
        f'AUROC: {auroc:.3f}\n' + \
        f'AUPRC: {auprc:.3f}\n' + \
        f'Accuracy: {accuracy:.3f}\n' + \
        f'F-measure: {f_measure:.3f}\n'

    # Output the scores to the screen or a file.
    if args.score_file:
        with open(args.score_file, 'w') as f:
            f.write(output_string)
    else:
        print(output_string)

    # Output a table a breakdown of the results by age to a file.
    if args.table_file:
        with open(args.table_file, 'w') as f:
            table_string = '\n'.join('\t'.join(map(str, row)) for row in table)
            f.write(table_string)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
