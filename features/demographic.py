# features/demographic.py
# Demographic feature extraction — features 0-9
#
# Iteration history:
#   v1 (2026-06-30): Initial — age, sex onehot, race onehot, BMI (10 features)

import numpy as np
from helper_code import load_age, load_sex, load_bmi, load_race


def extract_demographic_features(data):
    """
    Extracts and encodes demographic features from a metadata dictionary.

    Returns np.ndarray of length 10:
        [0]   Age (continuous, capped at 90)
        [1-3] Sex one-hot  (Female, Male, Unknown)
        [4-8] Race one-hot (Asian, Black, Others, Unavailable, White)
        [9]   BMI (continuous — NaN for ~15% of patients, imputed downstream)
    """
    age = np.array([load_age(data)])

    sex = load_sex(data, standardize=True)
    sex_vec = np.zeros(3)
    if sex == 'Female':  sex_vec[0] = 1
    elif sex == 'Male':  sex_vec[1] = 1
    else:                sex_vec[2] = 1

    race = load_race(data, standardize=True)
    race_vec = np.zeros(5)
    if race == 'Asian':         race_vec[0] = 1
    elif race == 'Black':       race_vec[1] = 1
    elif race == 'Others':      race_vec[2] = 1
    elif race == 'Unavailable': race_vec[3] = 1
    elif race == 'White':       race_vec[4] = 1
    else:                       race_vec[2] = 1  # default to Others

    bmi = np.array([load_bmi(data)])

    return np.concatenate([age, sex_vec, race_vec, bmi])