import os
import pandas as pd

demotest = pd.read_csv("data/test_set/demographics.csv")

cache_path = ".feature_cache"

for i in demotest.index:
    h = demotest.loc[i, "SiteID"]
    f = demotest.loc[i, "BidsFolder"]
    session = demotest.loc[i, "SessionID"]
    if os.path.exists(os.path.join(cache_path, h, f+"_ses-" + str(session)+".csv")):
        print(f"Removing {os.path.join(cache_path, h, f+"_ses-" + str(session)+".csv")}")
        os.remove(os.path.join(cache_path, h, f+"_ses-" + str(session)+".csv"))
    if os.path.exists(os.path.join(cache_path, h, f+"_ses-" + str(session)+".sav")):
        print(f"Removing {os.path.join(cache_path, h, f+"_ses-" + str(session)+".sav")}")
        os.remove(os.path.join(cache_path, h, f+"_ses-" + str(session)+".sav"))

os.remove(os.path.join(cache_path, "exports", "test_features_raw.csv"))
os.remove(os.path.join(cache_path, "exports", "test_features_preprocessed.csv"))