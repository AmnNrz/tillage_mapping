# fr_train_split.py
# =============================================================================
# Purpose:
#   Train a RandomForest classifier for fractional residue (FR) with a PCA
#   feature pipeline, using a stratified train/test split. Saves the fitted
#   scaler, PCA, best model, and grid-search object, plus the split datasets.
# =============================================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# =============================================================================
# Configuration & Paths
# =============================================================================
path_to_data = "/path/to/your/tillage_mapping_data_root/"             # <— replace with your data root
models_dir   = os.path.join(path_to_data, "fr_models")                # directory to store all artifacts
os.makedirs(models_dir, exist_ok=True)                                # create if missing (idempotent)

# Split ID from CLI (e.g., python fr_train_split.py 7)
split_num = int(sys.argv[1])                                          # reproducible seed for split

# =============================================================================
# Load Data & Basic Preparation
# =============================================================================
lsat_data = pd.read_csv(os.path.join(path_to_data, "dataset.csv"))    # load master dataset
lsat_data["cdl_cropType"] = lsat_data["cdl_cropType"].replace(        # encode crop types (consistent mapping)
    {"Grain": 1, "Legume": 2, "Canola": 3}
)
lsat_data = lsat_data.set_index("pointID")                            # use pointID as index

y = lsat_data["ResidueCov"]                                           # target variable (FR)

imagery_start_col = "B_S0_p0"                                         # first imagery-feature column
if imagery_start_col not in lsat_data.columns:                        # sanity check schema
    raise KeyError(f"'{imagery_start_col}' not found in dataset.csv")

x_imagery = lsat_data.loc[:, imagery_start_col:]                      # imagery (and following) columns
meta_cols = ["cdl_cropType"]                                          # auxiliary non-imagery feature

# =============================================================================
# Train/Test Split (Stratify on y + crop type)
# =============================================================================
groups = lsat_data["cdl_cropType"]                                    # for joint stratification
stratify_column = pd.DataFrame({"y": y, "cdl_cropType": groups})       # composite stratifier
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3,                # single split, 30% test
                            random_state=split_num)                    # seed by CLI

(train_idx, test_idx), = sss.split(x_imagery, stratify_column)         # indices for split

# Partition features/labels using the indices
x_img_tr, x_img_te = x_imagery.iloc[train_idx], x_imagery.iloc[test_idx]
y_train, y_test    = y.iloc[train_idx], y.iloc[test_idx]
meta_tr, meta_te   = (lsat_data.loc[x_img_tr.index, meta_cols],
                      lsat_data.loc[x_img_te.index, meta_cols])

# =============================================================================
# Scaling + PCA (Fit on TRAIN ONLY, apply to TEST)
# =============================================================================
scaler = StandardScaler()                                              # standardize features
x_img_tr_scaled = scaler.fit_transform(x_img_tr)                       # fit on train, transform train
x_img_te_scaled = scaler.transform(x_img_te)                           # transform test with same scaler

pca = PCA(n_components=0.7)                                            # keep components explaining 70% var
x_img_tr_pca = pca.fit_transform(x_img_tr_scaled)                      # fit PCA on train, transform train
x_img_te_pca = pca.transform(x_img_te_scaled)                          # transform test

# Wrap back into DataFrames, preserving indices for concatenation
x_img_tr_pca = pd.DataFrame(x_img_tr_pca, index=x_img_tr.index)
x_img_te_pca = pd.DataFrame(x_img_te_pca, index=x_img_te.index)

# Concatenate meta features with PCA components
X_train = pd.concat([meta_tr, x_img_tr_pca], axis=1)
X_test  = pd.concat([meta_te, x_img_te_pca], axis=1)
X_train.columns = X_train.columns.astype(str)                          # ensure string column names
X_test.columns  = X_test.columns.astype(str)

# =============================================================================
# Model & Hyperparameter Grid (GridSearchCV)
# =============================================================================
rf = RandomForestClassifier(random_state=42)  # base estimator

param_grid_baseline = {                                                 # broad but reasonable grid
    "n_estimators": [5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200],
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40],
    "min_samples_split": [3, 4, 5, 6, 7, 8, 9, 10, 20],
    "bootstrap": [True, False],
}

scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}             # optimize for balanced F1

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_baseline,
    cv=3,                                                               # 3-fold CV
    n_jobs=-1,                                                          # parallelize across cores
    verbose=1,                                                          # show progress
    scoring=scoring,                                                    # evaluate multiple metrics
    refit="f1_macro",                                                   # select best by macro-F1
    return_train_score=True,                                            # keep train scores for audit
)

# =============================================================================
# Fit, Evaluate (Quick Check), and Save Artifacts
# =============================================================================
grid_search.fit(X_train, y_train)                                      # run grid search
best_model = grid_search.best_estimator_                                # retrieve best RF

# Persist everything needed for inference/reproducibility
joblib.dump(grid_search, os.path.join(models_dir, f"fr_grid_search_split_{split_num}.pkl"))
joblib.dump(best_model,  os.path.join(models_dir, f"fr_best_model_split_{split_num}.pkl"))
joblib.dump(scaler,      os.path.join(models_dir, f"fr_scaler_split_{split_num}.pkl"))
joblib.dump(pca,         os.path.join(models_dir, f"fr_pca_split_{split_num}.pkl"))

# Save the splits (useful for debugging and downstream analysis)
X_train.to_csv(os.path.join(models_dir, f"fr_X_train_split_{split_num}.csv"))
y_train.to_csv(os.path.join(models_dir, f"fr_y_train_split_{split_num}.csv"))
X_test.to_csv( os.path.join(models_dir, f"fr_X_test_split_{split_num}.csv"))
y_test.to_csv( os.path.join(models_dir, f"fr_y_test_split_{split_num}.csv"))

print("Done.")
