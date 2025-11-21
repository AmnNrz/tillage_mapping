# train_tillage_fullfeatures.py
import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV

# ------------------------------------------------------------
# Make sure your module file is on sys.path, then import it
#   If you saved as: /home/.../code/custom_weighted_rf.py
#   this import is correct. Adjust if you used a package path.
# ------------------------------------------------------------
BASE_CODE = Path("/home/a.norouzikandelati/Projects/tillage_mapping/code")
if str(BASE_CODE) not in sys.path:
    sys.path.append(str(BASE_CODE))

from custom_rf import CustomWeightedRF  # or: from custom_models.custom_weighted_rf import CustomWeightedRF

# ----------------------------
# Paths & split number
# ----------------------------
path_to_data = "/home/a.norouzikandelati/Projects/data/tillage_mapping/"
models_dir   = os.path.join(path_to_data, "best_models/")
os.makedirs(models_dir, exist_ok=True)

split_num = int(sys.argv[1])  # 0..49 (or 1..50)

# ----------------------------
# Read dataset + add fr_maj
# ----------------------------
lsat_data = pd.read_csv(os.path.join(path_to_data, "dataset.csv"))
lsat_data = lsat_data.set_index("pointID")

# map crop type to ints (keeps your prior convention)
lsat_data["cdl_cropType"] = lsat_data["cdl_cropType"].replace({"Grain": 1, "Legume": 2, "Canola": 3})

# read majority vote file and merge as fr_maj
mv = pd.read_csv(os.path.join(path_to_data, "fr_models/majority_vote_predictions.csv"))
mv = mv.rename(columns={"majority_vote": "fr_maj"}).set_index("pointID")

# merge into dataset (left join to keep your dataset rows)
lsat_data = lsat_data.join(mv[["fr_maj"]], how="left")

# encode fr_maj to numeric; drop ties/missing if any
map_fr = {"0-15%": 1, "16-30%": 2, ">30%": 3}
lsat_data["fr_maj"] = lsat_data["fr_maj"].map(map_fr)


# ----------------------------
# Select imagery features with only p0 and p100
# ----------------------------
all_imagery_cols = lsat_data.loc[:, "B_S0_p0":].columns
use_cols = [c for c in all_imagery_cols if c.endswith(("_p0", "_p50", "_p100"))]
x_imagery = lsat_data[use_cols]

# ----------------------------
# Build features (scale+PCA)
# ----------------------------
scaler = StandardScaler()
x_imagery = lsat_data.loc[:, "B_S0_p0":]   # imagery-only block
x_imagery_scaled = scaler.fit_transform(x_imagery)

pca = PCA(n_components=0.7)  # keep 70% variance
x_imagery_pca = pca.fit_transform(x_imagery_scaled)

# save scaler/pca (per split for reproducibility)
joblib.dump(scaler, os.path.join(models_dir, f"tillage_scaler_split_{split_num}.pkl"))
joblib.dump(pca,    os.path.join(models_dir, f"tillage_pca_split_{split_num}.pkl"))

x_imagery_pca = pd.DataFrame(x_imagery_pca, index=x_imagery.index)

# Final X with cdl_cropType + fr_maj + PCA comps
X = pd.concat(
    [
        lsat_data["cdl_cropType"],   # categorical-as-int feature
        lsat_data["fr_maj"],         # majority-vote FR feature (encoded 1/2/3)
        x_imagery_pca,               # PCA features
    ],
    axis=1,
)
X.columns = X.columns.astype(str)

# Target
y = lsat_data["Tillage"]
groups = X["cdl_cropType"]

# ----------------------------
# Filter rare (Tillage × cdl_cropType) combos
# ----------------------------
combo_counts = pd.concat([y, groups], axis=1).groupby(["Tillage", "cdl_cropType"]).size()
valid_combos = combo_counts[combo_counts >= 2].index

stratify_column = pd.DataFrame({"y": y, "cdl_cropType": groups})
valid_mask = stratify_column.set_index(["y", "cdl_cropType"]).index.isin(valid_combos)

X_valid = X[valid_mask]
y_valid = y[valid_mask]
groups_valid = X_valid["cdl_cropType"]

stratify_column_valid = pd.DataFrame({"y": y_valid, "cdl_cropType": groups_valid})

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=split_num)
for train_index, test_index in sss.split(X_valid, stratify_column_valid):
    X_train, X_test = X_valid.iloc[train_index], X_valid.iloc[test_index]
    y_train, y_test = y_valid.iloc[train_index], y_valid.iloc[test_index]

# ----------------------------
# CV + Grid
# ----------------------------
cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}

param_grid = {
    "n_estimators": [5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200],
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40],
    "min_samples_split": [3, 4, 5, 6, 7, 8, 9, 10, 20],
    "bootstrap": [True, False],
    "a": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 4]
}

grid_search = GridSearchCV(
    CustomWeightedRF(),
    param_grid=param_grid,
    cv=cv_splitter,
    scoring=scoring,
    verbose=0,
    refit="f1_macro",
    return_train_score=True,
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# ----------------------------
# Save artifacts
# ----------------------------
joblib.dump(best_model, os.path.join(models_dir, f"tillage_best_model_split_{split_num}.pkl"))
joblib.dump(grid_search, os.path.join(models_dir, f"tillage_grid_search_split_{split_num}.pkl"))

X_train.to_csv(os.path.join(models_dir, f"tillage_X_train_split_{split_num}.csv"))
y_train.to_csv(os.path.join(models_dir, f"tillage_y_train_split_{split_num}.csv"))

X_test.to_csv(os.path.join(models_dir, f"tillage_X_test_split_{split_num}.csv"))
y_test.to_csv(os.path.join(models_dir, f"tillage_y_test_split_{split_num}.csv"))

print("Done")
