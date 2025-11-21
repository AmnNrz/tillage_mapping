# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
"""
Compute majority-vote tillage (FR) predictions across model splits.

This notebook-style script:
1. Loads trained FR (Field Residue) classification bundles.
2. Reconstructs per-split models, PCA, and scalers using utility functions.
3. Builds standardized feature matrices using the same preprocessing pipeline.
4. Computes majority-vote predictions across all model splits.
5. Saves the final consensus predictions to a CSV file.
"""

# =============================================================================
# Library imports
# =============================================================================
import sys
from pathlib import Path
import pandas as pd

# =============================================================================
# Define base paths (replace these with actual directories)
# =============================================================================
BASE_CODE = Path("/path/to/project/code")
BASE_DATA = Path("/path/to/project/data")

# Add project code directory to Python path for custom module imports
if str(BASE_CODE) not in sys.path:
    sys.path.append(str(BASE_CODE))

# =============================================================================
# Import custom utilities
# =============================================================================
from utils import (
    load_splits_bundle,     # Loads model, scaler, and PCA objects for each split
    compute_majority_vote,  # Combines predictions across splits using majority vote
    make_build_X,           # Builds standardized feature matrix for inference
)

# =============================================================================
# Define input/output paths
# =============================================================================
models_dir  = BASE_DATA / "fr_models"
dataset_csv = BASE_DATA / "dataset_updated_2.csv"
out_path    = BASE_DATA / "fr_models" / "majority_vote_predictions.csv"

# =============================================================================
# Load trained model bundles
# =============================================================================
"""
Each split bundle includes:
- Trained model (fr_best_model_split_X.pkl)
- Scaler and PCA objects used in training
- Corresponding train/test feature and label CSVs

These are combined by `load_splits_bundle()` into a unified structure
for consistent inference and evaluation.
"""
bundles = load_splits_bundle(
    models_dir=models_dir,
    model_basename="fr_best_model_split_{split}.pkl",
    gridsearch_basename="fr_grid_search_split_{split}.pkl",
    x_train_basename="fr_X_train_split_{split}.csv",
    y_train_basename="fr_y_train_split_{split}.csv",
    x_test_basename="fr_X_test_split_{split}.csv",
    y_test_basename="fr_y_test_split_{split}.csv",
    scaler_basename="fr_scaler_split_{split}.pkl",
    pca_basename="fr_pca_split_{split}.pkl",
)

# =============================================================================
# Feature builder for inference
# =============================================================================
"""
The FR (Residue Cover) classifier uses:
- Crop type (encoded as integer)
- PCA-transformed spectral features (imagery_start_col onward)

`make_build_X()` constructs a callable that standardizes feature creation
for both training and inference.
"""
col_maps = {"cdl_cropType": {"Grain": 1, "Legume": 2, "Canola": 3}}

build_fr_X = make_build_X(
    imagery_start_col="B_S0_p0",      # first imagery column
    feature_cols=["cdl_cropType"],    # categorical metadata features
    column_maps=col_maps,             # mapping for crop types
)

# =============================================================================
# Compute majority-vote predictions
# =============================================================================
"""
`compute_majority_vote()`:
- Applies each split model to the dataset.
- Aggregates predictions for each pointID.
- Assigns the most frequent predicted label as the final result.
- If votes are tied, assigns the label specified by `tie_label`.
"""
mv_df = compute_majority_vote(
    bundles=bundles,
    dataset=str(dataset_csv),
    id_col="pointID",
    build_X_fn=build_fr_X,
    tie_label="TIE",
)

# =============================================================================
# Save output
# =============================================================================
mv_df.to_csv(out_path, index=False)
print("Saved majority-vote predictions to:", out_path)
# -
