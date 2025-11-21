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
Dimensionality reduction and visual comparison between ground-truth and mapped
tillage datasets using PCA and t-SNE.

This notebook-style script:
1. Loads ground-truth and mapped per-field datasets.
2. Encodes categorical variables (Residue cover and crop type).
3. Applies PCA for feature compression.
4. Applies t-SNE for 2D visualization of feature similarity across years.
5. Saves combined results for each study year.
"""

# =============================================================================
# Library imports
# =============================================================================
import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely import wkt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# =============================================================================
# Define base paths (replace these placeholders with your actual project dirs)
# =============================================================================
BASE_CODE = Path("/path/to/project/code")
BASE_DATA = Path("/path/to/project/data")

# Ensure BASE_CODE is importable for any custom modules used in this pipeline
if str(BASE_CODE) not in sys.path:
    sys.path.append(str(BASE_CODE))

# =============================================================================
# Load and preprocess ground-truth dataset
# =============================================================================
gt = pd.read_csv(BASE_DATA / "dataset.csv")

# Fill missing numeric values with column medians to ensure stability in scaling
gt = gt.fillna(gt.median(numeric_only=True))

# Select relevant columns: metadata + spectral/feature bands
gt_cols = ['pointID', 'year', 'County', 'ResidueCov', 'cdl_cropType'] + gt.loc[:, "B_S0_p0":].columns.tolist()
gt_feats = gt[gt_cols]

# Encode residue cover and crop type categories numerically
fr_map = {'0-15%': 1, '16-30%': 2, '>30%': 3}
crop_map = {'Grain': 1, 'Legume': 2, 'Canola': 3}

gt_feats['ResidueCov'] = gt_feats['ResidueCov'].replace(fr_map)
gt_feats['cdl_cropType'] = gt_feats['cdl_cropType'].replace(crop_map)

# Rename columns for consistency
gt_feats = gt_feats.rename(columns={
    'ResidueCov': 'fr',
    'cdl_cropType': 'crop_type'
})

# Identify imagery feature columns (bands and indices)
feat_columns = gt_feats.loc[:, "fr":].columns.tolist()

# Label dataset source for later visualization
gt_feats.insert(1, 'Source', 'Ground-truth')

# =============================================================================
# Load and preprocess mapped dataset
# =============================================================================
map_df = pd.read_csv(BASE_DATA / "new_final_data" / "map_till_maj_full.csv")

map_cols = ['pointID', 'year', 'County', 'fr_maj', 'cdl_cropType'] + map_df.loc[:, "B_S0_p0":].columns.tolist()
map_feats = map_df[map_cols]

map_feats = map_feats.rename(columns={
    'fr_maj': 'fr',
    'cdl_cropType': 'crop_type'
})

# Tag the mapped dataset source and align columns with the ground-truth structure
map_feats.insert(1, 'Source', 'Map')
map_feats = map_feats[gt_feats.columns.tolist()]

# =============================================================================
# PCA transformation
# =============================================================================
def apply_pca(df):
    """
    Standardize imagery features and apply PCA to reduce dimensionality
    while retaining 70% of the variance.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset with spectral/feature bands starting at 'B_S0_p0'.

    Returns
    -------
    pandas.DataFrame
        Transformed dataset including PCA components, crop type,
        residue cover, and metadata.
    """
    scaler = StandardScaler()
    x_imagery = df.loc[:, "B_S0_p0":]
    x_scaled = scaler.fit_transform(x_imagery)

    # Keep enough components to explain 70% of total variance
    pca = PCA(n_components=0.7)
    x_pca = pca.fit_transform(x_scaled)
    x_pca = pd.DataFrame(x_pca, index=x_imagery.index)

    # Combine PCA components with metadata
    X = pd.concat(
        [
            df['pointID'],
            df['Source'],
            df['year'],
            df['County'],
            df['crop_type'],
            df['fr'],
            x_pca,
        ],
        axis=1,
    )
    X.columns = X.columns.astype(str)
    return X

# Apply PCA to both datasets
gt_pc = apply_pca(gt_feats)
map_pc = apply_pca(map_feats)

# =============================================================================
# t-SNE visualization
# =============================================================================
for year in [2012, 2017, 2022]:
    print(f"Processing year: {year}")

    # Subset mapped data for the target year
    map_filtered = map_feats.loc[map_feats['year'] == year].copy()

    # Combine mapped and ground-truth features for joint embedding
    combined_feature_set = pd.concat([map_filtered, gt_feats], axis=0, ignore_index=True)
    features_for_tsne = combined_feature_set[feat_columns].copy()

    # Apply t-SNE to project high-dimensional features into 2D
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features_for_tsne)

    # Append t-SNE results to the dataframe
    combined_feature_set['tsne_1'] = tsne_results[:, 0]
    combined_feature_set['tsne_2'] = tsne_results[:, 1]

    # Save per-year t-SNE embeddings
    combined_feature_set.to_csv(
        BASE_DATA / "new_final_data" / f"tsne_results_on_pca_{year}.csv",
        index=False,
    )
# -
