# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: tillmap
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import geopandas as gpd
import os

# +
path_to_data = ('/Users/aminnorouzi/Library/CloudStorage/'
                'OneDrive-WashingtonStateUniversity(email.wsu.edu)/'
                'Ph.D/Projects/Tillage_Mapping/Data/')


# -

# # Read data

# +
data_2012 = pd.read_csv(
    path_to_data + "MAPPING_DATA_2011_2012_2022/2012_2017_2022/data_2012.csv"
)

data_2017 = pd.read_csv(
    path_to_data + "MAPPING_DATA_2011_2012_2022/2012_2017_2022/data_2017.csv"
)

data_2022 = pd.read_csv(
    path_to_data + "MAPPING_DATA_2011_2012_2022/2012_2017_2022/data_2022.csv"
)
# -

# # Predict fr

# +
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

# Load best fr classifer
fr_classifier = joblib.load(path_to_data + "best_models/best_fr_classifier.pkl")

# Load the saved scaler for fr
scaler = joblib.load(path_to_data + "best_models/fr_scaler_model.pkl")

# Apply PCA
# Load the PCA object used during training
pca = joblib.load(path_to_data + "best_models/fr_pca_model.pkl")

def pred_fr(df):
    x_imagery = df.loc[:, "B_S0_p0":]
    x_imagery_scaled = scaler.transform(x_imagery)

    x_imagery_pca = pca.transform(x_imagery_scaled)
    x_imagery_pca = pd.DataFrame(x_imagery_pca)
    x_imagery_pca.set_index(x_imagery.index, inplace=True)

    X = pd.concat(
        [
            df["cdl_cropType"],
            df["min_NDTI_S0"],
            df["min_NDTI_S1"],
            x_imagery_pca,
        ],
        axis=1,
    )
    X.columns = X.columns.astype(str)
    y_preds = fr_classifier.predict(X)
    df["ResidueCov"] = y_preds
    cols = list(df.columns)
    # Move the merged column to the 4th position
    cols.insert(3, cols.pop(cols.index("ResidueCov")))
    df = df[cols]
    return df
data_2022 = pred_fr(data_2022)
data_2017 = pred_fr(data_2017)
data_2012 = pred_fr(data_2012)
# -

# # Predict tillage

# +
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# Custom weight formula function
def calculate_custom_weights(y, a):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    weight_dict = {}
    sum_weight = np.sum((1 / class_counts) ** a)
    for cls, cnt in zip(unique_classes, class_counts):
        weight_dict[cls] = (1 / cnt) ** a / sum_weight
    return weight_dict


class CustomWeightedRF(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        a=1,
        max_features=None,
        min_samples_split=None,
        min_samples_leaf=None,
        bootstrap=None,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.min_samples_split = min_samples_split
        self.a = a
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, **kwargs
        )

    def fit(self, X, y, **kwargs):
        # Calculate the target weights based on 'a'
        target_weights_dict = calculate_custom_weights(y, self.a)
        target_weights = np.array([target_weights_dict[sample] for sample in y])

        # If a == 0, remove "cdl_cropType" from the dataset
        if self.a == 0:
            X_mod = X.drop(columns=["cdl_cropType"])
            feature_weights = np.ones(X_mod.shape[0])  # No feature weights in this case
        else:
            X_mod = X.copy()
            feature_cols = ["cdl_cropType"]
            feature_weights = np.zeros(X_mod.shape[0])
            for col in feature_cols:
                feature_weights_dict = calculate_custom_weights(
                    X_mod[col].values, self.a
                )
                feature_weights += X_mod[col].map(feature_weights_dict).values

        # Calculate sample weights by combining target and feature weights
        sample_weights = target_weights * feature_weights

        # Fit the RandomForestClassifier with the computed weights and modified dataset
        self.rf.fit(X_mod, y, sample_weight=sample_weights)

        # Set the classes_ attribute
        self.classes_ = self.rf.classes_

        return self

    def predict(self, X, **kwargs):
        if self.a == 0:
            X_mod = X.drop(columns=["cdl_cropType"])
        else:
            X_mod = X.copy()
        return self.rf.predict(X_mod)

    def predict_proba(self, X, **kwargs):
        if self.a == 0:
            X_mod = X.drop(columns=["cdl_cropType"])
        else:
            X_mod = X.copy()
        return self.rf.predict_proba(X_mod)

    @property
    def feature_importances_(self):
        return self.rf.feature_importances_


# +
# Load best fr classifer
tillage_classifier = joblib.load(path_to_data + "best_models/best_tillage_classifier.pkl")

# Load the saved scaler for fr
scaler = joblib.load(path_to_data + "best_models/tillage_scaler_model.pkl")

# Apply PCA
# Load the PCA object used during training
pca = joblib.load(path_to_data + "best_models/tillage_pca_model.pkl")

def pred_tillage(df):
    x_imagery = df.loc[:, "B_S0_p0":]
    x_imagery_scaled = scaler.transform(x_imagery)
    x_imagery_pca = pca.transform(x_imagery_scaled)
    x_imagery_pca = pd.DataFrame(x_imagery_pca)
    x_imagery_pca.set_index(x_imagery.index, inplace=True)

    X = pd.concat(
        [
            df["cdl_cropType"],
            df["min_NDTI_S0"],
            df["min_NDTI_S1"],
            df["ResidueCov"],
            x_imagery_pca,
        ],
        axis=1,
    )

    to_replace = {"0-15%": 1, "16-30%": 2, ">30%": 3}
    X["ResidueCov"] = X["ResidueCov"].replace(to_replace)
    X
    X.columns = X.columns.astype(str)
    y_preds = tillage_classifier.predict(X)
    df["Tillage"] = y_preds
    cols = list(df.columns)
    # Move the merged column to the 4th position
    cols.insert(2, cols.pop(cols.index("Tillage")))
    df = df[cols]
    return df

data_2022 = pred_tillage(data_2022)
data_2017 = pred_tillage(data_2017)
data_2012 = pred_tillage(data_2012)
# -

data_2022

data_2022.to_csv(
    path_to_data + "MAPPING_DATA_2011_2012_2022/mapped_data/mapped_2022.csv", index=False)
data_2017.to_csv(
    path_to_data + "MAPPING_DATA_2011_2012_2022/mapped_data/mapped_2017.csv",
    index=False,
)
data_2012.to_csv(
    path_to_data + "MAPPING_DATA_2011_2012_2022/mapped_data/mapped_2012.csv",
    index=False,
)
