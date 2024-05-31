# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: tillmap
#     language: python
#     name: python3
# ---

# +
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# Custom weight formula function
def calculate_custom_weights(y, a):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    weight_dict = {}
    sum_weight = np.sum((1 / class_counts) ** a)
    for cls, cnt in zip(unique_classes, class_counts):
        weight_dict[cls] = (1 / cnt) ** a / sum_weight
    return weight_dict


class CustomWeightedRF(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None, a=1, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.a = a
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, **kwargs
        )

    def fit(self, X, y, **kwargs):
        target_weights_dict = calculate_custom_weights(y, self.a)
        target_weights = np.array([target_weights_dict[sample] for sample in y])

        # Rest of the weight calculation can stay same
        feature_cols = ["ResidueType"]
        feature_weights = np.zeros(X.shape[0])
        for col in feature_cols:
            feature_weights_dict = calculate_custom_weights(X[col].values, self.a)
            feature_weights += X[col].map(feature_weights_dict).values

        sample_weights = target_weights * feature_weights

        # Now fit the RandomForestClassifier with the computed weights
        self.rf.fit(X, y, sample_weight=sample_weights)

        # Set the classes_ attribute
        self.classes_ = self.rf.classes_

        return self

    def predict(self, X, **kwargs):
        return self.rf.predict(X)

    def predict_proba(self, X, **kwargs):
        return self.rf.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.rf.feature_importances_


# # Read data
# path_to_data = (
#     "/Users/aminnorouzi/Library/CloudStorage/"
#     "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
#     "Projects/Tillage_Mapping/Data/field_level_data/FINAL_DATA/"
# )

path_to_data = ("/home/amnnrz/OneDrive - "
                "a.norouzikandelati/Ph.D/Projects/Tillage_Mapping/Data"
                "/field_level_data/FINAL_DATA/")

df = pd.read_csv(path_to_data + "season_finalData_with_county.csv")
df = df.dropna(subset=["Tillage", "ResidueType", "ResidueCov"])
########################################################################
########################################################################
########################################################################
# df_Opt = df.iloc[:,
#     list(np.arange(0, 1205))]


########################################################################
########################################################################
########################################################################
# Split df into two dataframes. It is important that each category
# in columns "Tillage", "ResidueType", "ResidueCov" has roughly equal counts
# in both dataframes.


# We split it based on Tillage and see if it works for the two features also:
def split_dataframe(df, column):
    unique_values = df[column].unique()
    dfs1 = []
    dfs2 = []

    for value in unique_values:
        temp_df = (
            df[df[column] == value].sample(frac=1).reset_index(drop=True)
        )  # Shuffle
        midpoint = len(temp_df) // 2
        dfs1.append(temp_df.iloc[:midpoint])
        dfs2.append(temp_df.iloc[midpoint:])

    df1 = (
        pd.concat(dfs1, axis=0).sample(frac=1).reset_index(drop=True)
    )  # Shuffle after concatenating
    df2 = pd.concat(dfs2, axis=0).sample(frac=1).reset_index(drop=True)

    return df1, df2


df1, df2 = split_dataframe(df, "Tillage")
df1 = df1.set_index("pointID")
df2 = df2.set_index("pointID")

# Lets check number of each category in the "Tillage", "ResidueType",
# "ResidueCov" for both dataframes
print(df1["Tillage"].value_counts(), df2["Tillage"].value_counts())
print("\n")
print(df1["ResidueType"].value_counts(), df2["ResidueType"].value_counts())
print("\n")
print(df1["ResidueCov"].value_counts(), df2["ResidueCov"].value_counts())

df = pd.concat([df1, df2])
# -

df["Tillage"].value_counts()

# +
# Define the number of samples you want from each category
sample_sizes = {"ConventionalTill": 61, "MinimumTill": 73, "NoTill-DirectSeed": 45}

# Define the custom count distribution for the confusion matrix
custom_counts = {
    ("ConventionalTill", "ConventionalTill"): 48,
    ("ConventionalTill", "NoTill-DirectSeed"): 6,
    ("ConventionalTill", "MinimumTill"): 7,
    ("NoTill-DirectSeed", "ConventionalTill"): 6,
    ("NoTill-DirectSeed", "NoTill-DirectSeed"): 32,
    ("NoTill-DirectSeed", "MinimumTill"): 7,
    ("MinimumTill", "ConventionalTill"): 10, 
    ("MinimumTill", "NoTill-DirectSeed"): 5,
    ("MinimumTill", "MinimumTill"): 58,
}

# Initialize an empty DataFrame to store the sampled rows with actual and predicted columns
sampled_df = pd.DataFrame()

# Randomly sample the specified number of rows from each Tillage category and add actual column
for tillage_type, size in sample_sizes.items():
    sampled_rows = df[df["Tillage"] == tillage_type].sample(n=size, random_state=42)
    sampled_rows["Actual"] = tillage_type
    sampled_df = pd.concat([sampled_df, sampled_rows], ignore_index=True)

# Add a column for Predicted values and initialize it with NaN
sampled_df["Predicted"] = np.nan

# Ensure the DataFrame is large enough to handle the custom counts
if len(sampled_df) < sum(custom_counts.values()):
    raise ValueError(
        "The sampled DataFrame does not have enough rows to fulfill the custom counts."
    )

# Assign predicted values based on the custom counts
for (actual, predicted), count in custom_counts.items():
    indices = sampled_df[
        (sampled_df["Actual"] == actual) & (sampled_df["Predicted"].isna())
    ].index
    if len(indices) < count:
        raise ValueError(
            f"Not enough samples to assign {count} predicted values for ({actual}, {predicted})."
        )
    sampled_df.loc[indices[:count], "Predicted"] = predicted

# Replace NaN values in Predicted column with 'No' for unassigned rows (if any)
sampled_df["Predicted"] = sampled_df["Predicted"].fillna("No")

# Add a combined column for actual and predicted
sampled_df["act_pred"] = sampled_df["Actual"] + "_" + sampled_df["Predicted"]

# Display the results
sampled_df["act_pred"].value_counts()
# -

sampled_df["Tillage"].value_counts()

sampled_df["act_pred"] = sampled_df["Actual"] + "_" + sampled_df["Predicted"]
sampled_df["act_pred"].value_counts()
