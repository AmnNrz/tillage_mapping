# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# +
import pandas as pd
import numpy as np

path_to_data = ("/Users/aminnorouzi/Library/CloudStorage/"
                "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
                "Ph.D/Projects/Tillage_Mapping/Data/")

lsat_data = pd.read_csv(
    path_to_data + "field_level_data/FINAL_DATA/Landsat_metricBased.csv"
)
s1_data = pd.read_csv(
    path_to_data + "field_level_data/FINAL_DATA/Sentinel_1_metricBased.csv"
)
cdl_data = pd.read_csv(path_to_data + "field_level_data/FINAL_DATA/cdl_df.csv")

to_replace = {23: "Grain", 31: "Canola", 24: "Grain", 51: "Legume", 
              53: "Legume", 61: "Fallow/Idle Cropland", 52: "Legume",
              176: "Grassland/Pasture", 35: "Mustard", 21: "Grain",
              36: "Alfalfa"
}

cdl_data["most_frequent_crop"] = cdl_data["most_frequent_crop"].replace(to_replace)
cdl_data = cdl_data.loc[
    cdl_data["most_frequent_crop"].isin(["Grain", "Legume", "Canola"])
].copy()

############ Merge cdl ############
# Merge the specific column from df2 into df1 based on 'pointID'
lsat_data = pd.merge(
    lsat_data, cdl_data[["pointID", "most_frequent_crop"]], on="pointID", how="left"
)

# Rearrange the columns to place the merged column in the 4th position
cols = list(lsat_data.columns)
# Move the merged column to the 4th position (index 3)
cols.insert(7, cols.pop(cols.index("most_frequent_crop")))

# Reorder the DataFrame
lsat_data = lsat_data[cols]

# Rename crop type columns (survey: "PriorCropT", cdl:"most_frequent_crop")
lsat_data.rename(columns={"PriorCropT":"survey_cropType",
                          "most_frequent_crop":"cdl_cropType"}, inplace=True)

# Remove NaN from cdl
lsat_data = lsat_data.dropna(subset=["cdl_cropType", "ResidueCov"])

# # Encode crop type
# to_replace = {"Grain":1, "Legume": 2, "Canola":3}
# lsat_data["cdl_cropType"] = lsat_data["cdl_cropType"].replace(to_replace)

# Fill NaN in Landsat data
imagery_data = lsat_data.loc[:, "B_S0_p0":].copy()
lsat_data.loc[:, "B_S0_p0":] = imagery_data.fillna(imagery_data.mean())
lsat_data = lsat_data.reset_index(drop=True)
# lsat_data.to_csv(path_to_data + "aaaaa.csv")
lsat_data = pd.read_csv(path_to_data + "aaaaa.csv")
# Encode crop type
to_replace = {"Grain": 1, "Legume": 2, "Canola": 3}
lsat_data["cdl_cropType"] = lsat_data["cdl_cropType"].replace(to_replace)
lsat_data = lsat_data.set_index("pointID")

# +
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


scaler = StandardScaler()

# Apply PCA
x_imagery = lsat_data.loc[:, "B_S0_p0":]
x_imagery_scaled = scaler.fit_transform(x_imagery)

pca = PCA(n_components=0.7)
x_imagery_pca = pca.fit_transform(x_imagery_scaled)
x_imagery_pca = pd.DataFrame(x_imagery_pca)
x_imagery_pca.set_index(x_imagery.index, inplace=True)

X = pd.concat([lsat_data["cdl_cropType"], x_imagery_pca], axis=1)
X.columns = X.columns.astype(str)
y = lsat_data["ResidueCov"]

groups = X["cdl_cropType"]

# Combine y and cdl_cropType into a single column to stratify by both
stratify_column = pd.DataFrame({"y": y, "cdl_cropType": groups})

# Perform stratified split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in sss.split(X, stratify_column):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],       # Number of trees in the forest
    'max_features': ['log2', 'sqrt'],      # Number of features to consider at every split
    'max_depth': [10, 20, 45],       # Maximum number of levels in each decision tree
    'min_samples_split': [2, 5, 10],       # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],         # Minimum number of samples required at each leaf node
    'bootstrap': [True, False]             # Method of selecting samples for training each tree
}


# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring="accuracy"
)

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: ", grid_search.best_score_)

# Use the best estimator from grid search
best_rf = grid_search.best_estimator_

# Predict on the test set
y_pred = best_rf.predict(X_test)

# Evaluate the predictions
print("Test set accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# +
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter

# Initialize dictionaries to store misclassified instances and their labels
misclassified_instances_validation = defaultdict(list)
misclassified_instances_test = defaultdict(list)
misclassified_labels_validation = defaultdict(list)
misclassified_labels_test = defaultdict(list)


scaler = StandardScaler()

# Apply PCA
x_imagery = lsat_data.loc[:, "B_S0_p0":]
x_imagery_scaled = scaler.fit_transform(x_imagery)

pca = PCA(n_components=0.7)
x_imagery_pca = pca.fit_transform(x_imagery_scaled)
x_imagery_pca = pd.DataFrame(x_imagery_pca)
x_imagery_pca.set_index(x_imagery.index, inplace=True)

X = pd.concat(
    [
        lsat_data["cdl_cropType"],
        lsat_data["min_NDTI_S0"],
        lsat_data["min_NDTI_S1"],
        x_imagery_pca,
    ],
    axis=1,
)
X.columns = X.columns.astype(str)
y = lsat_data["ResidueCov"]

groups = X["cdl_cropType"]

# Combine y and cdl_cropType into a single column to stratify by both
stratify_column = pd.DataFrame({"y": y, "cdl_cropType": groups})

# Perform stratified split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in sss.split(X, stratify_column):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="accuracy",
)

# Define the number of runs
n_runs = 10

# Run the process multiple times
for run in range(n_runs):
    # Fit the model
    grid_search.fit(X_train, y_train)

    # Use the best estimator from grid search
    best_rf = grid_search.best_estimator_

    # Predict on the test set
    y_pred = best_rf.predict(X_test)

    # Predict on the validation set (cross-validation predictions)
    y_train_pred = grid_search.predict(X_train)

    # Evaluate misclassified instances in validation and test sets
    misclassified_val_positions = np.where(y_train != y_train_pred)[0]
    misclassified_test_positions = np.where(y_test != y_pred)[0]

    # Store the misclassified instances and their labels
    for pos in misclassified_val_positions:
        idx = X_train.index[pos]
        misclassified_instances_validation[idx].append(run)
        misclassified_labels_validation[idx].append(
            (y_train.iloc[pos], y_train_pred[pos])
        )

    for pos in misclassified_test_positions:
        idx = X_test.index[pos]
        misclassified_instances_test[idx].append(run)
        misclassified_labels_test[idx].append((y_test.iloc[pos], y_pred[pos]))

    # Print results for each run (optional)
    print(f"Run {run+1}:")

    print("train set accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Test set accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Aggregate the results to find the most frequent misclassified instances
most_frequent_misclassified_val = Counter(
    misclassified_instances_validation
).most_common()
most_frequent_misclassified_test = Counter(misclassified_instances_test).most_common()

# Sort the instances based on the length of the misclassification count
most_frequent_misclassified_val = sorted(
    misclassified_instances_validation.items(),
    key=lambda item: len(item[1]),
    reverse=True,
)

most_frequent_misclassified_test = sorted(
    misclassified_instances_test.items(), key=lambda item: len(item[1]), reverse=True
)

# Print the sorted results
print("Most frequent misclassified instances in validation set (sorted by frequency):")
for idx, runs in most_frequent_misclassified_val:
    print(
        f"Index: {idx}, Count: {len(runs)}, True/Predicted Labels: {misclassified_labels_validation[idx]}"
    )

print("\nMost frequent misclassified instances in test set (sorted by frequency):")
for idx, runs in most_frequent_misclassified_test:
    print(
        f"Index: {idx}, Count: {len(runs)}, True/Predicted Labels: {misclassified_labels_test[idx]}"
    )

# +
# Sort the instances based on the length of the misclassification count
most_frequent_misclassified_val = sorted(
    misclassified_instances_validation.items(),
    key=lambda item: len(item[1]),
    reverse=True,
)

most_frequent_misclassified_test = sorted(
    misclassified_instances_test.items(), key=lambda item: len(item[1]), reverse=True
)

# Print the sorted results
print("Most frequent misclassified instances in validation set (sorted by frequency):")
for idx, runs in most_frequent_misclassified_val:
    print(
        f"Index: {idx}, Count: {len(runs)}, True/Predicted Labels: {misclassified_labels_validation[idx]}"
    )

print("\nMost frequent misclassified instances in test set (sorted by frequency):")
for idx, runs in most_frequent_misclassified_test:
    print(
        f"Index: {idx}, Count: {len(runs)}, True/Predicted Labels: {misclassified_labels_test[idx]}"
    )
# -

len(most_frequent_misclassified_val)

X_test.shape, X_train.shape
