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

lsat_data = pd.read_csv(path_to_data + "aaaaa.csv")
# Encode crop type
to_replace = {"Grain": 1, "Legume": 2, "Canola": 3}
lsat_data["cdl_cropType"] = lsat_data["cdl_cropType"].replace(to_replace)
lsat_data = lsat_data.set_index("pointID")
# -

# # Fr classification

# +
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],       # Number of trees in the forest
    'max_features': ['log2', 'sqrt'],      # Number of features to consider at every split
    'max_depth': [10, 20, 45],       # Maximum number of levels in each decision tree
    'min_samples_split': [2, 5, 10],       # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],         # Minimum number of samples required at each leaf node
    'bootstrap': [True, False]             # Method of selecting samples for training each tree
}

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


# # Calculate the class frequencies
# class_counts = y_train.value_counts()

# # Define the exponent base, you can adjust this based on the level of importance you want to give
# exponent_base = 2.0

# # Calculate class weights exponentially
# class_weights = {
#     cls: exponent_base ** (len(y_train) / count) for cls, count in class_counts.items()
# }

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42, class_weight="balanced")

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring="accuracy",
)

# Fit the model
grid_search.fit(X_train, y_train)

# Use the best estimator from grid search
best_rf = grid_search.best_estimator_

# Predict on the test set
y_pred = best_rf.predict(X_test)

# Predict on the validation set (cross-validation predictions)
y_train_pred = grid_search.predict(X_train)

print("train set accuracy: ", accuracy_score(y_train, y_train_pred))
print("Test set accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# -

# ### Plot micro and macro agveraged validation accuracies (boxplot)

# +
import matplotlib.pyplot as plt

###### Plot micro and macro averaged validatin accuracies
# Extract cross-validation results
cv_results = grid_search.cv_results_
validation_scores = cv_results["mean_test_score"]

# Calculate micro and macro averaged validation accuracies
micro_accuracies = validation_scores  # Since 'accuracy' is used, this is micro-average
macro_accuracies = validation_scores  # Same for macro-average in this context

# Plot box plots for micro and macro averaged validation accuracies
plt.figure(figsize=(8, 6))

# Create box plots with different colors
boxprops = dict(facecolor="#1b9e77", color="#1b9e77")
medianprops = dict(color="#1b9e77")
plt.boxplot(
    micro_accuracies,
    positions=[1],
    widths=0.6,
    patch_artist=True,
    boxprops=boxprops,
    medianprops=medianprops,
    showmeans=True,
    meanline=True,
)

boxprops = dict(facecolor="#7570b3", color="#7570b3")
medianprops = dict(color="#7570b3")
plt.boxplot(
    macro_accuracies,
    positions=[2],
    widths=0.6,
    patch_artist=True,
    boxprops=boxprops,
    medianprops=medianprops,
    showmeans=True,
    meanline=True,
)

# Set x-axis labels
plt.xticks([1, 2], ["Micro", "Macro"])

# Add labels and title
plt.ylabel("Validation Accuracy")
plt.title("Micro and Macro Averaged Validation Accuracies")

# Add legend
plt.legend(
    [
        plt.Line2D([0], [0], color="#1b9e77", lw=4),
        plt.Line2D([0], [0], color="#7570b3", lw=4),
    ],
    ["Micro", "Macro"],
)

plt.ylim([0.6, 1.0])

plt.show()
# -

# ### Plot confusion matrix of the test set

# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

# Assuming y_test and y_pred are already defined

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting parameters
x_labels = ["0-15%", "16-30%", ">30%"]  # Replace with your actual class names
y_labels = ["0-15%", "16-30%", ">30%"]  # Replace with your actual class names
cmap = LinearSegmentedColormap.from_list("custom_green", ["white", "#1b9e77"], N=256)

# Create the confusion matrix plot
plt.figure(figsize=(5, 5))
heatmap = sns.heatmap(
    conf_matrix,
    annot=False,
    fmt="d",
    cmap=cmap,
    cbar=True,
    vmin=0,
    vmax=np.max(conf_matrix),
)

cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)

# Annotate the heatmap with text
for i, row in enumerate(conf_matrix):
    for j, value in enumerate(row):
        color = "white" if value > np.max(conf_matrix) / 2 else "black"
        plt.text(
            j + 0.5,
            i + 0.5,
            str(value),
            ha="center",
            va="center",
            color=color,
            fontsize=32,
        )

# Set axis labels and ticks
plt.xlabel("Predicted Class", fontsize=24)
plt.ylabel("Actual Class", fontsize=24)
plt.xticks([0.5 + i for i in range(len(x_labels))], x_labels, fontsize=24, rotation=45)
plt.yticks([0.5 + i for i in range(len(y_labels))], y_labels, fontsize=24, rotation=45)

# Adjust layout and display
plt.tight_layout()
plt.show()
# -

# ### See test set frequent misclassified instances

# +
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],       # Number of trees in the forest
    'max_features': ['log2', 'sqrt'],      # Number of features to consider at every split
    'max_depth': [10, 20, 45],       # Maximum number of levels in each decision tree
    'min_samples_split': [2, 5, 10],       # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],         # Minimum number of samples required at each leaf node
    'bootstrap': [True, False]             # Method of selecting samples for training each tree
}

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
    cv=3,
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
# -

# ### See Validation folds misclassified instances

# +
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter

# Define the parameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_features": ["log2", "sqrt"],
    "max_depth": [10, 20, 45],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

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
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")

    # Define StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Define the number of runs
    n_runs = 3

    # Run the process multiple times
    for run in range(n_runs):
        print(run)
        fold_idx = 0
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Fit the model on the current fold
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=skf,
                n_jobs=-1,
                verbose=0,
                scoring="accuracy",
            )
            grid_search.fit(X_train_fold, y_train_fold)

            best_rf = grid_search.best_estimator_

            # Predict on the validation fold
            y_val_pred = best_rf.predict(X_val_fold)

            # Evaluate misclassified instances in the validation fold
            misclassified_val_positions = np.where(y_val_fold != y_val_pred)[0]
            for pos in misclassified_val_positions:
                idx = X_train.index[val_idx[pos]]
                misclassified_instances_validation[idx].append((run, fold_idx))
                misclassified_labels_validation[idx].append(
                    (y_val_fold.iloc[pos], y_val_pred[pos])
                )

            fold_idx += 1

        # Predict on the test set
        y_pred = best_rf.predict(X_test)

        # Evaluate misclassified instances in the test set
        misclassified_test_positions = np.where(y_test != y_pred)[0]
        for pos in misclassified_test_positions:
            idx = X_test.index[pos]
            misclassified_instances_test[idx].append(run)
            misclassified_labels_test[idx].append((y_test.iloc[pos], y_pred[pos]))

        # Print results for each run (optional)
        print(f"Run {run+1}:")
        print("Train set accuracy: ", accuracy_score(y_train, best_rf.predict(X_train)))
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
        f"Index: {idx}, Count: {len(runs)}, Runs/Folds: {runs}, True/Predicted Labels: {misclassified_labels_validation[idx]}"
    )

print("\nMost frequent misclassified instances in test set (sorted by frequency):")
for idx, runs in most_frequent_misclassified_test:
    print(
        f"Index: {idx}, Count: {len(runs)}, True/Predicted Labels: {misclassified_labels_test[idx]}"
    )
# -

# # Tillage Classification

# +
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score
# Custom scoring function for micro and macro averages
from sklearn.metrics import f1_score

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
        lsat_data["ResidueCov"],
        x_imagery_pca,
    ],
    axis=1,
)

to_replace = {"0-15%": 1, "16-30%": 2, ">30%": 3}
X["ResidueCov"] = X["ResidueCov"].replace(to_replace)
X
X.columns = X.columns.astype(str)
y = lsat_data["Tillage"]

groups = X["cdl_cropType"]

combo_counts = lsat_data.groupby(["Tillage", "cdl_cropType"]).size()
# Filter out combinations with fewer than 2 instances
valid_combos = combo_counts[combo_counts >= 2].index


stratify_column = pd.DataFrame({"y": y, "cdl_cropType": groups})

# Keep only rows where the combination of y and cdl_cropType is in the valid_combos
valid_mask = stratify_column.set_index(["y", "cdl_cropType"]).index.isin(valid_combos)
X_valid = X[valid_mask]
y_valid = y[valid_mask]

groups = X_valid["cdl_cropType"]

# Perform the stratified split with the valid data
stratify_column_valid = pd.DataFrame(
    {"y": y_valid, "cdl_cropType": X_valid["cdl_cropType"]}
)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in sss.split(X_valid, stratify_column_valid):
    X_train, X_test = X_valid.iloc[train_index], X_valid.iloc[test_index]
    y_train, y_test = y_valid.iloc[train_index], y_valid.iloc[test_index]

# Remove cdl crop type column
X_train_nocrop = X_train.drop("cdl_cropType", axis=1)
X_test_nocrop = X_test.drop("cdl_cropType", axis=1)


# +
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


def train_model(X_train, y_train, X_test, y_test, cv, param_grid, classifier):
    # Define custom micro and macro scoring
    scoring = {
        "micro_accuracy": make_scorer(f1_score, average="micro"),
        "macro_accuracy": make_scorer(f1_score, average="macro"),
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=3,
        scoring=scoring,
        refit="micro_accuracy",
        return_train_score=False,
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def plot_val_scores(grid_search, X_test, y_test):
    
    # Extracting cross-validation results
    cv_results = grid_search.cv_results_

    # Extract the validation accuracies for micro and macro scores
    micro_accuracies = cv_results["mean_test_micro_accuracy"]
    macro_accuracies = cv_results["mean_test_macro_accuracy"]

    # Create a box plot of micro and macro accuracies
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [micro_accuracies, macro_accuracies], labels=["Micro Accuracy", "Macro Accuracy"]
    )
    plt.title("Micro and Macro Averaged Validation Accuracies")
    plt.ylabel("Accuracy")
    plt.show()

    # Get the best model and test its accuracy on the test set
    best_model = grid_search.best_estimator_
    test_accuracy = accuracy_score(y_test, best_model.predict(X_test))

    print(f"Best model test accuracy: {test_accuracy}")

    # Plot the confusion matrix for the best model on the test set
    conf_matrix = confusion_matrix(y_test, best_model.predict(X_test))
    ConfusionMatrixDisplay(conf_matrix).plot()
    plt.title("Confusion Matrix for the Best Model")
    plt.show()


# +
param_grid = {
    "n_estimators": [50, 100, 300],
    "max_features": ["log2", "sqrt"],
    "max_depth": [5, 40, 55],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "a": list(
        np.around(
            np.concatenate((np.arange(0, 1, 0.3), np.arange(2, 12, 3))), decimals=1
        )
    ),
    # "a":[0],
    "bootstrap": [True, False],
}

grid_search = train_model(
    X_train, y_train, X_test, y_test, 3, param_grid, CustomWeightedRF()
)
# plot_val_scores(grid_search, X_test_nocrop, y_test)

# +
def plot_val_scores_by_a(grid_search, param_grid, X_test, y_test):
    # Extract cross-validation results
    cv_results = grid_search.cv_results_

    # Get the `a` values and the corresponding validation accuracies
    a_values = param_grid["a"]
    accuracies_by_a = {a: [] for a in a_values}

    # Collect accuracies for each `a` value from cv_results
    for i, a in enumerate(cv_results["param_a"]):
        accuracies_by_a[a].append(cv_results["mean_test_micro_accuracy"][i])

    # Create box plots for each `a` value
    plt.figure(figsize=(10, 6))
    box_data = [accuracies_by_a[a] for a in a_values]
    plt.boxplot(box_data, labels=[str(a) for a in a_values])

    # Calculate mean validation accuracy for each `a` value and plot a line
    mean_accuracies = [np.mean(accuracies_by_a[a]) for a in a_values]
    plt.plot(
        range(1, len(a_values) + 1),
        mean_accuracies,
        marker="o",
        linestyle="--",
        color="r",
        label="Mean Accuracy",
    )

    # Add titles and labels
    plt.title("Validation Accuracies Across Different 'a' Values")
    plt.xlabel("a values")
    plt.ylabel("Micro F1 Accuracy")
    plt.legend()
    plt.show()

    # Get the best model and test its accuracy on the test set
    best_model = grid_search.best_estimator_

    # If `a == 0`, remove "cdl_cropType" from the test set
    if best_model.a == 0:
        X_test_mod = X_test.drop(columns=["cdl_cropType"])
    else:
        X_test_mod = X_test.copy()

    # Test the accuracy of the best model
    test_accuracy = accuracy_score(y_test, best_model.predict(X_test_mod))
    print(f"Best model test accuracy: {test_accuracy}")

    # Plot the confusion matrix for the best model on the test set
    conf_matrix = confusion_matrix(y_test, best_model.predict(X_test_mod))
    ConfusionMatrixDisplay(conf_matrix).plot()
    plt.title("Confusion Matrix for the Best Model")
    plt.show()


plot_val_scores_by_a(grid_search, param_grid, X_test, y_test)
# -

# # check for correctly classified frequencies

# +
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from collections import defaultdict

# Custom scoring function for micro and macro averages
scoring = {
    "micro_accuracy": make_scorer(f1_score, average="micro"),
    "macro_accuracy": make_scorer(f1_score, average="macro"),
}

# Number of cross-validation loops
n_loops = 5

# **Changes here**: Define a larger dictionary to store frequencies of correct classifications across folds and loops
correct_classification_freq_val = defaultdict(int)
correct_classification_freq_test = defaultdict(int)

# StratifiedKFold to maintain class distribution across folds
cv = StratifiedKFold(n_splits=3)

for loop in range(n_loops):  # Iterate across loops
    print(f"Running loop {loop + 1}/{n_loops}")

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        CustomWeightedRF(),
        param_grid,
        cv=cv,
        scoring=scoring,
        refit="micro_accuracy",
        return_train_score=False,
    )

    grid_search.fit(X_train, y_train)  # Fit the model for this loop

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # **Change 1 here**: Track correct classifications for validation data across folds
    for train_idx, val_idx in cv.split(X_train, y_train):
        val_preds = best_model.predict(X_train.iloc[val_idx])
        correct_val = (
            val_preds == y_train.iloc[val_idx]
        )  # This tracks correct validation classifications

        for idx, correct in zip(val_idx, correct_val):  # Zip over index to accumulate
            if correct:
                # **Change 2 here**: Increment for each fold, across all loops
                correct_classification_freq_val[
                    X_train.index[idx]
                ] += 1  # Accumulating across all loops and folds

    # Test set accuracy tracking for each loop
    test_preds = best_model.predict(X_test)
    correct_test = test_preds == y_test

    for idx, correct in zip(X_test.index, correct_test):
        if correct:
            # **Change 3 here**: Increment for test set across all loops
            correct_classification_freq_test[idx] += 1  # Increment across loops

# Create a box plot of micro and macro accuracies
plt.figure(figsize=(10, 6))
plt.boxplot(
    [
        grid_search.cv_results_["mean_test_micro_accuracy"],
        grid_search.cv_results_["mean_test_macro_accuracy"],
    ],
    labels=["Micro Accuracy", "Macro Accuracy"],
)
plt.title("Micro and Macro Averaged Validation Accuracies")
plt.ylabel("Accuracy")
plt.show()

# Print the frequency of correctly classified instances for validation and test sets
print("Correct classification frequencies in validation folds:")
for idx, count in correct_classification_freq_val.items():
    print(
        f"Instance {idx}: Correctly classified {count} times"
    )  # Printing frequencies for all instances

print("Correct classification frequencies in the test set:")
for idx, count in correct_classification_freq_test.items():
    print(
        f"Instance {idx}: Correctly classified {count} times"
    )  # Printing test set results

# Test set accuracy
test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
print(f"Best model test accuracy: {test_accuracy}")

# Plot the confusion matrix for the best model on the test set
conf_matrix = confusion_matrix(y_test, best_model.predict(X_test))
ConfusionMatrixDisplay(conf_matrix).plot()
plt.title("Confusion Matrix for the Best Model")
plt.show()

# +
# Get feature importances from the best model
feature_importances = best_model.feature_importances_

# Create a sorted list of features and their importance
important_features = sorted(
    zip(X_train.columns, feature_importances), key=lambda x: x[1], reverse=True
)

# Print the most important features and their importances
print("Feature importances:")
for feature, importance in important_features:
    print(f"{feature}: {importance:.4f}")

# Optionally, plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh([x[0] for x in important_features], [x[1] for x in important_features])
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()
# -

lsat_data['cdl_cropType'].value_counts()
