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
              36: "Alfalfa", 42: "Legume", 37: "Hay, nonAlfalfa"
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

lsat_data = pd.read_csv(path_to_data + "aaaaat.csv")
# Encode crop type
to_replace = {"Grain": 1, "Legume": 2, "Canola": 3}
lsat_data["cdl_cropType"] = lsat_data["cdl_cropType"].replace(to_replace)
lsat_data = lsat_data.set_index("pointID")
# -

# # Fr classification

# Train fr classifier

# +
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    make_scorer,
    precision_score,
)
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Define the parameter grid
param_grid = {
    "n_estimators": [100, 200],  # Number of trees in the forest
    "max_features": ["log2", "sqrt"],  # Number of features to consider at every split
    "max_depth": [10, 20, 45],  # Maximum number of levels in each decision tree
    "min_samples_split": [
        2,
        5,
        10,
    ],  # Minimum number of samples required to split a node
    "min_samples_leaf": [
        1,
        2,
        4,
    ],  # Minimum number of samples required at each leaf node
    "bootstrap": [True, False],  # Method of selecting samples for training each tree
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

# Save the fitted PCA object
joblib.dump(pca, path_to_data + "best_models/fr_pca_model.pkl")

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

# Get the unique classes and their counts in the target variable
unique_classes, class_counts = np.unique(y_train, return_counts=True)

# Define the exponent 'b'
b = 0.7

# Calculate the weights for each class
class_weights = {
    cls: (1 / count) ** b for cls, count in zip(unique_classes, class_counts)
}

# Normalize the weights so that they sum to 1
total_weight = sum(class_weights[cls] for cls in unique_classes)
class_weights = {cls: weight / total_weight for cls, weight in class_weights.items()}

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42, class_weight=class_weights)

# Define multiple scoring metrics
scoring = {"precision_micro": "precision_micro", "precision_macro": "precision_macro"}

# Initialize GridSearchCV with 3-fold cross-validation and multiple scoring metrics
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring=scoring,
    refit="precision_macro",  # Choose which metric to optimize for best estimator
    return_train_score=True,
)

# Fit the model
grid_search.fit(X_train, y_train)

# Use the best estimator from grid search
best_rf = grid_search.best_estimator_

# Predict on the test set
y_pred = best_rf.predict(X_test)

# Predict on the training set
y_train_pred = best_rf.predict(X_train)

print("Train set accuracy: ", accuracy_score(y_train, y_train_pred))
print("Test set accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# -

# Plot micro- and macro-averaged precision accuracy across all combinations of hyper-parameters

# +
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Extract cross-validation results
cv_results = grid_search.cv_results_

# Initialize lists to hold all micro and macro precision scores
micro_precisions = []
macro_precisions = []

# Number of CV folds
n_splits = grid_search.cv

# Extract precision scores for each fold and parameter combination
# GridSearchCV stores split scores as split0_test_<scorer>, split1_test_<scorer>, etc.
for i in range(grid_search.cv):
    split_micro = cv_results[f"split{i}_test_precision_micro"]
    split_macro = cv_results[f"split{i}_test_precision_macro"]
    micro_precisions.extend(split_micro)
    macro_precisions.extend(split_macro)

# Create a DataFrame for plotting
precision_data = pd.DataFrame(
    {
        "Precision Type": ["Micro"] * len(micro_precisions)
        + ["Macro"] * len(macro_precisions),
        "Precision Score": micro_precisions + macro_precisions,
    }
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Set global font sizes
plt.rcParams.update(
    {
        "font.size": 20,  # Base font size
        "axes.titlesize": 22,  # Font size for the title
        "axes.labelsize": 20,  # Font size for the x and y labels
        "xtick.labelsize": 18,  # Font size for x tick labels
        "ytick.labelsize": 18,  # Font size for y tick labels
        "legend.fontsize": 18,  # Font size for the legend
        "figure.titlesize": 24,  # Font size for figure title
    }
)

# Define custom colors for Micro and Macro
custom_colors = {"Micro": "#1b9e77", "Macro": "#7570b3"}

# Plotting the box plots with custom colors
plt.figure(figsize=(6, 6), dpi=100)
sns.boxplot(
    x="Precision Type", y="Precision Score", data=precision_data, palette=custom_colors
)

# Set labels and axis limits
plt.ylabel("Validation Accuracy")
plt.xlabel("")
plt.ylim(0.5, 1)  # Set y-axis limits

# Set y-axis ticks every 0.05
plt.yticks(np.arange(0.5, 1.05, 0.05))

# Add grid lines for each y tick
plt.grid(True, axis="y", linestyle="--", linewidth=0.7)

# Add vertical grid lines for Micro and Macro ticks on x-axis
plt.grid(True, axis="x", linestyle="--", linewidth=0.7)

# Create custom legend using patches with the same colors as box plots
micro_patch = mpatches.Patch(color=custom_colors["Micro"], label="Micro-averaged")
macro_patch = mpatches.Patch(color=custom_colors["Macro"], label="Macro-averaged")

# Add the custom legend to the plot
plt.legend(handles=[micro_patch, macro_patch], loc="upper right")

# Adjust layout and display plot
plt.tight_layout()
plt.show()
# -

# Use the best estimator from grid search
best_rf = grid_search.best_estimator_
best_rf

# Plot confusion matrix of the test set

# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

# Assuming y_test and y_pred are already defined

# Compute the confusion matrix
conf_matrix_fr = confusion_matrix(y_test, y_pred)

# Plotting parameters
x_labels = ["0-15%", "16-30%", ">30%"]  # Replace with your actual class names
y_labels = ["0-15%", "16-30%", ">30%"]  # Replace with your actual class names
cmap = LinearSegmentedColormap.from_list("custom_green", ["white", "#1b9e77"], N=256)

# Create the confusion matrix plot
plt.figure(figsize=(5, 5))
heatmap = sns.heatmap(
    conf_matrix_fr,
    annot=False,
    fmt="d",
    cmap=cmap,
    cbar=True,
    vmin=0,
    vmax=np.max(conf_matrix_fr),
)

cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)

# Annotate the heatmap with text
for i, row in enumerate(conf_matrix_fr):
    for j, value in enumerate(row):
        color = "white" if value > np.max(conf_matrix_fr) / 2 else "black"
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
plt.xlabel("Actual Class", fontsize=24)
plt.ylabel("Predicted Class", fontsize=24)
plt.xticks([0.5 + i for i in range(len(x_labels))], x_labels, fontsize=24, rotation=45)
plt.yticks([0.5 + i for i in range(len(y_labels))], y_labels, fontsize=24, rotation=45)

# Adjust layout and display
plt.tight_layout()
plt.show()

# +
import joblib

# Save the best model
# joblib.dump(best_rf, path_to_data + "best_models/best_fr_classifier.pkl")
# -

# # Tillage Classification

# Save test set for area-based accuracy assessment
def save_test_df(X_test, y_test, y_pred, best_model, config_num):
    y_pred = best_model.predict(X_test)
    X_test_y_test_pred = X_test.copy()
    X_test_y_test_pred["y_pred"] = y_pred
    X_test_y_test_pred["y_test"] = y_test
    X_test_y_test_pred = X_test_y_test_pred[["y_test", "y_pred"]]
    X_test_y_test_pred.to_csv(
        path_to_data + f"accuracy_assessment_data/X_test_y_test_pred_{config_num}.csv"
    )


# Cross-validate baseline model

# +
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
)

import joblib

scaler = StandardScaler()

# Apply PCA
x_imagery = lsat_data.loc[:, "B_S0_p0":]
x_imagery_scaled = scaler.fit_transform(x_imagery)

pca = PCA(n_components=0.7)
x_imagery_pca = pca.fit_transform(x_imagery_scaled)

# Save the fitted scaler
joblib.dump(scaler, path_to_data + "best_models/tillage_scaler_model.pkl")
# Save the fitted PCA object
joblib.dump(pca, path_to_data + "best_models/tillage_pca_model.pkl")

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

# Define the parameter grid
param_grid = {
    "n_estimators": [50, 100, 300],  # Number of trees in the forest
    "max_features": ["log2", "sqrt"],  # Number of features to consider at every split
    "max_depth": [10, 20, 45],  # Maximum number of levels in each decision tree
    "min_samples_split": [
        2,
        5,
        10,
    ],  # Minimum number of samples required to split a node
    "min_samples_leaf": [
        1,
        2,
        4,
    ],  # Minimum number of samples required at each leaf node
    "bootstrap": [True, False],  # Method of selecting samples for training each tree
}

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Define multiple scoring metrics
scoring = {"precision_micro": "precision_micro", "precision_macro": "precision_macro"}

# Initialize GridSearchCV with 3-fold cross-validation and multiple scoring metrics
grid_search_1 = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring=scoring,
    refit="precision_macro",  # Choose which metric to optimize for best estimator
    return_train_score=True,
)

# Fit the model
grid_search_1.fit(X_train_nocrop, y_train)

# Use the best estimator from grid search
best_model_1 = grid_search_1.best_estimator_

# Predict on the test set
y_pred = best_model_1.predict(X_test_nocrop)

# Predict on the training set
y_train_pred = best_model_1.predict(X_train_nocrop)

print("Train set accuracy: ", accuracy_score(y_train, y_train_pred))
print("Test set accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# -

X_train_nocrop

X_train


X_test_nocrop

# Plot validation scores for base-line model

# +
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Extract cross-validation results
cv_results_1 = grid_search_1.cv_results_

# Initialize lists to hold all micro and macro precision scores
micro_precisions = []
macro_precisions = []

# Number of CV folds
n_splits = grid_search_1.cv

# Extract precision scores for each fold and parameter combination
# GridSearchCV stores split scores as split0_test_<scorer>, split1_test_<scorer>, etc.
for i in range(grid_search_1.cv):
    split_micro = cv_results_1[f"split{i}_test_precision_micro"]
    split_macro = cv_results_1[f"split{i}_test_precision_macro"]
    micro_precisions.extend(split_micro)
    macro_precisions.extend(split_macro)

# Create a DataFrame for plotting
precision_data_1 = pd.DataFrame(
    {
        "Precision Type": ["Micro"] * len(micro_precisions)
        + ["Macro"] * len(macro_precisions),
        "Precision Score": micro_precisions + macro_precisions,
    }
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Set global font sizes
plt.rcParams.update(
    {
        "font.size": 20,  # Base font size
        "axes.titlesize": 22,  # Font size for the title
        "axes.labelsize": 20,  # Font size for the x and y labels
        "xtick.labelsize": 18,  # Font size for x tick labels
        "ytick.labelsize": 18,  # Font size for y tick labels
        "legend.fontsize": 18,  # Font size for the legend
        "figure.titlesize": 24,  # Font size for figure title
    }
)

# Define custom colors for Micro and Macro
custom_colors = {"Micro": "#1b9e77", "Macro": "#7570b3"}

# Plotting the box plots with custom colors
plt.figure(figsize=(6, 6), dpi=100)
sns.boxplot(
    x="Precision Type", y="Precision Score", data=precision_data_1, palette=custom_colors
)

# Set labels and axis limits
plt.ylabel("Validation Accuracy")
plt.xlabel("")
plt.ylim(0.5, 1)  # Set y-axis limits

# Set y-axis ticks every 0.05
plt.yticks(np.arange(0.5, 1.05, 0.05))

# Add grid lines for each y tick
plt.grid(True, axis="y", linestyle="--", linewidth=0.7)

# Add vertical grid lines for Micro and Macro ticks on x-axis
plt.grid(True, axis="x", linestyle="--", linewidth=0.7)

# Create custom legend using patches with the same colors as box plots
micro_patch = mpatches.Patch(color=custom_colors["Micro"], label="Micro-averaged")
macro_patch = mpatches.Patch(color=custom_colors["Macro"], label="Macro-averaged")

# Add the custom legend to the plot
plt.legend(handles=[micro_patch, macro_patch], loc="upper right")

# Adjust layout and display plot
plt.tight_layout()
plt.show()
# -

# Plot confusion matrix of the test set

# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

# Assuming y_test and y_pred are already defined

# Compute the confusion matrix
conf_matrix_1 = confusion_matrix(y_test, y_pred)

# Plotting parameters
x_labels = ["CT", "MT", "NT"]  # Replace with your actual class names
y_labels = ["CT", "MT", "NT"]  # Replace with your actual class names
cmap = LinearSegmentedColormap.from_list("custom_green", ["white", "#1b9e77"], N=256)

# Create the confusion matrix plot
plt.figure(figsize=(5, 5))
heatmap = sns.heatmap(
    conf_matrix_1,
    annot=False,
    fmt="d",
    cmap=cmap,
    cbar=True,
    vmin=0,
    vmax=np.max(conf_matrix_1),
)

cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)

# Annotate the heatmap with text
for i, row in enumerate(conf_matrix_1):
    for j, value in enumerate(row):
        color = "white" if value > np.max(conf_matrix_1) / 2 else "black"
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
plt.xlabel("Actual Class", fontsize=24)
plt.ylabel("Predicted Class", fontsize=24)
plt.xticks([0.5 + i for i in range(len(x_labels))], x_labels, fontsize=24, rotation=45)
plt.yticks([0.5 + i for i in range(len(y_labels))], y_labels, fontsize=24, rotation=45)

# Adjust layout and display
plt.tight_layout()
plt.show()
# -

save_test_df(X_test_nocrop, y_test, y_pred, best_model_1, 1)


# ## Train configuration 2

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

    # Define micro and macro scoring metrics
    scoring = {"precision_micro": "precision_micro", "precision_macro": "precision_macro"}

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=3,
        scoring=scoring,
        verbose=2,
        refit="precision_macro",
        return_train_score=True,
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


def plot_val_scores_by_a(grid_search, param_grid, X_test, y_test):
    # Extract cross-validation results
    cv_results = grid_search.cv_results_

    # Get the `a` values and the corresponding validation accuracies
    a_values = param_grid["a"]
    accuracies_by_a = {a: [] for a in a_values}

    # Collect accuracies for each `a` value from cv_results
    for i, a in enumerate(cv_results["param_a"]):
        accuracies_by_a[a].append(cv_results["mean_test_macro_accuracy"][i])

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
    "bootstrap": [True, False],
}

grid_search_2 = train_model(
    X_train, y_train, X_test, y_test, 3, param_grid, CustomWeightedRF()
)

# -

# Plot macro and micro validation scores

# +
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Extract cross-validation results
cv_results_2 = grid_search_2.cv_results_

# Initialize lists to hold all micro and macro precision scores
micro_precisions = []
macro_precisions = []

# Number of CV folds
n_splits = grid_search.cv

# Extract precision scores for each fold and parameter combination
# GridSearchCV stores split scores as split0_test_<scorer>, split1_test_<scorer>, etc.
for i in range(grid_search.cv):
    split_micro = cv_results_2[f"split{i}_test_precision_micro"]
    split_macro = cv_results_2[f"split{i}_test_precision_macro"]
    micro_precisions.extend(split_micro)
    macro_precisions.extend(split_macro)

# Create a DataFrame for plotting
precision_data_2 = pd.DataFrame(
    {
        "Precision Type": ["Micro"] * len(micro_precisions)
        + ["Macro"] * len(macro_precisions),
        "Precision Score": micro_precisions + macro_precisions,
    }
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Set global font sizes
plt.rcParams.update(
    {
        "font.size": 20,  # Base font size
        "axes.titlesize": 22,  # Font size for the title
        "axes.labelsize": 20,  # Font size for the x and y labels
        "xtick.labelsize": 18,  # Font size for x tick labels
        "ytick.labelsize": 18,  # Font size for y tick labels
        "legend.fontsize": 18,  # Font size for the legend
        "figure.titlesize": 24,  # Font size for figure title
    }
)

# Define custom colors for Micro and Macro
custom_colors = {"Micro": "#1b9e77", "Macro": "#7570b3"}

# Plotting the box plots with custom colors
plt.figure(figsize=(6, 6), dpi=100)
sns.boxplot(
    x="Precision Type", y="Precision Score", data=precision_data_2, palette=custom_colors
)

# Set labels and axis limits
plt.ylabel("Validation Accuracy")
plt.xlabel("")
plt.ylim(0.5, 1)  # Set y-axis limits

# Set y-axis ticks every 0.05
plt.yticks(np.arange(0.5, 1.05, 0.05))

# Add grid lines for each y tick
plt.grid(True, axis="y", linestyle="--", linewidth=0.7)

# Add vertical grid lines for Micro and Macro ticks on x-axis
plt.grid(True, axis="x", linestyle="--", linewidth=0.7)

# Create custom legend using patches with the same colors as box plots
micro_patch = mpatches.Patch(color=custom_colors["Micro"], label="Micro-averaged")
macro_patch = mpatches.Patch(color=custom_colors["Macro"], label="Macro-averaged")

# Add the custom legend to the plot
plt.legend(handles=[micro_patch, macro_patch], loc="upper right")

# Adjust layout and display plot
plt.tight_layout()
plt.show()
# -

# Plot confusion matrix of config 2

# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap


# Use the best estimator from grid search
best_model_2 = grid_search_2.best_estimator_

# Predict on the test set
y_pred = best_model_2.predict(X_test)
# Compute the confusion matrix
conf_matrix_2 = confusion_matrix(y_test, y_pred)

# Plotting parameters
x_labels = ["CT", "MT", "NT"]  # Replace with your actual class names
y_labels = ["CT", "MT", "NT"]  # Replace with your actual class names
cmap = LinearSegmentedColormap.from_list("custom_green", ["white", "#1b9e77"], N=256)

# Create the confusion matrix plot
plt.figure(figsize=(5, 5))
heatmap = sns.heatmap(
    conf_matrix_2,
    annot=False,
    fmt="d",
    cmap=cmap,
    cbar=True,
    vmin=0,
    vmax=np.max(conf_matrix_2),
)

cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)

# Annotate the heatmap with text
for i, row in enumerate(conf_matrix_2):
    for j, value in enumerate(row):
        color = "white" if value > np.max(conf_matrix_2) / 2 else "black"
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
plt.xlabel("Actual Class", fontsize=24)
plt.ylabel("Predicted Class", fontsize=24)
plt.xticks([0.5 + i for i in range(len(x_labels))], x_labels, fontsize=24, rotation=45)
plt.yticks([0.5 + i for i in range(len(y_labels))], y_labels, fontsize=24, rotation=45)

# Adjust layout and display
plt.tight_layout()
plt.show()
# -

# Plot a ~ validation scores

# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def plot_val_scores_by_a(grid_search, param_grid):
    # Extract cross-validation results
    cv_results = grid_search.cv_results_

    # Get the `a` values and the corresponding validation accuracies
    a_values = param_grid["a"]
    accuracies_by_a = {a: [] for a in a_values}
    # Collect accuracies for each `a` value from cv_results
    for i, a in enumerate(cv_results["param_a"]):
        accuracies_by_a[a].append(cv_results["mean_test_precision_macro"][i])

    mean_scores_by_a = {a: np.mean(accs) for a, accs in accuracies_by_a.items()}
    max_a = max(mean_scores_by_a, key=mean_scores_by_a.get)
    # Filter `a` for a = 0 and the a with best macro-averaged producer's accuracy
    a_filtered = [0]
    a_filtered.append(max_a)

    # Prepare data for seaborn boxplot
    box_data = []
    for a in a_filtered:
        for score in accuracies_by_a[a]:
            box_data.append(
                {
                    "Weighting exponent (a)": a,
                    "Macro_averaged producer's accuracy": score,
                }
            )

    df_box_data = pd.DataFrame(box_data)

    # Set global font sizes
    plt.rcParams.update(
        {
            "font.size": 14,  # Base font size
            "axes.titlesize": 14,  # Font size for the title
            "axes.labelsize": 14,  # Font size for the x and y labels
            "xtick.labelsize": 14,  # Font size for x tick labels
            "ytick.labelsize": 14,  # Font size for y tick labels
            "legend.fontsize": 14,  # Font size for the legend
            "figure.titlesize": 14,  # Font size for figure title
        }
    )
    # Define custom colors for Micro and Macro
    custom_colors = {0.0: "#f26419", max_a: "#2f4858"}

    plt.figure(figsize=(4, 6), dpi=100)

    # Use seaborn to plot the boxplot with the DataFrame
    sns.boxplot(
        x="Weighting exponent (a)",
        y="Macro_averaged producer's accuracy",
        data=df_box_data,
        palette=custom_colors,
    )

    # Set labels and axis limits
    plt.ylabel("Macro-averaged producer's accuracy")
    plt.xlabel("")
    # plt.ylim(0.6, 1)  # Set y-axis limits

    # Set y-axis ticks every 0.05
    plt.yticks(np.arange(0.6, 1.05, 0.05))

    # Add grid lines for each y tick
    plt.grid(True, axis="y", linestyle="--", linewidth=0.7)

    # Add vertical grid lines for Micro and Macro ticks on x-axis
    plt.grid(True, axis="x", linestyle="--", linewidth=0.7)

    # Create custom legend using patches with the same colors as box plots
    micro_patch = mpatches.Patch(color=custom_colors[0.0], label="a = 0")
    macro_patch = mpatches.Patch(color=custom_colors[max_a], label="Best a")

    # Add the custom legend to the plot
    plt.legend(handles=[micro_patch, macro_patch], loc="upper right")

    plt.show()


plot_val_scores_by_a(grid_search_2, param_grid)
# -

save_test_df(X_test, y_test, y_pred, best_model_2, 2)

# Save best model in config 2

# +
# import joblib

# best_model = grid_search.best_estimator_
# # Save the best model
# joblib.dump(best_model, path_to_data + "best_models/best_tillage_classifier.pkl")
# -

# ## Train configuration 3

# +
s1_data = pd.concat([s1_data["pointID"], s1_data.loc[:, "VV_S0_p5":]], axis=1)
lsat_s1 = pd.merge(lsat_data, s1_data, how="left", on="pointID")
lsat_s1 = lsat_s1.set_index("pointID")


scaler = StandardScaler()

# Apply PCA
x_imagery = lsat_s1.loc[:, "B_S0_p0":]
x_imagery = x_imagery.fillna(x_imagery.mean())
x_imagery_scaled = scaler.fit_transform(x_imagery)

pca = PCA(n_components=0.6)
x_imagery_pca = pca.fit_transform(x_imagery_scaled)

x_imagery_pca = pd.DataFrame(x_imagery_pca)
x_imagery_pca.set_index(x_imagery.index, inplace=True)

X = pd.concat(
    [
        lsat_s1["cdl_cropType"],
        lsat_s1["min_NDTI_S0"],
        lsat_s1["min_NDTI_S1"],
        lsat_s1["ResidueCov"],
        x_imagery_pca,
    ],
    axis=1,
)

to_replace = {"0-15%": 1, "16-30%": 2, ">30%": 3}
X["ResidueCov"] = X["ResidueCov"].replace(to_replace)
X
X.columns = X.columns.astype(str)
y = lsat_s1["Tillage"]

groups = X_valid["cdl_cropType"]

X_train, X_test = X.loc[X_train.index], X.loc[X_test.index]
y_train, y_test = y.loc[X_train.index], y.loc[X_test.index]

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
    "bootstrap": [True, False],
}

grid_search_3 = train_model(
    X_train, y_train, X_test, y_test, 3, param_grid, CustomWeightedRF()
)

# +
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Extract cross-validation results
cv_results_3 = grid_search_3.cv_results_

# Initialize lists to hold all micro and macro precision scores
micro_precisions = []
macro_precisions = []

# Number of CV folds
n_splits = grid_search_3.cv

# Extract precision scores for each fold and parameter combination
# GridSearchCV stores split scores as split0_test_<scorer>, split1_test_<scorer>, etc.
for i in range(grid_search_3.cv):
    split_micro = cv_results_3[f"split{i}_test_precision_micro"]
    split_macro = cv_results_3[f"split{i}_test_precision_macro"]
    micro_precisions.extend(split_micro)
    macro_precisions.extend(split_macro)

# Create a DataFrame for plotting
precision_data_3 = pd.DataFrame(
    {
        "Precision Type": ["Micro"] * len(micro_precisions)
        + ["Macro"] * len(macro_precisions),
        "Precision Score": micro_precisions + macro_precisions,
    }
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Set global font sizes
plt.rcParams.update(
    {
        "font.size": 20,  # Base font size
        "axes.titlesize": 22,  # Font size for the title
        "axes.labelsize": 20,  # Font size for the x and y labels
        "xtick.labelsize": 18,  # Font size for x tick labels
        "ytick.labelsize": 18,  # Font size for y tick labels
        "legend.fontsize": 18,  # Font size for the legend
        "figure.titlesize": 24,  # Font size for figure title
    }
)

# Define custom colors for Micro and Macro
custom_colors = {"Micro": "#1b9e77", "Macro": "#7570b3"}

# Plotting the box plots with custom colors
plt.figure(figsize=(6, 6), dpi=100)
sns.boxplot(
    x="Precision Type", y="Precision Score", data=precision_data_3, palette=custom_colors
)

# Set labels and axis limits
plt.ylabel("Validation Accuracy")
plt.xlabel("")
plt.ylim(0.5, 1)  # Set y-axis limits

# Set y-axis ticks every 0.05
plt.yticks(np.arange(0.5, 1.05, 0.05))

# Add grid lines for each y tick
plt.grid(True, axis="y", linestyle="--", linewidth=0.7)

# Add vertical grid lines for Micro and Macro ticks on x-axis
plt.grid(True, axis="x", linestyle="--", linewidth=0.7)

# Create custom legend using patches with the same colors as box plots
micro_patch = mpatches.Patch(color=custom_colors["Micro"], label="Micro-averaged")
macro_patch = mpatches.Patch(color=custom_colors["Macro"], label="Macro-averaged")

# Add the custom legend to the plot
plt.legend(handles=[micro_patch, macro_patch], loc="upper right")

# Adjust layout and display plot
plt.tight_layout()
plt.show()

# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap


# Use the best estimator from grid search
best_model_3 = grid_search_3.best_estimator_

# Predict on the test set
y_pred = best_model_3.predict(X_test)
# Compute the confusion matrix
conf_matrix_3 = confusion_matrix(y_test, y_pred)

# Plotting parameters
x_labels = ["CT", "MT", "NT"]  # Replace with your actual class names
y_labels = ["CT", "MT", "NT"]  # Replace with your actual class names
cmap = LinearSegmentedColormap.from_list("custom_green", ["white", "#1b9e77"], N=256)

# Create the confusion matrix plot
plt.figure(figsize=(5, 5))
heatmap = sns.heatmap(
    conf_matrix_3,
    annot=False,
    fmt="d",
    cmap=cmap,
    cbar=True,
    vmin=0,
    vmax=np.max(conf_matrix_3),
)

cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)

# Annotate the heatmap with text
for i, row in enumerate(conf_matrix_3):
    for j, value in enumerate(row):
        color = "white" if value > np.max(conf_matrix_3) / 2 else "black"
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
plt.xlabel("Actual Class", fontsize=24)
plt.ylabel("Predicted Class", fontsize=24)
plt.xticks([0.5 + i for i in range(len(x_labels))], x_labels, fontsize=24, rotation=45)
plt.yticks([0.5 + i for i in range(len(y_labels))], y_labels, fontsize=24, rotation=45)

# Adjust layout and display
plt.tight_layout()
plt.show()
# -

save_test_df(X_test, y_test, y_pred, best_model_3, 3)

# # Plots

path_to_plots = ('/Users/aminnorouzi/Library/CloudStorage/'
                 'OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/'
                 'Projects/Tillage_Mapping/plots/')

# Count-based confusion matrix for the three configurations

# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

conf_matrices = [conf_matrix_1, conf_matrix_2, conf_matrix_3]

# Plotting parameters
x_labels = ["CT", "MT", "NT"]
y_labels = ["CT", "MT", "NT"]

# Custom colormap
colors = ["#aaaaaa", "#bbbbbb", "#cccccc", "#dddddd", "#eeeeee"][::-1]
cmap = LinearSegmentedColormap.from_list("custom_green", colors, N=256)

# Create a 1-row, 3-column grid of subplots
fig, axes = plt.subplots(1, 3, figsize=(8, 1.7))

# Labels for the subplots
subplot_labels = ["a)", "b)", "c)"]

for idx, conf_matrix in enumerate(conf_matrices):
    heatmap = sns.heatmap(
        conf_matrix,
        annot=False,
        fmt="d",
        cmap=cmap,
        cbar=True,
        ax=axes[idx],
        vmin=0,
        vmax=65,
    )
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    for i, row in enumerate(conf_matrix):
        for j, value in enumerate(row):
            color = "white" if value > 20 else "black"
            axes[idx].text(
                j + 0.5,
                i + 0.5,
                str(value),
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )

    axes[idx].set_xlabel("Actual Class", fontsize=10)
    axes[idx].set_xticks([0.5 + i for i in range(len(x_labels))])
    axes[idx].set_xticklabels(x_labels, fontsize=10, rotation=0)

    # Set y-labels only for the first subplot
    if idx == 0:
        axes[idx].set_ylabel("Predicted Class", fontsize=10)
    axes[idx].set_yticks([0.5 + i for i in range(len(y_labels))])
    axes[idx].set_yticklabels(y_labels, fontsize=10, rotation=0)

    # Add subplot label (a), b), c)) at the upper right corner
    axes[idx].text(
        0,
        -0.2,  # Adjust the position based on the plot size
        subplot_labels[idx],
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="top",
    )

plt.subplots_adjust(wspace=0.4)
plt.savefig(
    path_to_plots + "cross_validation_test/count_based_cms.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300,
)
plt.show()

# +
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator


def convert_crossval_df_to_array(val_df):
    return {
        precision_type: np.array(group["Precision Score"].values)
        for precision_type, group in val_df.groupby("Precision Type")
    }


crossval_data_1 = convert_crossval_df_to_array(precision_data_1)
crossval_data_2 = convert_crossval_df_to_array(precision_data_2)
crossval_data_3 = convert_crossval_df_to_array(precision_data_3)

configs_data = {"1": crossval_data_1, "2": crossval_data_2, "3": crossval_data_3}


# Custom legend handles
micro_patch = mpatches.Patch(color="#1b9e77", label="Overall")
macro_patch = mpatches.Patch(color="#7570b3", label="Macro prod. acc.")

# Create figure and axes manually
fig = plt.figure(figsize=(15, 6))
axs = [fig.add_subplot(1, 3, i + 1) for i in range(3)]

# Custom x-axis labels for each scenario
custom_xlabels = [
    "",
    "",
    "",
]

# Plotting function with custom x-ticks and minor ticks
for i, (scenario_number, scenario) in enumerate(configs_data.items()):
    ax = axs[i]
    micro_data = [scenario["Micro"]]
    macro_data = [scenario["Macro"]]

    # Plotting micro and macro data
    ax.boxplot(
        micro_data,
        positions=[1],
        widths=0.35,
        patch_artist=True,
        meanline=True,
        showmeans=True,
        boxprops=dict(facecolor="#1b9e77", linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
        medianprops=dict(linewidth=0.07),
        showfliers=False,
    )
    ax.boxplot(
        macro_data,
        positions=[2],
        widths=0.35,
        patch_artist=True,
        meanline=True,
        showmeans=True,
        boxprops=dict(facecolor="#7570b3", linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
        medianprops=dict(linewidth=0.07),
        showfliers=False,
    )

    # Enable grid before plotting data to ensure grid lines are below data elements
    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=1,
        color="gray",
        alpha=0.7,
        zorder=1,
    )

    # Setting custom x-axis labels for micro and macro data
    ax.set_xticks([1, 2])  # Set tick positions at the center of each box plot
    ax.set_xticklabels(
        ["Overall", "Macro prod. acc."], fontsize=36
    )  # Label each position accordingly

    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(axis="both", which="both", labelsize=16)

    # Set y-axis limit and ticks for all subplots
    ax.set_ylim(0.5, 1.05)
    ax.set_yticks(np.arange(0.5, 1, 0.05))

    if i == 0:  # Add legend only to the first subplot to avoid repetition
        ax.legend(handles=[micro_patch, macro_patch], fontsize=16, loc="upper left")

for ax in axs:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Arial")  # Adjust the font as needed
        label.set_fontsize(14)

# Common Y-axis label and adjustments
fig.text(
    -0.01,
    0.5,
    "Validation accuracy",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=16,
)


plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.savefig(
    path_to_plots + "cross_validation_test/count_based_cms.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300,
)

plt.show()
