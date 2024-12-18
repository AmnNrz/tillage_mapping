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

# # Read data

# +
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
)

# Specify path to data and plots
path_to_data = ("/Users/aminnorouzi/Library/CloudStorage/"
                "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
                "Ph.D/Projects/Tillage_Mapping/Data/")

path_to_plots = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
    "Projects/Tillage_Mapping/plots/"
)

# Landsat data
lsat_data = pd.read_csv(path_to_data + "/to_share/training_data/lsat_data.csv")

# Sentinel 1 data
s1_data = pd.read_csv(path_to_data + "/to_share/training_data/s1_data.csv")

# CDL data
cdl_data = pd.read_csv(path_to_data + "/to_share/training_data/cdl_df.csv")

# Encode crop type
to_replace = {"Grain": 1, "Legume": 2, "Canola": 3}
lsat_data["cdl_cropType"] = lsat_data["cdl_cropType"].replace(to_replace)
lsat_data = lsat_data.set_index("pointID")
# -

# # Plot tillage vs crop confusion matrix

# +
# Map tillage classes and residue cover ranges for confusion matrix
tillage_mapping = {
    "ConventionalTill": "CT",
    "MinimumTill": "MT",
    "NoTill-DirectSeed": "NT",
}

residue_mapping = {
    1: "Grain",  # Adjust according to your residue classes
    2: "Legume",
    3: "Canola",
}

# Apply the mappings to your dataframe
tillage = lsat_data["Tillage"].map(tillage_mapping)
crop_res = lsat_data["cdl_cropType"].map(residue_mapping)

# Create the confusion matrix using crosstab
confusion_matrix = pd.crosstab(tillage, crop_res, margins=False)
confusion_matrix = confusion_matrix[["Grain", "Legume", "Canola"]]
confusion_matrix = np.array(confusion_matrix)

# Plotting parameters
x_labels = ["Grain", "Legume", "Canola"]  
y_labels = ["CT", "MT", "NT"] 


# Custom colormap
colors = ["#e0e2db", "#d2d4c8", "#b8bdb5", "#889696", "#5f7470"]

cmap = LinearSegmentedColormap.from_list("custom_green", colors, N=256)

# Create the confusion matrix plot
plt.figure(figsize=(3.4, 3))
heatmap = sns.heatmap(
    confusion_matrix,
    annot=False,
    fmt="d",
    cmap=cmap,
    cbar=True,
    vmin=0,
    vmax=np.max(confusion_matrix),
    linewidths=0.6,
    linecolor="black",
    square=True,
)

cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)

# Annotate the heatmap with text
for i, row in enumerate(confusion_matrix):
    for j, value in enumerate(row):
        color = "white" if value > np.max(confusion_matrix) / 2 else "black"
        plt.text(
            j + 0.5,
            i + 0.5,
            str(value),
            ha="center",
            va="center",
            color=color,
            fontsize=14,
        )

# Set axis labels and ticks
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Actual Class", fontsize=12)
plt.xticks([0.5 + i for i in range(len(x_labels))], x_labels, fontsize=12, rotation=45)
plt.yticks([0.5 + i for i in range(len(y_labels))], y_labels, fontsize=12, rotation=45)


# Save the plot with the same settings as the first one
plt.savefig(
    path_to_plots + "fig_imbalance_data.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300,
)

# Adjust layout and display
plt.tight_layout()
plt.show()


# -

# # Fr classification

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


# +
# Define the parameter grids
param_grid_fr = {
    "n_estimators": [5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200],
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40],
    "min_samples_split": [3, 4, 5, 6, 7, 8, 9, 10, 20],
    "bootstrap": [True, False],
}

param_grid_baseline = {
    "n_estimators": [5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200],
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40],
    "min_samples_split": [3, 4, 5, 6, 7, 8, 9, 10, 20],
    "bootstrap": [True, False],

}
param_grid_config_2 = {
    "n_estimators": [5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200],
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40],
    "min_samples_split": [3, 4, 5, 6, 7, 8, 9, 10, 20],
    "bootstrap": [True],
    "a": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 4],
    "sample_weight_mode": ["mixed", "tillage"]
}
param_grid_config_3 = {
    "n_estimators": [5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200],
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40],
    "min_samples_split": [3, 4, 5, 6, 7, 8, 9, 10, 20],
    "bootstrap": [True, False],
    "a": [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 4],
    "sample_weight_mode": ["mixed", "tillage"]
}
# -

# # Train fr classifier

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
from sklearn.model_selection import StratifiedKFold


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
scoring = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
}

# Initialize GridSearchCV with 3-fold cross-validation and multiple scoring metrics
grid_search_fr = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_fr,
    cv=3,
    n_jobs=-1,
    verbose=0,
    scoring=scoring,
    refit="f1_macro",  # Choose which metric to optimize for best estimator
    return_train_score=True,
)

# Fit the model
grid_search_fr.fit(X_train, y_train)

# Use the best estimator from grid search
best_rf = grid_search_fr.best_estimator_

# Predict on the test set
y_pred = best_rf.predict(X_test)

# Predict on the training set
y_train_pred = best_rf.predict(X_train)


print("Train set accuracy: ", accuracy_score(y_train, y_train_pred))
print("Test set accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

save_test_df(X_test, y_test, y_pred, best_rf, 'fr')
# -

# Best hyper-parameters for fr classifier
best_hyperparameters = grid_search_fr.best_params_
print("Best hyperparameters:", best_hyperparameters)

# Plot overall and f1-macro accuracy across all combinations of hyper-parameters

# +
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import numpy as np

# Extract cross-validation results
cv_results = grid_search_fr.cv_results_

# Initialize lists to hold all accuracy and f1_macro scores
accuracies = []
f1_macros = []

# Number of CV folds
n_splits = grid_search_fr.cv

# Extract accuracy and f1_macro scores for each fold and parameter combination
# GridSearchCV stores split scores as split0_test_<scorer>, split1_test_<scorer>, etc.
for i in range(grid_search_fr.cv):
    split_accuracy = cv_results[f"split{i}_test_accuracy"]
    split_f1_macro = cv_results[f"split{i}_test_f1_macro"]
    accuracies.extend(split_accuracy)
    f1_macros.extend(split_f1_macro)

# Create a DataFrame for plotting
score_data = pd.DataFrame(
    {
        "Score Type": ["Overall accuracy"] * len(accuracies)
        + ["F1 Macro"] * len(f1_macros),
        "Score": accuracies + f1_macros,
    }
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Set global font sizes to match the first figure
plt.rcParams.update(
    {
        "font.size": 16,  # Base font size
        "axes.titlesize": 16,  # Font size for the title
        "axes.labelsize": 16,  # Font size for the x and y labels
        "xtick.labelsize": 16,  # Font size for x tick labels
        "ytick.labelsize": 16,  # Font size for y tick labels
        "legend.fontsize": 16,  # Font size for the legend
        "figure.titlesize": 16,  # Font size for figure title
    }
)

# Define custom colors for Accuracy and F1 Macro to match the first figure
custom_colors = {"Overall accuracy": "#4A6274", "F1 Macro": "#94ACBF"}

# Create the figure to match the size of the first plot
plt.figure(figsize=(5, 6), dpi=300)

# Plotting the box plots with custom colors and adjusting the width of the boxes
sns.boxplot(
    x="Score Type",
    y="Score",
    data=score_data,
    palette=custom_colors,
    width=0.35,  # Match the width of the box plots from the first figure
    linewidth=2,  # Match the linewidth
    showfliers=True
)

# Set labels and axis limits
plt.ylabel("Validation Score")
plt.xlabel("")
plt.ylim(0.5, 1)  # Set y-axis limits

# Set y-axis ticks every 0.05 to match the first plot
plt.yticks(np.arange(0.5, 1.05, 0.05))

# Add grid lines for each y tick (matching the style of the first plot)
plt.grid(True, axis="y", linestyle="-", linewidth=1, color="gray", alpha=0.7)

# Add vertical grid lines for Accuracy and F1 Macro ticks on x-axis (optional)
plt.grid(True, axis="x", linestyle="--", linewidth=0.7)

# Create custom legend using patches with the same colors as box plots
accuracy_patch = mpatches.Patch(
    color=custom_colors["Overall accuracy"], label="Overall accuracy"
)
f1_macro_patch = mpatches.Patch(color=custom_colors["F1 Macro"], label="F1 Macro")

# Add the custom legend to the plot
plt.legend(handles=[accuracy_patch, f1_macro_patch], loc="upper right")

# Save the plot with the same settings as the first one
# plt.savefig(
#     path_to_plots + "cross_validation_test/fig_fr_val_new.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     dpi=300,
# )

plt.savefig(
    path_to_plots + "fig_fr_val_new.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300,
)

# Adjust layout and display plot
plt.tight_layout()
plt.show()
# -

# Train and test results across other groups

# +
## **********
## Check sub-classes for misclassified instances of the test set
## **********

y_test_ = pd.DataFrame(y_test).copy()
y_test_["pointID"] = y_test_.index
y_test_["y_pred"] = y_pred
y_test_["residue_type"] = X_test["cdl_cropType"]
y_test_["WhereInRan"] = lsat_data.loc[y_test_.index.to_list()][
    "WhereInRan"
]  # Directly add WhereInRan values


# Filter for misclassified instances
wrongs = y_test_.loc[y_test_["ResidueCov"] != y_test_["y_pred"]]

# Create a list to store each row as a dictionary
rows = []

# Populate the rows list with misclassified data
for _, row in wrongs.iterrows():
    rows.append(
        {
            "pointID": row["pointID"],
            "ResidueCov": row["ResidueCov"],
            "y_pred": row["y_pred"],
            "residue_type": row["residue_type"],
            "WhereInRan": row["WhereInRan"],
        }
    )

# Convert the list of rows into a DataFrame
df_wrong = pd.DataFrame(rows)

# Replace residue_type and fr_pred values with the mappings
mapping_residuetype_dict = {1: "Grain", 2: "Legume", 3: "Canola"}
df_wrong["residue_type"] = df_wrong["residue_type"].replace(mapping_residuetype_dict)

mapping_fr_dict = {
    1: "0-15%",  # Adjust according to your residue classes
    2: "16-30%",
    3: ">30%",
}
df_wrong["ResidueCov"] = df_wrong["ResidueCov"].replace(mapping_fr_dict)
df_wrong["y_pred"] = df_wrong["y_pred"].replace(mapping_fr_dict)

df_wrong.to_csv(
    path_to_data + "wrong_df_fr.csv",
    index=False,
)
# -

# Plot confusion matrix of fr classification on the test set

# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

# Compute the confusion matrix
conf_matrix_fr = confusion_matrix(y_test, y_pred)

# Plotting parameters
x_labels = ["0-15%", "16-30%", ">30%"]  # Replace with your actual class names
y_labels = ["0-15%", "16-30%", ">30%"]  # Replace with your actual class names


# Custom colormap
colors = ["#e0e2db", "#d2d4c8", "#b8bdb5", "#889696", "#5f7470"]

cmap = LinearSegmentedColormap.from_list("custom_green", colors, N=256)

# Create the confusion matrix plot
plt.figure(figsize=(3.4,3))
heatmap = sns.heatmap(
    conf_matrix_fr,
    annot=False,
    fmt="d",
    cmap=cmap,
    cbar=True,
    vmin=0,
    vmax=np.max(conf_matrix_fr),
    linewidths=0.6,
    linecolor="black",
    square=True,
)

cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)

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
            fontsize=14,
        )

# Set axis labels and ticks
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Actual Class", fontsize=12)
plt.xticks([0.5 + i for i in range(len(x_labels))], x_labels, fontsize=12, rotation=45)
plt.yticks([0.5 + i for i in range(len(y_labels))], y_labels, fontsize=12, rotation=45)


# Save the plot with the same settings as the first one
plt.savefig(
    path_to_plots + "fig_fr_test_CM_new.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300,
)

# Adjust layout and display
plt.tight_layout()
plt.show()

# +
# Overall Accuracy
correct_predictions = np.trace(conf_matrix_fr)
total_instances = np.sum(conf_matrix_fr)
overall_accuracy = correct_predictions / total_instances
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

# Users' Accuracy (Precision)
predicted_per_class = np.sum(conf_matrix_fr, axis=0)
true_positives = np.diag(conf_matrix_fr)
users_accuracy = true_positives / predicted_per_class
users_accuracy = np.where(predicted_per_class != 0, users_accuracy, 0)

print("\nUsers' Accuracy (Precision) for each class:")
for idx, ua in enumerate(users_accuracy):
    print(f"Class {idx}: {ua:.4f}")

# Producers' Accuracy (Recall)
actual_per_class = np.sum(conf_matrix_fr, axis=1)
producers_accuracy = true_positives / actual_per_class
producers_accuracy = np.where(actual_per_class != 0, producers_accuracy, 0)

print("\nProducers' Accuracy (Recall) for each class:")
for idx, pa in enumerate(producers_accuracy):
    print(f"Class {idx}: {pa:.4f}")

# +
import joblib

# Save the best model
joblib.dump(best_rf, path_to_data + "best_models/best_fr_classifier.pkl")
# -

# # Tillage Classification

# Cross-validate baseline model

# Predict fr for ground-truth data

X_ = X.copy()
X_['fr_pred'] = best_rf.predict(X_)
X_ = X_[['fr_pred']]
lsat_data = pd.concat(
    [lsat_data.iloc[:, :4], X_["fr_pred"], lsat_data.iloc[:, 4:]], axis=1
)
lsat_data

# Prepare data

# +
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
        lsat_data["fr_pred"],
        x_imagery_pca,
    ],
    axis=1,
)

to_replace = {"0-15%": 1, "16-30%": 2, ">30%": 3}
X["fr_pred"] = X["fr_pred"].replace(to_replace)
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

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Define multiple scoring metrics
scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
# -

# Train baseline

# +
# Initialize GridSearchCV with 3-fold cross-validation and multiple scoring metrics
grid_search_1 = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_baseline,
    cv=3,
    n_jobs=-1,
    verbose=0,
    scoring=scoring,
    refit="f1_macro",  # Choose which metric to optimize for best estimator
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

save_test_df(X_test_nocrop, y_test, y_pred, best_model_1, "1")
# -

best_hyperparameters = grid_search_1.best_params_
print("Best hyperparameters:", best_hyperparameters)

# Save test results for baseline

# +
X_test_to_save = X_test[["cdl_cropType", "fr_pred"]].copy()
X_test_to_save['pointID'] = X_test_to_save.index
X_test_to_save["WhereInRan"] = lsat_data.loc[X_test_to_save.index.to_list()][
    "WhereInRan"
] 
X_test_to_save["fr_act"] = lsat_data.loc[X_test_to_save.index.to_list()]["ResidueCov"]
X_test_to_save["tillage_act"] = lsat_data.loc[X_test_to_save.index.to_list()][
    "Tillage"
]
X_test_to_save["tillage_pred"] = y_pred

X_test_to_save.to_csv(
    path_to_data + "test_df_baseline.csv",
    index=False,
)

# -

# Plot validation scores for base-line model

# +
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Extract cross-validation results
cv_results_1 = grid_search_1.cv_results_

# Initialize lists to hold all micro and macro precision scores
accuracies = []
f1_macros = []

# Number of CV folds
n_splits = grid_search_1.cv

# Extract precision scores for each fold and parameter combination
# GridSearchCV stores split scores as split0_test_<scorer>, split1_test_<scorer>, etc.
for i in range(grid_search_1.cv):
    split_accuracy = cv_results_1[f"split{i}_test_accuracy"]
    split_f1_macro = cv_results_1[f"split{i}_test_f1_macro"]
    accuracies.extend(split_accuracy)
    f1_macros.extend(split_f1_macro)

# Create a DataFrame for plotting
score_data_1 = pd.DataFrame(
    {
        "Score Type": ["Accuracy"] * len(accuracies) + ["F1 Macro"] * len(f1_macros),
        "Score": accuracies + f1_macros,
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
custom_colors = {"Accuracy": "#1b9e77", "F1 Macro": "#7570b3"}

# Plotting the box plots with custom colors
plt.figure(figsize=(6, 6), dpi=100)
sns.boxplot(x="Score Type", y="Score", data=score_data_1, palette=custom_colors)

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
accuracy_patch = mpatches.Patch(color=custom_colors["Accuracy"], label="Accuracy")
f1_macro_patch = mpatches.Patch(color=custom_colors["F1 Macro"], label="F1 Macro")

# Add the custom legend to the plot
plt.legend(handles=[accuracy_patch, f1_macro_patch], loc="upper right")
# Save the plot with the same settings as the first one
plt.savefig(
    path_to_plots + "fig_crosval_baseline.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300,
)
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
plt.xlabel("Predicted Class", fontsize=24)
plt.ylabel("Actual Class", fontsize=24)
plt.xticks([0.5 + i for i in range(len(x_labels))], x_labels, fontsize=24, rotation=45)
plt.yticks([0.5 + i for i in range(len(y_labels))], y_labels, fontsize=24, rotation=45)

# Save the plot with the same settings as the first one
plt.savefig(
    path_to_plots + "fig_test_baseline.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300,
)

# Adjust layout and display
plt.tight_layout()
plt.show()
# -

# Save test results 
save_test_df(X_test_nocrop, y_test, y_pred, best_model_1, 1)

# +
## **********
## Check sub-classes for misclassified instances of the test set
## **********

y_test_ = pd.DataFrame(y_test).copy()
y_test_["pointID"] = y_test_.index
y_test_["y_pred"] = y_pred
y_test_["residue_type"] = X_test["cdl_cropType"]
y_test_["fr_pred"] = X_test["fr_pred"]
y_test_["fr_act"] = lsat_data.loc[y_test_.index.to_list()]["ResidueCov"]
y_test_["WhereInRan"] = lsat_data.loc[y_test_.index.to_list()][
    "WhereInRan"
]  

# Directly add WhereInRan values


# Filter for misclassified instances
wrongs = y_test_.loc[y_test_["Tillage"] != y_test_["y_pred"]]

# Create a list to store each row as a dictionary
rows = []

# Populate the rows list with misclassified data
for _, row in wrongs.iterrows():
    rows.append(
        {
            "pointID": row["pointID"],
            "Tillage": row["Tillage"],
            "y_pred": row["y_pred"],
            "residue_type": row["residue_type"],
            "fr_pred": row["fr_pred"],
            "fr_act": row["fr_act"],
            "WhereInRan": row["WhereInRan"],
        }
    )

# Convert the list of rows into a DataFrame
df_wrong = pd.DataFrame(rows)

# Replace residue_type and fr_pred values with the mappings
mapping_residuetype_dict = {1: "Grain", 2: "Legume", 3: "Canola"}
df_wrong["residue_type"] = df_wrong["residue_type"].replace(mapping_residuetype_dict)

mapping_fr_dict = {
    1: "0-15%",  # Adjust according to your residue classes
    2: "16-30%",
    3: ">30%",
}
df_wrong["fr_pred"] = df_wrong["fr_pred"].replace(mapping_fr_dict)

df_wrong.to_csv(
    path_to_data + "wrong_df_baseline.csv",
    index=False,
)

# +
## **********
## Check sub-classes all instances of the test set
## **********

y_test_ = pd.DataFrame(y_test).copy()
y_test_["pointID"] = y_test_.index
y_test_["y_pred"] = y_pred
y_test_["residue_type"] = X_test["cdl_cropType"]
y_test_["fr_pred"] = X_test["fr_pred"]
y_test_["fr_act"] = lsat_data.loc[y_test_.index.to_list()]["ResidueCov"]
y_test_["WhereInRan"] = lsat_data.loc[y_test_.index.to_list()][
    "WhereInRan"
]  # Directly add WhereInRan values

# Filter for misclassified instances
# y_test_ = y_test_.loc[y_test_["Tillage"] != y_test_["y_pred"]]

# Create a list to store each row as a dictionary
rows = []

# Populate the rows list with misclassified data
for _, row in y_test_.iterrows():
    rows.append(
        {
            "pointID": row["pointID"],
            "Tillage": row["Tillage"],
            "y_pred": row["y_pred"],
            "residue_type": row["residue_type"],
            "fr_pred": row["fr_pred"],
            "fr_act": row["fr_act"],
            "WhereInRan": row["WhereInRan"],
        }
    )

# Convert the list of rows into a DataFrame
test_df = pd.DataFrame(rows)

# Replace residue_type and fr_pred values with the mappings
mapping_residuetype_dict = {1: "Grain", 2: "Legume", 3: "Canola"}
test_df["residue_type"] = test_df["residue_type"].replace(mapping_residuetype_dict)

mapping_fr_dict = {
    1: "0-15%",  # Adjust according to your residue classes
    2: "16-30%",
    3: ">30%",
}
test_df["fr_pred"] = test_df["fr_pred"].replace(mapping_fr_dict)

test_df.to_csv(
    path_to_data + "test_df_baseline.csv",
    index=False,
)


# -

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
        sample_weight_mode="mixed", 
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
        self.sample_weight_mode = sample_weight_mode  # Store the mode
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, **kwargs
        )

    def fit(self, X, y, **kwargs):
        # Calculate the target weights based on 'a'
        target_weights_dict = calculate_custom_weights(y, self.a)
        target_weights = np.array([target_weights_dict[sample] for sample in y])

        # Calculate sample weights by combining target and feature weights
        if self.sample_weight_mode == 'mixed':
            # Mixed sample weights: target_weights * feature_weights
            X_mod = X.copy()
            feature_cols = ["cdl_cropType"]
            feature_weights = np.zeros(X_mod.shape[0])
            for col in feature_cols:
                feature_weights_dict = calculate_custom_weights(
                    X_mod[col].values, self.a
                )
                feature_weights += X_mod[col].map(feature_weights_dict).values

            sample_weights = target_weights * feature_weights
        elif self.sample_weight_mode == 'tillage':
            # Tillage sample weights: target_weights
            X_mod = X.copy()
            sample_weights = target_weights
        else:
            raise ValueError("Invalid sample_weight_mode. Choose 'mixed' or 'tillage'.")

        # Fit the RandomForestClassifier with the computed weights and modified dataset
        self.rf.fit(X_mod, y, sample_weight=sample_weights)
        # Set the classes_ attribute
        self.classes_ = self.rf.classes_

        return self

    def predict(self, X, **kwargs):
        X_mod = X.copy()
        return self.rf.predict(X_mod)

    def predict_proba(self, X, **kwargs):
        X_mod = X.copy()
        return self.rf.predict_proba(X_mod)

    @property
    def feature_importances_(self):
        return self.rf.feature_importances_

def train_model(X_train, y_train, cv, param_grid, classifier):

    # Define micro and macro scoring metrics
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=3,
        scoring=scoring,
        verbose=0,
        refit="f1_macro",
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


# -

# Train model configuration 2
#

# +
X_train_config_2 = X_train
X_test_config_2 = X_test

grid_search_2 = train_model(
    X_train, y_train, 3, param_grid_config_2, CustomWeightedRF()
)
# -

# Plot macro and micro validation scores

# +
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import numpy as np

# Extract cross-validation results
cv_results_2 = grid_search_2.cv_results_

# Filter out entries with param_sample_weight_mode equal to "tillage"
filtered_indices = [
    i
    for i, param in enumerate(cv_results_2["param_sample_weight_mode"])
    if param != "tillage"
]

# Initialize lists to hold all micro and macro precision scores
accuracies = []
f1_macros = []

# Extract precision scores for each fold and parameter combination, excluding "tillage" entries
for i in range(grid_search_2.cv):
    split_accuracy = cv_results_2[f"split{i}_test_accuracy"][filtered_indices]
    split_f1_macro = cv_results_2[f"split{i}_test_f1_macro"][filtered_indices]
    accuracies.extend(split_accuracy)
    f1_macros.extend(split_f1_macro)

# Create a DataFrame for plotting
score_data_2 = pd.DataFrame(
    {
        "Score Type": ["Accuracy"] * len(accuracies) + ["F1 Macro"] * len(f1_macros),
        "Score": accuracies + f1_macros,
    }
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Set global font sizes
plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.titlesize": 24,
    }
)

# Define custom colors for Micro and Macro
custom_colors = {"Accuracy": "#1b9e77", "F1 Macro": "#7570b3"}

# Plotting the box plots with custom colors
plt.figure(figsize=(6, 6), dpi=100)
sns.boxplot(x="Score Type", y="Score", data=score_data_2, palette=custom_colors)

# Set labels and axis limits
plt.ylabel("Validation Accuracy")
plt.xlabel("")
plt.ylim(0.5, 1)

# Set y-axis ticks every 0.05
plt.yticks(np.arange(0.5, 1.05, 0.05))

# Add grid lines for each y tick
plt.grid(True, axis="y", linestyle="--", linewidth=0.7)

# Add vertical grid lines for Micro and Macro ticks on x-axis
plt.grid(True, axis="x", linestyle="--", linewidth=0.7)

# Create custom legend using patches with the same colors as box plots
accuracy_patch = mpatches.Patch(color=custom_colors["Accuracy"], label="Accuracy")
f1_macro_patch = mpatches.Patch(color=custom_colors["F1 Macro"], label="F1 Macro")

# Add the custom legend to the plot
plt.legend(handles=[accuracy_patch, f1_macro_patch], loc="upper right")

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
plt.xlabel("Predicted Class", fontsize=24)
plt.ylabel("Actual Class", fontsize=24)
plt.xticks([0.5 + i for i in range(len(x_labels))], x_labels, fontsize=24, rotation=45)
plt.yticks([0.5 + i for i in range(len(y_labels))], y_labels, fontsize=24, rotation=45)

# Adjust layout and display
plt.tight_layout()
plt.show()
# -

# Best hyper-paramters for model configuration 2
best_hyperparameters = grid_search_2.best_params_
print("Best hyperparameters:", best_hyperparameters)

# +
## **********
## Check sub-classes all instances of the test set
## **********
y_pred_train = best_model_2.predict(X_train)
y_train_ = pd.DataFrame(y_train).copy()
y_train_["pointID"] = y_train_.index
y_train_["y_pred_train"] = y_pred_train
y_train_["residue_type"] = X_train["cdl_cropType"]
y_train_["fr_pred"] = X_train["fr_pred"]
y_train_["fr_act"] = lsat_data.loc[y_train_.index.to_list()]["ResidueCov"]
y_train_["WhereInRan"] = lsat_data.loc[y_train_.index.to_list()][
    "WhereInRan"
]  # Directly add WhereInRan values


# Create a list to store each row as a dictionary
rows = []

# Populate the rows list with misclassified data
for _, row in y_train_.iterrows():
    rows.append(
        {
            "pointID": row["pointID"],
            "Tillage": row["Tillage"],
            "y_pred_train": row["y_pred_train"],
            "residue_type": row["residue_type"],
            "fr_pred": row["fr_pred"],
            "fr_act": row["fr_act"],
            "WhereInRan": row["WhereInRan"],
        }
    )

# Convert the list of rows into a DataFrame
train_df = pd.DataFrame(rows)

# Replace residue_type and fr_pred values with the mappings
mapping_residuetype_dict = {1: "Grain", 2: "Legume", 3: "Canola"}
train_df["residue_type"] = train_df["residue_type"].replace(mapping_residuetype_dict)

mapping_fr_dict = {
    1: "0-15%",  # Adjust according to your residue classes
    2: "16-30%",
    3: ">30%",
}
train_df["fr_pred"] = train_df["fr_pred"].replace(mapping_fr_dict)

train_df.to_csv(path_to_data + "train_df_config_2.csv",
    index=False,
)

train_df.groupby(["Tillage", "y_pred_train", "residue_type", "fr_pred", "fr_act"]).size().reset_index(name="Count")

# +
## **********
## Check sub-classes all instances of the test set
## **********

y_test_ = pd.DataFrame(y_test).copy()
y_test_["pointID"] = y_test_.index
y_test_["y_pred"] = y_pred
y_test_["residue_type"] = X_test["cdl_cropType"]
y_test_["fr_pred"] = X_test["fr_pred"]
y_test_["fr_act"] = lsat_data.loc[y_test_.index.to_list()]["ResidueCov"]
y_test_["WhereInRan"] = lsat_data.loc[y_test_.index.to_list()][
    "WhereInRan"
]  # Directly add WhereInRan values

# Create a list to store each row as a dictionary
rows = []

# Populate the rows list with misclassified data
for _, row in y_test_.iterrows():
    rows.append(
        {
            "pointID": row["pointID"],
            "Tillage": row["Tillage"],
            "y_pred": row["y_pred"],
            "residue_type": row["residue_type"],
            "fr_pred": row["fr_pred"],
            "fr_act": row["fr_act"],
            "WhereInRan": row["WhereInRan"],
        }
    )

# Convert the list of rows into a DataFrame
test_df = pd.DataFrame(rows)

# Replace residue_type and fr_pred values with the mappings
mapping_residuetype_dict = {1: "Grain", 2: "Legume", 3: "Canola"}
test_df["residue_type"] = test_df["residue_type"].replace(mapping_residuetype_dict)

mapping_fr_dict = {
    1: "0-15%",  # Adjust according to your residue classes
    2: "16-30%",
    3: ">30%",
}
test_df["fr_pred"] = test_df["fr_pred"].replace(mapping_fr_dict)

test_df.to_csv(
    path_to_data + "test_df_config_2.csv",
    index=False,
)

test_df.groupby(["Tillage", "y_pred", "residue_type", "fr_pred", "fr_act"]).size().reset_index(name="Count")
# -

# Plot a ~ validation scores

# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Extract cross-validation results
cv_results_2 = grid_search_2.cv_results_

a_values = param_grid_config_2["a"]

# Filter out entries with param_sample_weight_mode equal to "tillage"
filtered_indices_for_tillage = [
    i
    for i, param in enumerate(cv_results_2["param_sample_weight_mode"])
    if param == "tillage"
]

filtered_indices_for_mixed = [
    i
    for i, param in enumerate(cv_results_2["param_sample_weight_mode"])
    if param != "tillage"
]

f1_macros_by_a_tillage = {a: [] for a in a_values}
f1_macros_by_a_mixed = {a: [] for a in a_values}
# Extract precision scores for each fold and parameter combination, excluding "tillage" entries
for i, a in enumerate(cv_results_2["param_a"]):
    for j in range(grid_search_2.cv):
        if i in filtered_indices_for_tillage:
            f1_macros_by_a_tillage[a].append(cv_results_2[f"split{j}_test_f1_macro"][i])
        if i in filtered_indices_for_mixed:
            f1_macros_by_a_mixed[a].append(cv_results_2[f"split{j}_test_f1_macro"][i])

mean_scores_by_a_mixed = {a: np.mean(accs) for a, accs in f1_macros_by_a_mixed.items()}
max_a_mixed = max(mean_scores_by_a_mixed, key=mean_scores_by_a_mixed.get)

mean_scores_by_a_tillage = {a: np.mean(accs) for a, accs in f1_macros_by_a_tillage.items()}
max_a_tillage = max(mean_scores_by_a_tillage, key=mean_scores_by_a_tillage.get)


# Prepare data for seaborn boxplot
box_data = []

a_filtered_mixed_0 = 0
for score in f1_macros_by_a_mixed[a_filtered_mixed_0]:
    box_data.append(
        {
            "Weighting exponent (a)": f"a = 0 mixed",
            "f1_Macro": score,
        }
    )

for score in f1_macros_by_a_mixed[max_a_mixed]:
    box_data.append(
        {
            "Weighting exponent (a)": f"best a mixed",
            "f1_Macro": score,
        }
    )


# Filter `a` for best f1_macro of sample_weight_mode 'tillage'

for score in f1_macros_by_a_tillage[max_a_tillage]:
    box_data.append(
        {
            "Weighting exponent (a)": f"best a tillage",
            "f1_Macro": score,
        }
    )


# Add the Base-line boxplot from grid_search_1
cv_results_1 = grid_search_1.cv_results_
f1_macros_baseline = []
for i in range(grid_search_1.cv):
    split_f1_macro_baseline = cv_results_1[f"split{i}_test_f1_macro"]
    f1_macros_baseline.extend(split_f1_macro_baseline)

# Baseline scores
for score in f1_macros_baseline:
    box_data.append(
        {
            "Weighting exponent (a)": "Baseline",
            "f1_Macro": score,
        }
    )

# Convert to DataFrame
df_box_data = pd.DataFrame(box_data)

# Ensure custom order for categories
df_box_data["Weighting exponent (a)"] = pd.Categorical(
    df_box_data["Weighting exponent (a)"],
    categories=[
        "Baseline",
        f"a = 0 mixed",
        f"best a tillage",
        f"best a mixed"
        
    ],
    ordered=True,
)

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

# Define custom colors
custom_colors = {
    "Baseline": "#dd6e42",
    f"a = 0 mixed": "#e8dab2",
    f"best a tillage": "#88c0d0",
    f"best a mixed": "#4f6d7a"
}

plt.figure(figsize=(4, 6), dpi=100)

# Plot boxplot with seaborn
sns.boxplot(
    x="Weighting exponent (a)",
    y="f1_Macro",
    data=df_box_data,
    palette=custom_colors,
    # showmeans = True
)

# Set labels and axis limits
plt.ylabel("F1 Macro Score")
plt.xlabel("Weighting exponent (a)")
# Manually set custom x-axis ticks
plt.xticks(
    ticks=[0, 1, 2, 3],  # Adjusted tick positions
    labels=["BL", "0 (T*F)", f"{max_a_tillage} (T)", f"{max_a_mixed} (T*F)"],
    rotation=90,
)

# Add a vertical line to separate Baseline from other categories
plt.axvline(
    x=0.5, color="gray", linestyle="-", linewidth=1
)  # Adjust position as needed
plt.xlabel("Weighting exponent (a)", labelpad=10, loc="right")  # Move right

plt.yticks(np.arange(0.3, 1.05, 0.05))
plt.ylim(0.5, 0.9)  # Set y-axis limits

# Add grid lines for each y tick
plt.grid(True, axis="y", linestyle="--", linewidth=0.7)

plt.savefig(
    path_to_plots + "fig_a_accuracies_new.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300,
)

plt.show()
# -

# Save test results of configuration 2
save_test_df(X_test, y_test, y_pred, best_model_2, 2)

# Save best model in config 2

# Save the best model configuration 2
joblib.dump(best_model_2, path_to_data + "best_models/best_tillage_classifier_config2.pkl")

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

pca = PCA(n_components=0.7)
x_imagery_pca = pca.fit_transform(x_imagery_scaled)

x_imagery_pca = pd.DataFrame(x_imagery_pca)
x_imagery_pca.set_index(x_imagery.index, inplace=True)

X = pd.concat(
    [
        lsat_s1["cdl_cropType"],
        lsat_s1["fr_pred"],
        x_imagery_pca,
    ],
    axis=1,
)

to_replace = {"0-15%": 1, "16-30%": 2, ">30%": 3}
X["fr_pred"] = X["fr_pred"].replace(to_replace)
X
X.columns = X.columns.astype(str)
y = lsat_s1["Tillage"]

groups = X_valid["cdl_cropType"]

X_train_config_3, X_test_config_3 = X.loc[X_train.index], X.loc[X_test.index]
y_train, y_test = y.loc[X_train.index], y.loc[X_test.index]

# +

grid_search_3 = train_model(
    X_train_config_3, y_train, 3, param_grid_config_3, CustomWeightedRF()
)

# +
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Extract cross-validation results
cv_results_3 = grid_search_3.cv_results_

# Initialize lists to hold all micro and macro precision scores
accuracies = []
f1_macros = []

# Number of CV folds
n_splits = grid_search_3.cv

# Extract precision scores for each fold and parameter combination
# GridSearchCV stores split scores as split0_test_<scorer>, split1_test_<scorer>, etc.
for i in range(grid_search_3.cv):
    split_accuracy = cv_results_3[f"split{i}_test_accuracy"]
    split_f1_macro = cv_results_3[f"split{i}_test_f1_macro"]
    accuracies.extend(split_accuracy)
    f1_macros.extend(split_f1_macro)

# Create a DataFrame for plotting
score_data_3 = pd.DataFrame(
    {
        "Score Type": ["Accuracy"] * len(accuracies) + ["F1 Macro"] * len(f1_macros),
        "Score": accuracies + f1_macros,
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
custom_colors = {"Accuracy": "#1b9e77", "F1 Macro": "#7570b3"}

# Plotting the box plots with custom colors
plt.figure(figsize=(6, 6), dpi=100)
sns.boxplot(x="Score Type", y="Score", data=score_data_3, palette=custom_colors)

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
accuracy_patch = mpatches.Patch(color=custom_colors["Accuracy"], label="Accuracy")
f1_macro_patch = mpatches.Patch(color=custom_colors["F1 Macro"], label="F1 Macro")

# Add the custom legend to the plot
plt.legend(handles=[accuracy_patch, f1_macro_patch], loc="upper right")

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
y_pred = best_model_3.predict(X_train_config_3)
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
plt.xlabel("Predicted Class", fontsize=24)
plt.ylabel("Actual Class", fontsize=24)
plt.xticks([0.5 + i for i in range(len(x_labels))], x_labels, fontsize=24, rotation=45)
plt.yticks([0.5 + i for i in range(len(y_labels))], y_labels, fontsize=24, rotation=45)

# Adjust layout and display
plt.tight_layout()
plt.show()
# -

# Best hyper-parameters of model configuration 3
best_hyperparameters = grid_search_3.best_params_
print("Best hyperparameters:", best_hyperparameters)

# +
## **********
## Check sub-classes all instances of the train set
## **********
y_pred_train = best_model_3.predict(X_train)
y_train_ = pd.DataFrame(y_train).copy()
y_train_["pointID"] = y_train_.index
y_train_["y_pred_train"] = y_pred_train
y_train_["residue_type"] = X_train["cdl_cropType"]
y_train_["fr_pred"] = X_train["fr_pred"]
y_train_["fr_act"] = lsat_data.loc[y_train_.index.to_list()]["ResidueCov"]
y_train_["WhereInRan"] = lsat_data.loc[y_train_.index.to_list()][
    "WhereInRan"
]  # Directly add WhereInRan values
# Filter for misclassified instances
# y_train_ = y_train_.loc[y_train_["Tillage"] != y_train_["y_pred_train"]]

# Create a list to store each row as a dictionary
rows = []

# Populate the rows list with misclassified data
for _, row in y_train_.iterrows():
    rows.append(
        {
            "pointID": row["pointID"],
            "Tillage": row["Tillage"],
            "y_pred_train": row["y_pred_train"],
            "residue_type": row["residue_type"],
            "fr_pred": row["fr_pred"],
            "fr_act": row["fr_act"],
            "WhereInRan": row["WhereInRan"],
        }
    )

# Convert the list of rows into a DataFrame
train_df = pd.DataFrame(rows)

# Replace residue_type and fr_pred values with the mappings
mapping_residuetype_dict = {1: "Grain", 2: "Legume", 3: "Canola"}
train_df["residue_type"] = train_df["residue_type"].replace(mapping_residuetype_dict)

mapping_fr_dict = {
    1: "0-15%",  # Adjust according to your residue classes
    2: "16-30%",
    3: ">30%",
}
train_df["fr_pred"] = train_df["fr_pred"].replace(mapping_fr_dict)

train_df.to_csv(path_to_data + "train_df_config_3.csv",
    index=False,
)

train_df.groupby(["Tillage", "y_pred_train", "residue_type", "fr_pred", "fr_act"]).size().reset_index(name="Count")

# +
## **********
## Check sub-classes all instances of the test set
## **********

y_test_ = pd.DataFrame(y_test).copy()
y_test_["pointID"] = y_test_.index
y_test_["y_pred"] = y_pred
y_test_["residue_type"] = X_test["cdl_cropType"]
y_test_["fr_pred"] = X_test["fr_pred"]
y_test_["fr_act"] = lsat_data.loc[y_test_.index.to_list()]["ResidueCov"]
y_test_["WhereInRan"] = lsat_data.loc[y_test_.index.to_list()][
    "WhereInRan"
]  # Directly add WhereInRan values

# Filter for misclassified instances
# y_test_ = y_test_.loc[y_test_["Tillage"] != y_test_["y_pred"]]

# Create a list to store each row as a dictionary
rows = []

# Populate the rows list with misclassified data
for _, row in y_test_.iterrows():
    rows.append(
        {
            "pointID": row["pointID"],
            "Tillage": row["Tillage"],
            "y_pred": row["y_pred"],
            "residue_type": row["residue_type"],
            "fr_pred": row["fr_pred"],
            "fr_act": row["fr_act"],
            "WhereInRan": row["WhereInRan"],
        }
    )

# Convert the list of rows into a DataFrame
test_df = pd.DataFrame(rows)

# Replace residue_type and fr_pred values with the mappings
mapping_residuetype_dict = {1: "Grain", 2: "Legume", 3: "Canola"}
test_df["residue_type"] = test_df["residue_type"].replace(mapping_residuetype_dict)

mapping_fr_dict = {
    1: "0-15%",  # Adjust according to your residue classes
    2: "16-30%",
    3: ">30%",
}
test_df["fr_pred"] = test_df["fr_pred"].replace(mapping_fr_dict)

test_df.to_csv(
    path_to_data + "test_df_config_3.csv",
    index=False,
)

test_df.groupby(["Tillage", "y_pred", "residue_type", "fr_pred", "fr_act"]).size().reset_index(name="Count")
# -

save_test_df(X_test_config_3, y_test, y_pred, best_model_3, 3)

# # Plots

# Count-based confusion matrix for the three configurations

# +
from sklearn.metrics import confusion_matrix
import geopandas as gpd

test_df_1 = pd.read_csv(
    path_to_data + "accuracy_assessment_data/X_test_y_test_pred_1.csv"
)
test_df_2 = pd.read_csv(
    path_to_data + "accuracy_assessment_data/X_test_y_test_pred_2.csv"
)
test_df_3 = pd.read_csv(
    path_to_data + "accuracy_assessment_data/X_test_y_test_pred_3.csv"
)



# Mapping the string labels to the desired labels
label_mapping = {
    "ConventionalTill": "CT",
    "MinimumTill": "MT",
    "NoTill-DirectSeed": "NT",
}


def create_ordered_count_based_confusion_matrix(df, x_labels, y_labels, label_mapping):
    # Map the y_test and y_pred columns to the new labels
    df["y_test_mapped"] = df["y_test"].map(label_mapping)
    df["y_pred_mapped"] = df["y_pred"].map(label_mapping)

    # Create a confusion matrix based on the counts of mapped y_test and y_pred
    matrix = confusion_matrix(df["y_test_mapped"], df["y_pred_mapped"], labels=y_labels)

    # Convert to a DataFrame for better readability
    confusion_df = pd.DataFrame(matrix, index=y_labels, columns=x_labels)

    return confusion_df


# Define the labels for x and y
x_labels = ["CT", "MT", "NT"]
y_labels = ["CT", "MT", "NT"]

# Generate count-based confusion matrices for each dataframe
conf_matrix_1 = create_ordered_count_based_confusion_matrix(
    test_df_1, x_labels, y_labels, label_mapping
)
conf_matrix_2 = create_ordered_count_based_confusion_matrix(
    test_df_2, x_labels, y_labels, label_mapping
)
conf_matrix_3 = create_ordered_count_based_confusion_matrix(
    test_df_3, x_labels, y_labels, label_mapping
)

conf_matrix_1 = np.array(conf_matrix_1)
conf_matrix_2 = np.array(conf_matrix_2)
conf_matrix_3 = np.array(conf_matrix_3)

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
colors = ["#e0e2db", "#d2d4c8", "#b8bdb5", "#889696", "#5f7470"]

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
        vmax=55,
        linewidths=0.6,
        linecolor="black",
        square=True,
    )
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    for i, row in enumerate(conf_matrix):
        for j, value in enumerate(row):
            color = "white" if value > 30 else "black"
            axes[idx].text(
                j + 0.5,
                i + 0.5,
                str(value),
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )

    axes[idx].set_xlabel("Predicted Class", fontsize=10)
    axes[idx].set_xticks([0.5 + i for i in range(len(x_labels))])
    axes[idx].set_xticklabels(x_labels, fontsize=10, rotation=0)

    # Set y-labels only for the first subplot
    if idx == 0:
        axes[idx].set_ylabel("Actual Class", fontsize=10)
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
    path_to_plots + "fig_Tillage_test_CM_new.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300,
)
plt.show()

# +
import numpy as np


def calculate_accuracies(conf_matrix):
    # Overall Accuracy
    overall_acc = np.trace(conf_matrix) / np.sum(conf_matrix)

    # User's Accuracy (Precision)
    user_acc = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)

    # Producer's Accuracy (Recall)
    producer_acc = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)

    return overall_acc, user_acc, producer_acc


conf_matrices = [conf_matrix_1, conf_matrix_2, conf_matrix_3]

# Calculate and display accuracies for each matrix
for i, conf_matrix in enumerate(conf_matrices):
    overall_acc, user_acc, producer_acc = calculate_accuracies(conf_matrix)
    print(f"Confusion Matrix {i+1}:")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"User's Accuracy (Precision): {user_acc}")
    print(f"Producer's Accuracy (Recall): {producer_acc}\n")
# -

# Plot area-based confusion matrices

# +
import geopandas as gpd

def create_area_based_confusion_matrix(df):
    # Drop rows where ExactAcres is NaN
    df = df.dropna(subset=["ExactAcres"])

    # Create a confusion matrix where each cell represents the sum of ExactAcres
    confusion_matrix = pd.pivot_table(
        df,
        values="ExactAcres",
        index="y_test",
        columns="y_pred",
        aggfunc=np.sum,
        fill_value=0,
    )

    # Calculate total area
    total_area = confusion_matrix.values.sum()

    # Normalize each cell by dividing by the total area
    confusion_matrix = confusion_matrix / total_area

    return confusion_matrix


test_df_1 = pd.read_csv(
    path_to_data + "accuracy_assessment_data/X_test_y_test_pred_1.csv"
)
test_df_2 = pd.read_csv(
    path_to_data + "accuracy_assessment_data/X_test_y_test_pred_2.csv"
)
test_df_3 = pd.read_csv(
    path_to_data + "accuracy_assessment_data/X_test_y_test_pred_3.csv"
)

shpfile_2122 = gpd.read_file(
    path_to_data + "GIS_Data/final_shpfiles/final_shp_2122.shp"
)
shpfile_2223 = gpd.read_file(
    path_to_data + "GIS_Data/final_shpfiles/final_shp_2223.shp"
)
ground_merged = pd.concat([shpfile_2122, shpfile_2223])


test_df_1 = pd.merge(
    test_df_1, ground_merged[["pointID", "ExactAcres"]], on="pointID", how="left"
)
test_df_2 = pd.merge(
    test_df_2, ground_merged[["pointID", "ExactAcres"]], on="pointID", how="left"
)
test_df_3 = pd.merge(
    test_df_3, ground_merged[["pointID", "ExactAcres"]], on="pointID", how="left"
)
# Generate area-based confusion matrices for each dataframe
conf_matrix_1_area = np.array(create_area_based_confusion_matrix(test_df_1))
conf_matrix_2_area = np.array(create_area_based_confusion_matrix(test_df_2))
conf_matrix_3_area = np.array(create_area_based_confusion_matrix(test_df_3))

conf_matrix_1_area = np.round(conf_matrix_1_area, 2)
conf_matrix_2_area = np.round(conf_matrix_2_area, 2)
conf_matrix_3_area = np.round(conf_matrix_3_area, 2)

conf_matrices_area = [conf_matrix_1_area, conf_matrix_2_area, conf_matrix_3_area]

# Plotting parameters
x_labels = ["CT", "MT", "NT"]
y_labels = ["CT", "MT", "NT"]

colors = ["#e0e2db", "#d2d4c8", "#b8bdb5", "#889696", "#5f7470"]

cmap = LinearSegmentedColormap.from_list("custom_green", colors, N=256)

# Create a 1-row, 3-column grid of subplots
fig, axes = plt.subplots(1, 3, figsize=(8, 1.7))

# Labels for the subplots
subplot_labels = ["a)", "b)", "c)"]

for idx, conf_matrix in enumerate(conf_matrices_area):
    heatmap = sns.heatmap(
        conf_matrix,
        annot=False,
        fmt="d",
        cmap=cmap,
        cbar=True,
        ax=axes[idx],
        vmin=0,
        vmax=0.4,
        linewidths=0.6,
        linecolor="black",
        square=True,
    )
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    for i, row in enumerate(conf_matrix):
        for j, value in enumerate(row):
            color = "white" if value > 0.3 else "black"
            axes[idx].text(
                j + 0.5,
                i + 0.5,
                str(value),
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )

    axes[idx].set_xlabel("Predicted Class", fontsize=10)
    axes[idx].set_xticks([0.5 + i for i in range(len(x_labels))])
    axes[idx].set_xticklabels(x_labels, fontsize=10, rotation=0)

    # Set y-labels only for the first subplot
    if idx == 0:
        axes[idx].set_ylabel("Actual Class", fontsize=10)
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
    path_to_plots + "cross_validation_test/fig_Tillage_test_CM_new_areabased.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300,
)
plt.show()
# -

# Plot paneled cross-validation results

# +
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator


def convert_crossval_df_to_array(val_df):
    return {
        score_type: np.array(group["Score"].values)
        for score_type, group in val_df.groupby("Score Type")
    }


crossval_data_1 = convert_crossval_df_to_array(score_data_1)
crossval_data_2 = convert_crossval_df_to_array(score_data_2)
crossval_data_3 = convert_crossval_df_to_array(score_data_3)

configs_data = {"1": crossval_data_1, "2": crossval_data_2, "3": crossval_data_3}

# Custom legend handles
accuracy_patch = mpatches.Patch(color="#4A6274", label="Overall accuracy")
f1_macro_patch = mpatches.Patch(color="#94ACBF", label="F1 Macro")

# Create figure and axes manually
fig = plt.figure(figsize=(15, 6))
axs = [fig.add_subplot(1, 3, i + 1) for i in range(3)]

# Plotting function with custom x-ticks and minor ticks
for i, (scenario_number, scenario) in enumerate(configs_data.items()):
    ax = axs[i]
    accuracy_data = [scenario["Accuracy"]]
    f1_macro_data = [scenario["F1 Macro"]]

    # Plotting Accuracy and F1 Macro data
    ax.boxplot(
        accuracy_data,
        positions=[1],
        widths=0.35,
        patch_artist=True,
        meanline=False,
        showmeans=False,
        boxprops=dict(facecolor="#4A6274", linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
        medianprops=dict(linewidth=1, color="black"),
        showfliers=True,
    )
    ax.boxplot(
        f1_macro_data,
        positions=[2],
        widths=0.35,
        patch_artist=True,
        meanline=False,
        showmeans=False,
        boxprops=dict(facecolor="#94ACBF", linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
        medianprops=dict(linewidth=1, color="black"),
        showfliers=True,
    )

    # Enable grid
    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=1,
        color="gray",
        alpha=0.7,
        zorder=1,
    )

    # Setting custom x-axis labels for Accuracy and F1 Macro data
    ax.set_xticks([1, 2])  # Set tick positions at the center of each box plot
    ax.set_xticklabels(
        ["Overall accuracy", "F1 Macro"], fontsize=36
    )  # Label each position accordingly

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="both", which="both", labelsize=16)

    # Set y-axis limit and ticks for all subplots
    ax.set_ylim(0.5, 1)
    ax.set_yticks(np.arange(0.5, 1, 0.05))

    if i == 0:  # Add legend only to the first subplot
        ax.legend(
            handles=[accuracy_patch, f1_macro_patch], fontsize=16, loc="upper left"
        )

for ax in axs:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Arial")  # Adjust the font as needed
        label.set_fontsize(14)

# Common Y-axis label and adjustments
fig.text(
    -0.01,
    0.5,
    "Validation score",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=16,
)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.savefig(
    path_to_plots + "fig_tillage_cv.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300,
)

plt.show()


# -

# Plot crop-wise confusion matrices

# +
def plot_cropwise_cm_multiple(grid_searches, X_tests, y_test, X_with_crop_info):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    from matplotlib.colors import LinearSegmentedColormap

    # Mapping from numerical values to crop classes
    to_replace = {"Grain": 1, "Legume": 2, "Canola": 3}
    inverse_to_replace = {v: k for k, v in to_replace.items()}  # Reverse the dictionary

    # Get unique classes in cdl_cropType (numerical)
    unique_classes = X_with_crop_info["cdl_cropType"].unique()

    # Define the labels based on actual labels in y_test and y_pred
    labels = np.unique(
        np.concatenate((y_test, y_test))
    )  # Keep same labels for all tests
    labels = labels.tolist()  # Convert to list if necessary

    # Map labels to class names
    label_to_class = {0: "CT", 1: "MT", 2: "NT"}  # Adjust mapping as per your data
    class_to_label = {v: k for k, v in label_to_class.items()}

    # Use class names for axis labels
    x_labels = ["CT", "MT", "NT"]
    y_labels = ["CT", "MT", "NT"]

    # Create a mapping from class names to label indices used in y_test and y_pred
    class_indices = [class_to_label.get(cls, cls) for cls in x_labels]
    # Custom colormap
    # colors = ["#595959", "#7f7f7f", "#a5a5a5", "#cccccc", "#f2f2f2"][::-1]
    colors = ["#e0e2db", "#d2d4c8", "#b8bdb5", "#889696", "#5f7470"]

    cmap = LinearSegmentedColormap.from_list("custom_green", colors, N=256)

    # cmap = LinearSegmentedColormap.from_list(
    #     "custom_green", ["white", "#1b9e77"], N=256
    # )

    num_searches = len(grid_searches)  # Number of grid searches provided

    # Create a figure for the confusion matrices
    plt.figure(
        figsize=(2 * num_searches, 2 * len(unique_classes))
    )  # Adjusted for multiple grids

    plt.subplots_adjust(wspace=0.04)  # Reduce this value to make the columns closer

    heatmap = None
    for idx, crop_class_num in enumerate(unique_classes):
        crop_class_name = inverse_to_replace.get(crop_class_num, crop_class_num)

        for gs_idx, (grid_search, X_test) in enumerate(zip(grid_searches, X_tests)):
            # Use the best estimator from each grid search
            best_model = grid_search.best_estimator_

            # Predict on the test set
            y_pred = best_model.predict(X_test)

            # Filter test set and predictions based on each class in "cdl_cropType"
            mask = X_with_crop_info["cdl_cropType"] == crop_class_num
            y_test_filtered = y_test[mask]
            y_pred_filtered = y_pred[mask]

            # Check if y_test_filtered is empty
            if y_test_filtered.size == 0:
                conf_matrix = np.zeros((len(labels), len(labels)), dtype=int)
            else:
                conf_matrix = confusion_matrix(
                    y_test_filtered, y_pred_filtered, labels=labels
                )

            # Create a subplot for each confusion matrix
            ax = plt.subplot(
                len(unique_classes), num_searches, gs_idx + 1 + idx * num_searches
            )

            # Plot the heatmap and store the first one for the color bar
            heatmap = sns.heatmap(
                conf_matrix,
                annot=False,
                fmt="d",
                cmap=cmap,
                cbar=False,  # Disable individual colorbars
                vmin=0,
                vmax=np.max(conf_matrix),
                linewidths=0.6,
                linecolor='black',
                square=True
            )

            if idx == 0:
                # Define the list of labels for the columns
                column_labels = ["Baseline", "Configuration 2", "Configuration 3"]

                # Add this line to label each column
                ax.set_title(
                    column_labels[gs_idx], fontsize=12, weight="bold"
                )  # You can customize the label here

            # Annotate the heatmap with text
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    value = conf_matrix[i, j]
                    color = "white" if value > np.max(conf_matrix) / 2 else "black"
                    plt.text(
                        j + 0.5,
                        i + 0.5,
                        str(value),
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=12,
                    )

            # Set axis labels and ticks
            # Add x-axis labels and ticks only to the lowest row
            if idx == len(unique_classes) - 1:
                plt.xlabel("Predicted Class", fontsize=12)
                plt.xticks([0.5 + i for i in range(len(x_labels))], x_labels, fontsize=12, rotation=45)
            else:
                plt.xticks([])  # Remove x-axis ticks for other rows

            # Add y-axis labels and ticks only to the most left column
            if gs_idx == 0:
                plt.ylabel("Actual Class", fontsize=12)
                plt.yticks([0.5 + i for i in range(len(y_labels))], y_labels, fontsize=12, rotation=45)
            else:
                plt.yticks([])  # Remove y-axis ticks for other columns

            # Add crop class name on the left of the confusion matrix
            if gs_idx == 0:  # Only add the crop class label on the first column
                ax.text(
                    -1.3,
                    conf_matrix.shape[0] / 2,
                    f"{crop_class_name}",
                    va="center",
                    ha="right",
                    fontsize=12,
                    weight="bold",
                    color="black",
                    rotation=90,
                )

    plt.savefig(
        path_to_plots + "fig_crop_wise_CMs.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    # Adjust layout and display all confusion matrices
    plt.tight_layout()
    plt.show()


# List of grid searches and corresponding test datasets
grid_searches = [grid_search_1, grid_search_2, grid_search_3]
X_tests = [X_test_nocrop, X_test_config_2, X_test_config_3]

# Call the plotting function
plot_cropwise_cm_multiple(grid_searches, X_tests, y_test, X_test)
# -

# Calculate accuracy metrics based on confusion matrices

# +
import numpy as np
from sklearn.metrics import confusion_matrix


# Function to extract confusion matrices and compute accuracies
def extract_confusion_matrices_and_accuracies(
    grid_searches, X_tests, y_test, X_with_crop_info
):
    # Mapping from numerical values to crop classes
    to_replace = {"Grain": 1, "Legume": 2, "Canola": 3}
    inverse_to_replace = {v: k for k, v in to_replace.items()}

    # Get unique classes in cdl_cropType (numerical)
    unique_classes = X_with_crop_info["cdl_cropType"].unique()

    # Define the labels based on actual labels in y_test
    labels = np.unique(np.concatenate((y_test, y_test)))  # Same labels for all tests
    labels = labels.tolist()  # Convert to list if necessary

    # Dictionary to store confusion matrices and accuracies
    results = {}

    for idx, crop_class_num in enumerate(unique_classes):
        crop_class_name = inverse_to_replace.get(crop_class_num, crop_class_num)
        results[crop_class_name] = {}

        for gs_idx, (grid_search, X_test) in enumerate(zip(grid_searches, X_tests)):
            # Use the best estimator from each grid search
            best_model = grid_search.best_estimator_

            # Predict on the test set
            y_pred = best_model.predict(X_test)

            # Filter test set and predictions based on each class in "cdl_cropType"
            mask = X_with_crop_info["cdl_cropType"] == crop_class_num
            y_test_filtered = y_test[mask]
            y_pred_filtered = y_pred[mask]

            # Check if y_test_filtered is empty
            if y_test_filtered.size == 0:
                conf_matrix = np.zeros((len(labels), len(labels)), dtype=int)
            else:
                conf_matrix = confusion_matrix(
                    y_test_filtered, y_pred_filtered, labels=labels
                )

            # Store the confusion matrix
            config_name = f"Configuration {gs_idx + 1}"
            results[crop_class_name][config_name] = {
                "conf_matrix": conf_matrix,
            }

            # Calculate overall, user's (precision), and producer's (recall) accuracy
            total_correct = np.trace(conf_matrix)
            total_instances = np.sum(conf_matrix)
            overall_accuracy = total_correct / total_instances

            # User's Accuracy (Precision)
            with np.errstate(divide='ignore', invalid='ignore'):
                users_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)  # Precision
                users_accuracy = np.where(np.isnan(users_accuracy), np.nan, users_accuracy)  # Optional: Set default to 0 if needed

            # Producer's Accuracy (Recall)
            with np.errstate(divide='ignore', invalid='ignore'):
                producers_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)  # Recall
                producers_accuracy = np.where(np.isnan(producers_accuracy), np.nan, producers_accuracy)  # Optional: Set default to 0 if needed


            # Store accuracies
            results[crop_class_name][config_name]["overall_accuracy"] = overall_accuracy
            results[crop_class_name][config_name]["users_accuracy"] = users_accuracy
            results[crop_class_name][config_name][
                "producers_accuracy"
            ] = producers_accuracy

    return results


# List of grid searches and corresponding test datasets
grid_searches = [grid_search_1, grid_search_2, grid_search_3]
X_tests = [X_test_nocrop, X_test_config_2, X_test_config_3]

# Call the function
confusion_matrices_and_accuracies = extract_confusion_matrices_and_accuracies(
    grid_searches, X_tests, y_test, X_test
)

# Print the results
for crop_class, configs in confusion_matrices_and_accuracies.items():
    print(f"\n{crop_class}:")
    for config_name, data in configs.items():
        conf_matrix = data["conf_matrix"]
        overall_accuracy = data["overall_accuracy"]
        users_accuracy = data["users_accuracy"]
        producers_accuracy = data["producers_accuracy"]

        print(f"\n{config_name}:")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Overall Accuracy: {overall_accuracy:.2f}")
        print(f"User's Accuracy (Precision): {users_accuracy}")
        print(f"Producer's Accuracy (Recall): {producers_accuracy}")
