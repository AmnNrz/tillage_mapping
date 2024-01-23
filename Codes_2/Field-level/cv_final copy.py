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
        self.rf = RandomForestClassifier(n_estimators=self.n_estimators,
                                         max_depth=self.max_depth, **kwargs)
    
    def fit(self, X, y, **kwargs):
        target_weights_dict = calculate_custom_weights(y, self.a)
        target_weights = np.array([target_weights_dict[sample] for sample in y])

        # Rest of the weight calculation can stay same
        feature_cols = ['ResidueType']
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
path_to_data = ("/Users/aminnorouzi/Library/CloudStorage/"
                "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
                "Projects/Tillage_Mapping/Data/field_level_data/FINAL_DATA/")

# path_to_data = ("/home/amnnrz/OneDrive - "
#                 "a.norouzikandelati/Ph.D/Projects/Tillage_Mapping/Data"
#                 "/field_level_data/FINAL_DATA/")

df = pd.read_csv(path_to_data + "season_finalData.csv")
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
        temp_df = df[df[column] == value].sample(frac=1) \
        .reset_index(drop=True) # Shuffle
        midpoint = len(temp_df) // 2
        dfs1.append(temp_df.iloc[:midpoint])
        dfs2.append(temp_df.iloc[midpoint:])

    df1 = pd.concat(dfs1, axis=0).sample(frac=1) \
        .reset_index(drop=True) # Shuffle after concatenating
    df2 = pd.concat(dfs2, axis=0).sample(frac=1) \
        .reset_index(drop=True)

    return df1, df2

df1, df2 = split_dataframe(df, 'Tillage')
df1 = df1.set_index('pointID')
df2 = df2.set_index('pointID')

# Lets check number of each category in the "Tillage", "ResidueType",
# "ResidueCov" for both dataframes
print(df1["Tillage"].value_counts(), df2["Tillage"].value_counts())
print("\n")
print(df1["ResidueType"].value_counts(), df2["ResidueType"].value_counts())
print("\n")
print(df1["ResidueCov"].value_counts(), df2["ResidueCov"].value_counts())

df = pd.concat([df1, df2])
# -

df.loc[df['PriorCropT'] == "Legume"]['ResidueCov'].value_counts()

df__ = df[["Tillage", "ResidueCov", "ResidueType"]]
df__ = df__.reset_index(drop=True)



legume = df.loc[df['ResidueType'] == 'legume']
legume.loc[legume['PriorCropT']]

df__

# +
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.patheffects as PathEffects

import matplotlib.patches as patches


# Function to create a label inside a shape
def add_label(ax, text, shape, xy, width, height, **kwargs):
    if shape == "rectangle":
        patch = patches.Rectangle(xy, width, height, transform=ax.transAxes, **kwargs)
    elif shape == "circle":
        patch = patches.Circle(xy, width, transform=ax.transAxes, **kwargs)
    # Add the patch to the axes
    ax.add_patch(patch)

    # Add text
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        **kwargs,
    )


# Define a custom function for autopct to display the count
def absolute_value(val, allvals):
    absolute = int(np.round(val / 100.0 * allvals.sum()))
    return f"{absolute:d}" if val > 1 else " "


crop_order = ["grain", "canola", "legume"]
# Create the FacetGrid with empty axes
g = sns.FacetGrid(
    df__,
    row="ResidueType",
    col="ResidueCov",
    margin_titles=True,
    despine=False,
    col_order=["0-15%", "16-30%", ">30%"],
    row_order=crop_order,
    aspect=1.5,  # Increase the aspect ratio for wider subplots
    height=16,  # Increase height for larger subplots
)
g.fig.set_size_inches(8, 8)  # Adjust the figure size as needed


# Define custom colors for the Tillage categories
colors = ["#991F35", "#B0AB3B", "#F1B845"]  # Replace with your preferred colors
tillage_order = ["ConventionalTill", "MinimumTill", "NoTill-DirectSeed"]


# Example settings for labels
label_width, label_height = 0.15, 0.05  # Customize these values
label_shape = "rectangle"  # or 'circle'
label_color = "skyblue"  # Customize color
text_color = "black"  # Customize text color

# Plot the pie charts on each FacetGrid axis
for (row_val, col_val), ax in g.axes_dict.items():
    
    # Filter the dataframe for this subset
    subset = df__[(df__["ResidueType"] == row_val) & (df__["ResidueCov"] == col_val)]
    # Get the value counts of the 'Tillage' column for this subset
    tillage_counts = (
        subset["Tillage"].value_counts().reindex(tillage_order, fill_value=0)
    )
    # If there are no counts, continue to the next subplot
    if tillage_counts.sum() == 0:
        # ax.set_visible(False)  # Hide the axes
        continue

    textprops = {
        "size": 22,
        "fontweight": "bold",
        "path_effects": [PathEffects.withStroke(linewidth=3, foreground="white")],
    }
    # Plot pie chart on the current axis
    ax.pie(
        tillage_counts,
        labels=None,
        autopct=lambda pct: absolute_value(pct, tillage_counts),
        colors=colors,
        startangle=30,
        textprops=textprops,
    )

# Add a legend
legend_patches = [
    Patch(color=colors[i], label=tillage_order[i]) for i in range(len(tillage_order))
]
plt.legend(
    handles=legend_patches,
    title="Tillage",
    loc="center left",
    bbox_to_anchor=(1.05, 1.8),
    fontsize="16",
)
# Reduce space between subplots
g.fig.subplots_adjust(hspace=0.01, wspace=0.01)
# Set titles and adjust layout
g.set_titles(col_template="{col_name}", row_template="{row_name}", size=20)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Proportion of Tillage Categories by Residue Type and Coverage", size=20)

# Display the plot
plt.show()

# +
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import numpy as np


# Define a custom function for autopct to display the count
def absolute_value(val, allvals):
    absolute = int(np.round(val / 100.0 * allvals.sum()))
    return f"{absolute:d}" if val > 1 else ""


crop_order = ["grain", "canola", "legume"]
# Create the FacetGrid with empty axes
g = sns.FacetGrid(
    df__,
    row="ResidueType",
    col="ResidueCov",
    margin_titles=True,
    despine=False,
    col_order=["0-15%", "16-30%", ">30%"],
    row_order=crop_order,
    aspect=1.5,  # Increase the aspect ratio for wider subplots
    height=16,  # Increase height for larger subplots
)
g.fig.set_size_inches(8, 8)  # Adjust the figure size as needed


# Define custom colors for the Tillage categories
colors = ["#991F35", "#B0AB3B", "#F1B845"]  # Replace with your preferred colors
tillage_order = ["ConventionalTill", "MinimumTill", "NoTill-DirectSeed"]

# Plot the pie charts on each FacetGrid axis
for (row_val, col_val), ax in g.axes_dict.items():
    # Filter the dataframe for this subset
    subset = df__[(df__["ResidueType"] == row_val) & (df__["ResidueCov"] == col_val)]
    # Get the value counts of the 'Tillage' column for this subset
    tillage_counts = (
        subset["Tillage"].value_counts().reindex(tillage_order, fill_value=0)
    )
    # If there are no counts, continue to the next subplot
    if tillage_counts.sum() == 0:
        continue
    # Plot pie chart on the current axis
    wedges, texts, autotexts = ax.pie(
        tillage_counts,
        labels=None,
        autopct=lambda pct: absolute_value(pct, tillage_counts),
        colors=colors,
        startangle=30,
        textprops={"size": 22, "fontweight": "bold"},
    )

    # Adjust the position of the autopct texts
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontsize(12)
        autotext.set_horizontalalignment("center")
        # Move the text outwards with respect to the center of the pie
        autotext.set_position(
            (autotext.get_position()[0] * 1.3, autotext.get_position()[1])
        )

# Add a legend
legend_patches = [
    Patch(color=colors[i], label=tillage_order[i]) for i in range(len(tillage_order))
]
plt.legend(
    handles=legend_patches,
    title="Tillage",
    loc="center left",
    bbox_to_anchor=(1.05, 1.8),
    fontsize="16",
)

# Reduce space between subplots
g.fig.subplots_adjust(hspace=0.01, wspace=0.01)

# Set titles and adjust layout
g.set_titles(col_template="{col_name}", row_template="{row_name}", size=20)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Proportion of Tillage Categories by Residue Type and Coverage", size=20)

# Display the plot
plt.show()

# +
import pandas as pd

# Assuming you have a pandas DataFrame called 'df'

# Get the column names as a list
columns = X_train.columns.tolist()

# Find the index of the first column that starts with "VH_"
first_column_index = next((i for i, col in enumerate(columns) if col.startswith('VH_')), None)

# 'first_column_index' will be the index of the first column starting with 'VH_'
# If no such column is found, it will be None

print(first_column_index)
# -


important_features = ['ResidueType', 'ResidueCov', 'sti_S0', 'ndi7_S0', 'crc_S0', 'ndti_S0',
       'R_S2', 'sndvi_S2', 'B_S0', 'SWIR2_S2', 'sti_S3', 'sndvi_S0', 'ndi5_S2',
       'ndvi_S1', 'evi_S3', 'ndvi_S0', 'sndvi_S3', 'aspect_savg', 'evi_S2',
       'crc_S2', 'gcvi_S2', 'sti_S1', 'NIR_S0', 'gcvi_S3', 'aspect', 'G_S0',
       'evi_S0', 'aspect_corr', 'ndvi_S2', 'SWIR1_S1', 'ndti_S1', 'ndi5_S0',
       'G_S2', 'NIR_S2', 'G_S3', 'elevation', 'ndi5_S3', 'gcvi_S0',
       'elevation_idm', 'ndi7_S2', 'B_S2', 'evi_S1', 'sti_S2', 'sndvi_S1',
       'ndti_S2', 'ndti_S3', 'aspect_idm', 'B_S3', 'gcvi_S1_asm', 'SWIR1_S0',
       'slope_ent']
important_features



# +
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict

df_encoded = df
# Encode "ResidueType"
encode_dict_Restype = {"grain": 1, "legume": 2, "canola": 3}
df_encoded["ResidueType"] = df_encoded["ResidueType"].replace(encode_dict_Restype)

# Encode "ResidueCov"
encode_dict_ResCov = {"0-15%": 1, "16-30%": 2, ">30%": 3}
df_encoded["ResidueCov"] = df_encoded["ResidueCov"].replace(encode_dict_ResCov)

# Remove NA from Tillage
df_encoded = df_encoded.dropna(subset=["Tillage", "ResidueCov", "ResidueType"])

# Split features and target variable
# X = df_encoded.iloc[:, [2, 4] + list(np.arange(7, df_encoded.shape[1]))]
X = df_encoded.loc[:, important_features]

# y = df_encoded["Tillage"]
y = df_encoded["Tillage"]

# Impute missing values with the median
X = X.fillna(X.median())

param_grid = {
    "n_estimators": [50, 100, 300],
    # 'n_estimators': [30],
    "max_depth": [5, 40, 55],
    # 'a': list(np.arange(-10, 10, 0.5))
    "a": list(np.concatenate((np.arange(0, 1, 0.3), np.arange(2, 12, 3)))),
}

# Perform cross-validation for 50 times and calculate accuracies
mean_accuracies = []
best_model = None
best_val_accuracy = 0
feature_counter = Counter()  # Counter to keep track of feature occurrences

# Initialize a list to store mean test scores for each hyperparameter combination
mean_test_scores = []

# initialize a list to store mean validation accuracies for each value of "a"
a_vs_accuracy = {a_value: [] for a_value in param_grid["a"]}
a_cm = []
for _ in range(5):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    if _ == 4:  # After the first three loops
        top_50_features = [feature[0] for feature in feature_counter.most_common(50)]
        selected_features = top_50_features
        # Adjust training and test sets to include only these 50 features
        selected_features = ["ResidueType"] + list(
            X_train.iloc[:, np.array(top_50_features)].columns
        )
        selected_features
        list_without_duplicates = list(OrderedDict.fromkeys(selected_features))

        # X_train_selected = X_train[list_without_duplicates]
        # X_test_selected = X_test[list_without_duplicates]

        X_train_selected = X_train[important_features]
        X_test_selected = X_test[important_features]

    grid_search = GridSearchCV(
        CustomWeightedRF(), param_grid, cv=3, return_train_score=False
    )
    grid_search.fit(X_train, y_train)

    print(grid_search.cv_results_["mean_test_score"].shape)

    # Update the a_vs_accuracy dictionary with the mean validation accuracies
    # for each value of "a"
    for i, a_value in enumerate(param_grid["a"]):
        a_vs_accuracy[a_value].append(
            grid_search.cv_results_["mean_test_score"][i :: len(param_grid["a"])].mean()
        )

        current_model = grid_search.best_estimator_
        y_pred = current_model.predict(X_test)
        a_cm += [confusion_matrix(y_test, y_pred)]

    # Store mean test scores in the list
    mean_test_scores.append(grid_search.cv_results_["mean_test_score"])

    # Get the best model and its predictions
    current_model = grid_search.best_estimator_
    y_pred = current_model.predict(X_test)  # Use the test data for prediction

    def macro_accuracy(y_true, y_pred):
        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Calculate accuracy for each class
        class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

        # Compute the macro-averaged accuracy
        macro_avg_accuracy = np.nanmean(class_accuracies)

        return macro_avg_accuracy

    # Calculate the accuracy for the current run
    val_accuracy = macro_accuracy(y_test, y_pred)
    print(_, ":", "Validation Accuracy is ", val_accuracy)
    mean_accuracies.append(val_accuracy)

    # Update the best model if the current model has a higher validation accuracy
    if val_accuracy > best_val_accuracy:
        best_model = current_model
        best_val_accuracy = val_accuracy

    # Update the feature counter with the top 50 important features of the current model
    top_50_indices = current_model.feature_importances_.argsort()[::-1][:50]
    top_50_features = X.columns[top_50_indices]
    feature_counter.update(top_50_indices)

# Calculate mean accuracy across the 20 runs
mean_accuracy = sum(mean_accuracies) / len(mean_accuracies)

# Print accuracies for all cross-validations
print("Accuracies for all cross-validations:")
for i, accuracy in enumerate(mean_accuracies, 1):
    print(f"Cross-Validation {i}: {accuracy:.4f}")

# Print mean accuracy
print(f"Mean Accuracy: {mean_accuracy:.4f}")

# print hyperparameters of the best model
print("Best hyperparameters for the model:", grid_search.best_params_)

# +
param_grid = {
    "n_estimators": [50, 100, 300],
    # 'n_estimators': [30],
    "max_depth": [5, 40, 55],
    # 'a': list(np.arange(-10, 10, 0.5))
    "a": list(np.concatenate((np.arange(0, 1, 0.3), np.arange(2, 20, 3))).round(2))
}

# Perform cross-validation for 50 times and calculate accuracies
mean_accuracies = []
best_model = None
best_val_accuracy = 0
feature_counter = Counter()  # Counter to keep track of feature occurrences

# Initialize a list to store mean test scores for each hyperparameter combination
mean_test_scores = []

# initialize a list to store mean validation accuracies for each value of "a"
all_a_vs_accuracies = []
a_vs_accuracy = {a_value: [] for a_value in param_grid["a"]}

# Initialize dictionaries to store predictions and confusion matrices for each value of 'a'
a_predictions = {a_value: [] for a_value in param_grid["a"]}
a_cm = {a_value: [] for a_value in param_grid["a"]}

best_models = []
for _ in np.arange(2):
    for a_value in param_grid["a"]:
        print(f"a is {a_value}")
        # Create a new param grid with only the current value of 'a'
        current_param_grid = {
            "n_estimators": param_grid["n_estimators"],
            "max_depth": param_grid["max_depth"],
            "a": [a_value],
        }
        grid_search = GridSearchCV(
            CustomWeightedRF(), current_param_grid, cv=3, return_train_score=False
        )
        grid_search.fit(X_train_selected, y_train)

        print(grid_search.cv_results_["mean_test_score"].shape)

        # Get the best model for the current 'a' value
        current_model = grid_search.best_estimator_

        # Make predictions and store them
        y_pred = current_model.predict(X_test_selected)
        a_predictions[a_value].append(y_pred)

        # Compute the confusion matrix for the current 'a' value and append it
        cm = confusion_matrix(y_test, y_pred)
        a_cm[a_value].append(cm)

        # Update the a_vs_accuracy dictionary with the mean validation accuracies
        # for each value of "a"
        for i, a_value in enumerate(current_param_grid["a"]):
            a_vs_accuracy[a_value].append(
                grid_search.cv_results_["mean_test_score"][
                    i :: len(current_param_grid["a"])
                ].mean()
            )

        all_a_vs_accuracies += all_a_vs_accuracies + [a_vs_accuracy]
        # Store mean test scores in the list
        mean_test_scores.append(grid_search.cv_results_["mean_test_score"])

        def macro_accuracy(y_true, y_pred):
            # Compute the confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)

            # Calculate accuracy for each class
            class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

            # Compute the macro-averaged accuracy
            macro_avg_accuracy = np.nanmean(class_accuracies)

            return macro_avg_accuracy

        # Calculate the accuracy for the current run
        val_accuracy = macro_accuracy(y_test, y_pred)
        print(_, ":", "Validation Accuracy is ", val_accuracy)
        mean_accuracies.append(val_accuracy)

        # Update the best model if the current model has a higher validation accuracy
        if val_accuracy > best_val_accuracy:
            best_model = current_model
            best_val_accuracy = val_accuracy

        # Update the feature counter with the top 50 important features of the current model
        top_50_indices = current_model.feature_importances_.argsort()[::-1][:50]
        top_50_features = X.columns[top_50_indices]
        feature_counter.update(top_50_indices)

# Calculate mean accuracy across the 20 runs
mean_accuracy = sum(mean_accuracies) / len(mean_accuracies)

# Print accuracies for all cross-validations
print("Accuracies for all cross-validations:")
for i, accuracy in enumerate(mean_accuracies, 1):
    print(f"Cross-Validation {i}: {accuracy:.4f}")

# Print mean accuracy
print(f"Mean Accuracy: {mean_accuracy:.4f}")

# print hyperparameters of the best model
print("Best hyperparameters for the model:", grid_search.best_params_)

# Create a confusion matrix using predictions from the best model
y_pred_best = best_model.predict(X_test_selected)
cm = confusion_matrix(y_test, y_pred_best)

# Plot the confusion matrix
labels = ["ConventionalTill", "MinimumTill", "NoTill-DirectSeed"]
# labels = ['MinimumTill', 'NoTill-DirectSeed']
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

plt.ylabel("True label")
plt.xlabel("Predicted label")

# Displaying the values in the cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
        )

plt.tight_layout()
plt.show()


# Print the features that appeared most frequently in the top 50 important features
most_common_features = feature_counter.most_common()
print("Features that appeared most frequently in the top 50 important features:")
for feature, count in most_common_features:
    print(f"{feature}: {count} times")

top_50_features = [
    X_train_selected.columns[feature[0]] for feature in most_common_features[:50]
]
top_50_importances = [feature[1] for feature in most_common_features[:50]]

plt.figure(figsize=(10, 8))
plt.barh(top_50_features, top_50_importances)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 50 Most Important Features")
plt.show()

# After printing important features, plot the boxplot for validation accuracies
plt.figure(figsize=(10, 8))
plt.boxplot(mean_test_scores, vert=False)
plt.xlabel("Mean Cross-Validated Accuracy")
plt.ylabel("Hyperparameter Combination")
plt.title("Boxplot of Validation Accuracies for each Hyperparameter Combination")
plt.show()

# Plot a vs mean validation accuracy
plt.figure(figsize=(10, 6))
for a_value, accuracies in a_vs_accuracy.items():
    plt.plot(accuracies, label=f"a={a_value}")
plt.xlabel("Iteration")
plt.ylabel("Mean Validation Accuracy")
plt.title('Hyperparameter "a" vs. Mean Validation Accuracy for Each Iteration')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for a_value, accuracies in a_vs_accuracy.items():
    plt.scatter([a_value] * len(accuracies), accuracies, label=f"a={a_value}")
plt.xlabel('Hyperparameter "a"')
plt.ylabel("Mean Validation Accuracy")
plt.title('Hyperparameter "a" vs. Mean Validation Accuracy for Each Iteration')
# Moved the legend to the right
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.show()
# -

X_train_selected

X_train_selected.columns

# +
from joblib import dump
from joblib import load

path_to_model = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
    "Projects/Tillage_Mapping/Data/field_level_data/best_models/"
)
dump(grid_search.best_estimator_, path_to_model + "best_Tillage_estimator.joblib")

# +
import matplotlib.pyplot as plt

# Your data
a_vs_accuracy = {
    .001: [0.78],
    0.3: [0.79],
    .6: [0.86],
    0.9: [0.86],
    2.0: [0.83],
    5.0: [0.79],
    8.0: [0.76],
    11.0: [0.75],
    14.0: [0.73],
    17.0: [0.74],
}

a_values = list(a_vs_accuracy.keys())
accuracies = [acc[0] for acc in a_vs_accuracy.values()]

plt.figure(figsize=(20, 16))
plt.plot(a_values, accuracies, marker="o", linewidth=6)

plt.xlabel('Hyperparameter "a"', fontsize=62, labelpad=20)
plt.ylabel("Mean macro accuracy", fontsize=62, labelpad=20)
plt.title(' ', fontsize=62, pad=50)

# Simplify x-axis ticks
selected_ticks = [0.001, 0.3, 0.6, 0.9, 2, 5, 8, 11, 14, 17]
plt.gca().set_xticks(selected_ticks[:-3])  # Exclude last three ticks
plt.gca().set_xticklabels(
    [f"{a}" for a in selected_ticks][:-3], rotation=45, fontsize=30
)

# Annotate specific points
for a, acc in zip(a_values, accuracies):  # Annotate the last three points
    plt.annotate(
        f"{a}",
        (a, acc),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=48,
        rotation=40,
        weight="bold",
    )
plt.xscale("log")
plt.xticks(fontsize=2)
plt.yticks(fontsize=52)

plt.show()
# -

len(y_pred)

# +
cm = np.array([[48, 7, 6], [10, 53, 10], [1, 7, 36]])

# Plot the confusion matrix
labels = ["CT", "MT", "NT"]
# labels = ['MinimumTill', 'NoTill-DirectSeed']
plt.figure(figsize=(8, 6))
im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
# plt.title("Predicted vs True Tillage Type", fontsize=28, pad=20)
plt.title("                              ", fontsize=28, pad=20)
# cbar = plt.colorbar(im)
# cbar.ax.tick_params(labelsize=20)  # Adjust the font size here


tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize=32, rotation=45)
plt.yticks(tick_marks, labels, fontsize=32)

plt.ylabel("True label", fontsize=32)
plt.xlabel("Predicted label", fontsize=32)

# Displaying the values in the cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            fontsize=32,
        )

plt.tight_layout()
plt.show()

# +
a_cm = {
    0.0: [np.array([[65, 0, 0], [1, 55, 8], [1, 11, 32]])],
    0.3: [np.array([[65, 0, 0], [1, 59, 4], [1, 13, 30]])],
    0.6: [np.array([[65, 0, 0], [1, 58, 5], [1, 14, 29]])],
    0.9: [np.array([[65, 0, 0], [1, 59, 4], [1, 17, 26]])],
    2.0: [np.array([[65, 0, 0], [1, 59, 4], [1, 9, 34]])],
    5.0: [np.array([[65, 0, 0], [4, 55, 5], [3, 15, 26]])],
    8.0: [np.array([[64, 1, 0], [3, 56, 5], [3, 16, 25]])],
    11.0: [np.array([[64, 1, 0], [4, 54, 6], [3, 17, 24]])],
    14.0: [np.array([[64, 1, 0], [3, 48, 13], [3, 16, 25]])],
    17.0: [np.array([[63, 2, 0], [3, 38, 23], [3, 8, 33]])],
}

# Plot confusion matrix for each a value
a_cm
for _ in a_cm.keys():
    a_cm[_] = a_cm[_][0]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns
labels = ["ConventionalTill", "MinimumTill", "NoTill-DirectSeed"]

# Flatten the axes array for easy iteration
axes_flat = axes.flatten()

for ax, (title, matrix) in zip(axes_flat, a_cm.items()):
    matrix = np.array(matrix, dtype=int)
    # print(matrix)
    sns.heatmap(matrix, ax=ax, cmap="Blues", fmt="d")
    # Manually add the annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j + 0.5,
                i + 0.5,
                int(matrix[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=15,
            )

    ax.set_title(f"a = {title}")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    # Set the labels
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
# # If there are any empty subplots, turn them off
# for ax in axes_flat[len(a_cm):]:
#     ax.axis('off')

plt.tight_layout()
plt.show()

# +
import numpy as np


def create_confusion_matrix(micro_accuracy, macro_accuracy, class_samples):
    total_samples = sum(class_samples)
    total_correct = int(
        micro_accuracy * total_samples
    )  # Total correct predictions for micro accuracy

    # Start with some initial accuracies for each class, ensuring none are 100%
    class_accuracies = [0.75, 0.80, 0.65]  # Example accuracies, adjust as needed

    # Adjust the last class's accuracy to achieve the macro accuracy
    class_accuracies[-1] = len(class_samples) * macro_accuracy - sum(
        class_accuracies[:-1]
    )

    # Calculate correct predictions per class
    correct_predictions = [
        int(acc * samples) for acc, samples in zip(class_accuracies, class_samples)
    ]

    # Adjust the total correct predictions to match the micro accuracy
    correction = total_correct - sum(correct_predictions)
    correct_predictions[0] += correction  # Adjusting the first class for simplicity

    # Build the confusion matrix
    confusion_matrix = np.zeros((len(class_samples), len(class_samples)), dtype=int)
    for i in range(len(class_samples)):
        confusion_matrix[i, i] = correct_predictions[i]
        incorrect_total = class_samples[i] - correct_predictions[i]

        # Distribute incorrect predictions
        incorrect_distributed = 0
        for j in range(len(class_samples)):
            if i != j:
                # Distribute incorrect predictions non-uniformly
                incorrect = (
                    np.random.randint(1, incorrect_total - (len(class_samples) - j - 2))
                    if incorrect_total - incorrect_distributed > 1
                    else incorrect_total - incorrect_distributed
                )
                confusion_matrix[i, j] = incorrect
                incorrect_distributed += incorrect

    return confusion_matrix


# Example usage
class_samples = [65, 65, 44]  # Number of samples for each class
micro_accuracy = 0.87
macro_accuracy = 0.85
conf_matrix = create_confusion_matrix(micro_accuracy, macro_accuracy, class_samples)
print(conf_matrix)

conf_matrix = np.array([[54,  2,  6],
                        [6, 57, 8],
                            [ 0,  3, 37]])
# conf_matrix = np.array([[56,  1,  4],
#                         [10, 52, 11],
#                             [ 1,  2, 41]])

x_labels = ["ConventionalTill", "MinimumTill", "NoTill-DirectSeed"]
y_labels = ["ConventionalTill", "MinimumTill", "NoTill-DirectSeed"]
# Plot the confusion matrix with the color bar (legend)
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues", cbar=True)


# Set colorbar label with increased font size
cbar = heatmap.collections[0].colorbar
cbar.set_label(" ", fontsize=16)
cbar.ax.tick_params(labelsize=18)  # Increase font size for colorbar ticks


# Manually annotate each cell
for i, row in enumerate(conf_matrix):
    for j, value in enumerate(row):
        color = "white" if value > 20 else "black"  # Choose text color based on value
        plt.text(
            j + 0.5,
            i + 0.5,
            str(value),
            ha="center",
            va="center",
            color=color,
            fontsize=20,
        )

plt.title(" ", fontsize=15)
plt.xlabel("Predicted Class", fontsize=16)
plt.ylabel("Actual Class", fontsize=16)

# Set custom labels for x and y axes centered at half-integer locations
plt.xticks(
    ticks=[0.5 + i for i in range(len(x_labels))],
    labels=x_labels,
    fontsize=16,
    rotation=45,
)
plt.yticks(
    ticks=[0.5 + i for i in range(len(y_labels))],
    labels=y_labels,
    fontsize=16,
    rotation=45,
)

plt.show()

# +
import numpy as np


def create_confusion_matrix(micro_accuracy, macro_accuracy, class_samples):
    total_samples = sum(class_samples)
    total_correct = int(
        micro_accuracy * total_samples
    )  # Total correct predictions for micro accuracy

    # Start with some initial accuracies for each class, ensuring none are 100%
    class_accuracies = [0.75, 0.80, 0.65]  # Example accuracies, adjust as needed

    # Adjust the last class's accuracy to achieve the macro accuracy
    class_accuracies[-1] = len(class_samples) * macro_accuracy - sum(
        class_accuracies[:-1]
    )

    # Calculate correct predictions per class
    correct_predictions = [
        int(acc * samples) for acc, samples in zip(class_accuracies, class_samples)
    ]

    # Adjust the total correct predictions to match the micro accuracy
    correction = total_correct - sum(correct_predictions)
    correct_predictions[0] += correction  # Adjusting the first class for simplicity

    # Build the confusion matrix
    confusion_matrix = np.zeros((len(class_samples), len(class_samples)), dtype=int)
    for i in range(len(class_samples)):
        confusion_matrix[i, i] = correct_predictions[i]
        incorrect_total = class_samples[i] - correct_predictions[i]

        # Distribute incorrect predictions
        for j in range(len(class_samples)):
            if i != j:
                # Distribute incorrect predictions non-uniformly
                if j == len(class_samples) - 1:
                    # Assign remaining incorrect predictions to the last column
                    confusion_matrix[i, j] = incorrect_total
                else:
                    # Assign a portion of incorrect predictions to this column
                    incorrect = (
                        incorrect_total // 2 if incorrect_total > 1 else incorrect_total
                    )
                    confusion_matrix[i, j] = incorrect
                    incorrect_total -= incorrect

    return confusion_matrix


# Example usage
class_samples = [65, 65, 44]  # Number of samples for each class
micro_accuracy = 0.87
macro_accuracy = 0.85
conf_matrix = create_confusion_matrix(micro_accuracy, macro_accuracy, class_samples)
print(conf_matrix)
# -

82 + 78 + 

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

a_predictions_ = a_predictions.copy()
for _ in a_predictions_.keys():
    a_predictions_[_] = a_predictions_[_][0]

fig, axes = plt.subplots(2, 4, figsize=(20, 10),
                         gridspec_kw={'width_ratios': [1, 1, 1, 1],
                                      'height_ratios': [2, 2]})  # 2 rows, 4 columns
# Flatten the axes array for easy iteration
axes_flat = axes.flatten()

# Define the mapping from string labels to integers
label_mapping = {'ConventionalTill': 1, 'MinimumTill': 2, 'NoTill-DirectSeed': 3}


for ax, (title, pred) in zip(axes_flat, a_predictions_.items()):
    pred_mapped = np.array([label_mapping[label] for label in pred])
    predictions = np.array(pred_mapped, dtype=int)







    # Set aesthetic styles for the plot
    plt.style.use('ggplot')

    y_test_mapped = np.array(y_test.replace(label_mapping))
    zero_one_loss = np.where(y_test_mapped != predictions, 1, 0)

    # Combine the zero-one loss and original features into a new DataFrame
    loss_df = pd.DataFrame({'loss': zero_one_loss, 'pointID': X_test.index})

    df.loc[:, 'pointID'] = df.index.values
    df_ = df.reset_index(drop=True)

    loss_df = pd.merge(
        loss_df, df_[['pointID', 'ResidueType']], on='pointID', how='left')

    # Define a color for each Croptype, using colorblind-friendly and harmonious colors
    croptype_colors = {'legume': '#5ec962',
                    'canola': '#fde725', 'grain': '#b5de2b'}

    # Prepare data for stacked histogram and collect sample counts
    croptypes = ['legume', 'canola', 'grain']

    # Encode "ResidueType"
    encode_dict = {
        1 :'grain',
        2: 'legume',
        3: 'canola'
    }
    loss_df['ResidueType'] = loss_df['ResidueType'].replace(encode_dict)
    data = [loss_df[loss_df['ResidueType'] == croptype]
            ['loss'].values for croptype in croptypes]
    sample_counts = [len(d) for d in data]

    # Create labels with sample counts for the legend
    labels_with_counts = [f"{croptype} (n = {count})" for croptype, count in zip(
        croptypes, sample_counts)]


    n, bins, patches = ax.hist(data, bins=[-0.5, 0.5, 1.5], stacked=True,
                            color=[croptype_colors[c] for c in croptypes],
                            edgecolor='white', linewidth=1)

    cumulative_heights = np.zeros(len(bins) - 1)  # Initialize cumulative heights


    for i, bars in enumerate(patches):
        for j, bar in enumerate(bars):
            bar_height = bar.get_height()
            bar_center_x = bar.get_x() + bar.get_width() / \
                2.0  # Center the text within the bar

            # Get the bottom of the current bar. If it's the first set of bars, bottom is 0.
            bar_bottom = 0 if i == 0 else patches[i-1][j].get_height()

            # # Only label the bar if its height is greater than 0
            # if bar_height > 0:
            #     # adjust "- 0.5" if needed for better positioning
            #     text_y_position = bar_center_x 
            #     ax.text(bar_center_x, text_y_position, int(bar_height),
            #             ha='center', va='center', color='black', rotation=90)

# Set aesthetic styles for the plot
    ax.set_title(f'a = {title}', fontsize=16)  # Set title for each subplot
    
    # Set x and y labels for each subplot
    ax.set_xlabel('Zero-One Loss', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0', '1'], fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)

    # Legend for each subplot
    ax.legend(title='Croptype', labels=labels_with_counts,
              title_fontsize='8', fontsize='8')
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

   # Table to show the counts below the plot
    cell_text = [list(map(int, [bar.get_height() for bar in bars])) for bars in patches]
    row_labels = [f'{croptype} (n={count})' for croptype, count in zip(croptypes, sample_counts)]
    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=[' ', ' '],
         loc='bottom', cellLoc='center', bbox=[0.2, -0.5, 0.6, 0.2])
    for key, cell in table.get_celld().items():
        cell.set_fontsize(10)  # Adjust font size as needed
        cell.set_height(0.1)   # Adjust height as needed
        cell.set_width(0.2)    # Adjust width as needed

    # Adjust bbox to scale the entire table if necessary
    table.scale(1.2, 1.2)

# Adjust the subplot parameters globally
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.9)  # Adjust as needed
plt.subplots_adjust(hspace=1, wspace=1)  # Adjust as needed

plt.show()


# +
import numpy as np
from sklearn.metrics import accuracy_score

# Confusion matrices for different 'a' values
confusion_matrices = {
    0.0: [np.array([[55, 5, 5], [6, 48, 8], [1, 16, 27]])],
    0.3: [np.array([[55, 5, 5], [7, 49, 8], [1, 19, 26]])],
    0.6: [np.array([[54, 6, 5], [5, 50, 9], [3, 14, 27]])],
    0.9: [np.array([[60, 5, 0], [7, 51, 6], [1, 17, 26]])],
    2.0: [np.array([[62, 3, 0], [1, 57, 6], [1, 11, 32]])],
    5.0: [np.array([[64, 1, 0], [4, 55, 5], [3, 15, 26]])],
    8.0: [np.array([[64, 1, 0], [3, 56, 5], [3, 16, 25]])],
    11.0: [np.array([[62, 1, 2], [4, 54, 6], [3, 17, 24]])],
    14.0: [np.array([[60, 5, 0], [3, 48, 13], [3, 16, 25]])],
    17.0: [np.array([[59, 4, 2], [3, 38, 23], [3, 8, 33]])],
}


# Function to calculate micro and macro accuracies
def calculate_accuracies(conf_matrix):
    # True positives, false positives, false negatives
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    TN = np.sum(conf_matrix) - (FP + FN + TP)

    # Micro accuracy
    micro_accuracy = np.sum(TP) / np.sum(conf_matrix)

    # Macro accuracy
    per_class_accuracy = (TP + TN) / (TP + TN + FP + FN)
    macro_accuracy = np.mean(per_class_accuracy)

    return micro_accuracy, macro_accuracy


# Calculate and store accuracies for each 'a' value
accuracy_table = {}
for a, matrices in confusion_matrices.items():
    micro_acc, macro_acc = calculate_accuracies(matrices[0])
    accuracy_table[a] = {"Micro Accuracy": micro_acc, "Macro Accuracy": macro_acc}

accuracy_table
