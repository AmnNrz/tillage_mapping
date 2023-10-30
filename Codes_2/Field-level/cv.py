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
#%%
import pandas as pd

# # Read data
path_to_data = ("/Users/aminnorouzi/Library/CloudStorage/"
                "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
                "Projects/Tillage_Mapping/Data/field_level_data/FINAL_DATA/")

# path_to_data = ("/home/amnnrz/OneDrive - "
#                 "a.norouzikandelati/Ph.D/Projects/Tillage_Mapping/Data/")

df = pd.read_csv(path_to_data + "metric_finalDATA.csv", index_col=0)
df = df.dropna(subset=["Tillage", "ResidueType", "ResidueCov"])

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


# Lets check number of each category in the "Tillage", "ResidueType",
# "ResidueCov" for both dataframes
print(df1["Tillage"].value_counts(), df2["Tillage"].value_counts())
print("\n")
print(df1["ResidueType"].value_counts(), df2["ResidueType"].value_counts())
print("\n")
print(df1["ResidueCov"].value_counts(), df2["ResidueCov"].value_counts())



#%%
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
        return self
    
    def predict(self, X, **kwargs):
        return self.rf.predict(X)
    
    def predict_proba(self, X, **kwargs):
        return self.rf.predict_proba(X)
    
    @property
    def feature_importances_(self):
        return self.rf.feature_importances_


#%%

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


# df1 = df1.drop(columns='ResidueCov')

# Load your dataframe with categorical features
df = df1

# # # Perform one-hot encoding for "Residue Cover" features
# df_encoded = pd.get_dummies(df, columns=['ResidueCov'])
df_encoded = df

# Encode "ResidueType"
encode_dict = {
    'grain': 1,
    'legume': 2,
    'canola': 3
}
df_encoded['ResidueType'] = df_encoded['ResidueType'].replace(encode_dict)
df_encoded

# # Place the one-hot encoded columns on the left side of the dataframe
# ordered_columns = list(df_encoded.columns.difference(df.columns)) + \
# [col for col in df.columns if col not in ['ResidueCov']]
# df_ordered = df_encoded[ordered_columns]

df_ordered = df_encoded

# Remove NA from Tillage
df_ordered = df_ordered.dropna(subset=["Tillage", "ResidueCov", "ResidueType"])

le = LabelEncoder()
df_ordered['ResidueCov'] = le.fit_transform(df_ordered['ResidueCov'])

# Split features and target variable
X = df_ordered.iloc[:, np.concatenate(
    [np.arange(3, 4), np.arange(8, df_ordered.shape[1])]
    )]
    
y = df_ordered['ResidueCov']

# Impute missing values with the median
X = X.fillna(X.median())



# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    'n_estimators': [40, 50, 100, 300, 800],
    # 'n_estimators': [30],
    'max_depth': [5, 40, None],
    # 'a': list(np.arange(-10, 10, 0.5))
    'a': [0, 0.1, 0.2, 0.3, 0.5, 1]
}

# Perform cross-validation for 20 times and calculate accuracies
mean_accuracies = []
best_model = None
best_val_accuracy = 0
feature_counter = Counter()  # Counter to keep track of feature occurrences

# Initialize a list to store mean test scores for each hyperparameter combination
mean_test_scores = []

# initialize a list to store mean validation accuracies for each value of "a"
a_vs_accuracy = {a_value: [] for a_value in param_grid['a']}

for _ in range(10):

    if _ == 3:  # After the first three loops
        top_20_features = [feature[0]
            for feature in feature_counter.most_common(20)]
        selected_features = top_20_features
        # Adjust training and test sets to include only these 20 features
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    grid_search = GridSearchCV(
        CustomWeightedRF(), param_grid, cv=5, return_train_score=False)
    grid_search.fit(X_train, y_train)

    print(grid_search.cv_results_['mean_test_score'].shape)

    # Update the a_vs_accuracy dictionary with the mean validation accuracies 
    # for each value of "a"
    for i, a_value in enumerate(param_grid['a']):
        a_vs_accuracy[a_value].append(grid_search.cv_results_[
            'mean_test_score'][i::len(param_grid['a'])].mean())


    # Store mean test scores in the list
    mean_test_scores.append(grid_search.cv_results_['mean_test_score'])

    # Get the best model and its predictions
    current_model = grid_search.best_estimator_
    y_pred = current_model.predict(X_test)  # Use the test data for prediction

    # Calculate the accuracy for the current run
    val_accuracy = accuracy_score(y_test, y_pred)
    print(_, ":", "Validation Accuracy is ", val_accuracy)
    mean_accuracies.append(val_accuracy)

    # Update the best model if the current model has a higher validation accuracy
    if val_accuracy > best_val_accuracy:
        best_model = current_model
        best_val_accuracy = val_accuracy

    # Update the feature counter with the top 20 important features of the current model
    top_20_indices = current_model.feature_importances_.argsort()[::-1][:50]
    top_20_features = X.columns[top_20_indices]
    feature_counter.update(top_20_features)

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
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

# Plot the confusion matrix
labels = ['ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed']
# labels = ['MinimumTill', 'NoTill-DirectSeed']
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

plt.ylabel('True label')
plt.xlabel('Predicted label')

# Displaying the values in the cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.tight_layout()
plt.show()


# Print the features that appeared most frequently in the top 20 important features
most_common_features = feature_counter.most_common()
print("Features that appeared most frequently in the top 20 important features:")
for feature, count in most_common_features:
    print(f"{feature}: {count} times")

# Plot the 20 most important features over all runs
top_20_features = [feature[0] for feature in most_common_features[:20]]
top_20_importances = [feature[1] for feature in most_common_features[:20]]

plt.figure(figsize=(10, 8))
plt.barh(top_20_features, top_20_importances)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Top 20 Most Important Features')
plt.show()

# After printing important features, plot the boxplot for validation accuracies
plt.figure(figsize=(10, 8))
plt.boxplot(mean_test_scores, vert=False)
plt.xlabel('Mean Cross-Validated Accuracy')
plt.ylabel('Hyperparameter Combination')
plt.title('Boxplot of Validation Accuracies for each Hyperparameter Combination')
plt.show()

# Plot a vs mean validation accuracy
plt.figure(figsize=(10, 6))
for a_value, accuracies in a_vs_accuracy.items():
    plt.plot(accuracies, label=f'a={a_value}')
plt.xlabel('Iteration')
plt.ylabel('Mean Validation Accuracy')
plt.title('Hyperparameter "a" vs. Mean Validation Accuracy for Each Iteration')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for a_value, accuracies in a_vs_accuracy.items():
    plt.scatter([a_value] * len(accuracies), accuracies, label=f'a={a_value}')
plt.xlabel('Hyperparameter "a"')
plt.ylabel('Mean Validation Accuracy')
plt.title('Hyperparameter "a" vs. Mean Validation Accuracy for Each Iteration')
# Moved the legend to the right
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
# -


#%%

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set aesthetic styles for the plot
plt.style.use('ggplot')

# Compute the zero-one loss for each sample in the test set
y_pred_best = best_model.predict(X_test)
zero_one_loss = np.where(y_test != y_pred_best, 1, 0)

# Combine the zero-one loss and original features into a new DataFrame
loss_df = pd.DataFrame({'loss': zero_one_loss, 'pointID': X_test.index})
loss_df = pd.merge(
    loss_df, df[['pointID', 'PriorCropT']], on='pointID', how='left')

# Define a color for each Croptype, using colorblind-friendly and harmonious colors
croptype_colors = {'legume': '#5ec962',
                   'canola': '#fde725', 'grain': '#b5de2b'}

# Prepare data for stacked histogram and collect sample counts
croptypes = ['legume', 'canola', 'grain']
data = [loss_df[loss_df['PriorCropT'] == croptype]
        ['loss'].values for croptype in croptypes]
sample_counts = [len(d) for d in data]

# Create labels with sample counts for the legend
labels_with_counts = [f"{croptype} (n = {count})" for croptype, count in zip(
    croptypes, sample_counts)]

# Plot stacked histogram
fig, ax = plt.subplots()
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

        # Only label the bar if its height is greater than 0
        if bar_height > 0:
            # adjust "- 0.5" if needed for better positioning
            text_y_position = bar_bottom + bar_height - 0.5
            ax.text(bar_center_x, text_y_position, int(bar_height),
                    ha='center', va='center', color='black')

# Existing code for customization
plt.xlabel('Zero-One Loss', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks([0, 1], fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Croptype', labels=labels_with_counts,
           title_fontsize='13', fontsize='12')
plt.title('Histogram of Loss Across Each Croptype', fontsize=16)
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

plt.tight_layout()
plt.show()

# -

grid_search.cv_results_['mean_test_score']


# +
# scores_matrix = grid_search.cv_results_['mean_test_score'][4].reshape(
#     len(param_grid['max_depth']),
#     len(param_grid['n_estimators'])
# )
# plt.figure(figsize=(12, 6))
# annot_kws = {"size": 10}

# sns.heatmap(scores_matrix, annot=False,
#             xticklabels=param_grid['n_estimators'],
#             yticklabels=param_grid['max_depth'], cmap="YlGnBu")
# plt.xlabel('Number of Estimators (n_estimators)')
# plt.ylabel('Maximum Depth (max_depth)')
# plt.title('Mean Test Scores for Hyperparameter Combinations')
# plt.show()


# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

param_grid = {
    'n_estimators': [20, 30, 40, 50, 100, 300],
    'max_depth': [5, 10, 20, 30, 45],
    'a': [0, 0.2, 0.4, 0.6, 1, 3, 10]
}

# Set up the colormap
cmap = plt.get_cmap('viridis')
norm = mpl.colors.Normalize(vmin=0, vmax=len(param_grid['max_depth']) - 1)

# Create a subplot for each value of 'a' in 2-column layout
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 4 * 4))
axes = axes.ravel()  # Flatten axes to easily iterate

# Remove the last unused ax
fig.delaxes(axes[-1])

# Create custom legend handles
legend_handles = [mpl.patches.Patch(color=cmap(norm(i)), label=f"max_depth={depth}")
                  for i, depth in enumerate(param_grid['max_depth'])]

for ax, a_val in zip(axes, param_grid['a']):
    for i, depth in enumerate(param_grid['max_depth']):
        start_index = (param_grid['a'].index(a_val) * len(param_grid['max_depth']) * len(param_grid['n_estimators'])) + \
                      (i * len(param_grid['n_estimators']))
        end_index = start_index + len(param_grid['n_estimators'])
        scores_for_depth = mean_test_scores[4][start_index:end_index]

        # Get color for this depth from the colormap
        color = cmap(norm(i))

        ax.plot(param_grid['n_estimators'], scores_for_depth,
                marker='o', linestyle='-', color=color)
        ax.set_title(f'a = {a_val}')
        ax.set_xlabel('n_estimators')
        ax.set_ylabel('Validation Accuracy')

# Create a common legend for max_depth using the custom handles
fig.subplots_adjust(right=0.8, hspace=0.4, wspace=0.3)
axes[-1].legend(handles=legend_handles, loc="lower right", title="max_depth")

plt.tight_layout()
plt.show()

