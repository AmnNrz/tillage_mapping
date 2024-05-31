# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: tillenv
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np

# final_df = pd.read_csv(r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\field_level_data\final_dataframe_landsat.csv")
final_df = pd.read_csv(
    "/Users/aminnorouzi/Library/CloudStorage/GoogleDrive-a.norouzikandelati@wsu.edu/My Drive/Ph.D._projects/Tillage_Mapping/Data/field_level_data/final_dataframe_landsat.csv")
final_df_withTillage = pd.read_csv(
    "/Users/aminnorouzi/Library/CloudStorage/GoogleDrive-a.norouzikandelati@wsu.edu/My Drive/Ph.D._projects/Tillage_Mapping/Data/field_level_data/finalCleaned.csv")
# crop_pred = pd.read_csv(r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\crop_predictions_2021_2022.csv", index_col=0)
cdl_data = pd.read_csv(
    '/Users/aminnorouzi/Library/CloudStorage/GoogleDrive-a.norouzikandelati@wsu.edu/My Drive/Ph.D._projects/Tillage_Mapping/Data/CDL crop data/most_frequent_cdl_classes_survey_polygons.csv')

# -

final_df_withTillage

cdl_data.most_frequent_class.value_counts().index

# +
replacement_dict = {24: 'grain', 23: 'grain', 51: 'legume',
                    51: 'legume', 31: 'canola', 53: 'legume',
                    21: 'grain', 51: 'legume', 52: 'legume',
                    28: 'grain' }

cdl_data['most_frequent_class'] = cdl_data['most_frequent_class'].replace(replacement_dict)
cdl_data = cdl_data.loc[cdl_data['most_frequent_class'].isin(['grain', 'legume', 'canola'])]
cdl_data['most_frequent_class'].value_counts()
# -

cdl_data = cdl_data[['pointID', 'most_frequent_class']]
cdl_data

s1 = final_df.pointID.reset_index(drop=True)
s2 = final_df_withTillage.pointID.reset_index(drop=True)

s2.isin(s1).value_counts()

final_df

# +
final_df_rc = final_df[['pointID', 'ResidueCov']]
final_df_rest = final_df.drop(labels="ResidueCov", axis=1)

# Impute missing values with the median
final_df_rest = final_df_rest.fillna(final_df_rest.median())

final_df = pd.merge(final_df_rc, final_df_rest, on="pointID", how="left")

# Verify that all missing values have been imputed
print(final_df.isnull().sum())
final_df

# -

final_df
rc_crop_df = pd.merge(final_df, cdl_data, on='pointID', how='left')
rc_crop_df

# +

rc_crop_df.rename({"most_frequent_class":"cropType_pred"}, axis=1, inplace=True)
rc_crop_df["rc_crop"] = rc_crop_df["ResidueCov"] + "_" + rc_crop_df["cropType_pred"]
rc_crop_df
# -

rc_crop_df["rc_crop"].unique()

final_df

# +
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# Load your dataframe
df = final_df

# Split features and target variable
X = df.drop(['ResidueCov', 'pointID'], axis=1)
y = df['ResidueCov']

# Perform oversampling to balance the classes
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Create a new dataframe with the resampled data
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['ResidueCov'] = y_resampled

# Train-test split on the resampled data
X_train, X_test, y_train, y_test = train_test_split(
    df_resampled.drop('ResidueCov', axis=1), df_resampled['ResidueCov'],
    test_size=0.2, random_state=42
)

# Train a random forest classifier to find important features
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_

# Sort feature importances in descending order
sorted_indices = importances.argsort()[::-1]

# Select the top 100 most important features
top_100_indices = sorted_indices[:100]
top_100_features = X_train.columns[top_100_indices]

# Train a new random forest model with the top 100 features using cross-validation
param_grid = {
    'n_estimators': [10, 20, 25, 30, 40, 50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20, 30, 35, 40]
}

# Perform cross-validation for 20 times and calculate mean accuracies
mean_train_accuracy = []
mean_val_accuracy = []
best_model = None
best_val_accuracy = 0
feature_counter = Counter()  # Counter to keep track of feature occurrences

for _ in range(10):
    print(_)
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid_search.fit(X_train[top_100_features], y_train)

    # Get the best model and its predictions
    current_model = grid_search.best_estimator_
    y_pred = current_model.predict(X_test[top_100_features])

    # Calculate the mean accuracies for training and validation sets
    train_accuracy = grid_search.best_score_
    val_accuracy = accuracy_score(y_test, y_pred)
    
    mean_train_accuracy.append(train_accuracy)
    mean_val_accuracy.append(val_accuracy)
    
    # Update the best model if the current model has a higher validation accuracy
    if val_accuracy > best_val_accuracy:
        best_model = current_model
        best_val_accuracy = val_accuracy
        
    # Update the feature counter with the top 20 important features of the current model
    top_20_indices = current_model.feature_importances_.argsort()[::-1][:20]
    top_20_features = top_100_features[top_20_indices]
    feature_counter.update(top_20_features)

# Calculate mean accuracies across the 20 runs
mean_train_accuracy = sum(mean_train_accuracy) / len(mean_train_accuracy)
mean_val_accuracy = sum(mean_val_accuracy) / len(mean_val_accuracy)

# Create a confusion matrix using predictions from the best model
y_pred_best = best_model.predict(X_test[top_100_features])
cm = confusion_matrix(y_test, y_pred_best)

# Plot the confusion matrix
# labels = ['16-30%_grain', '0-15%_grain', '0-15%_Canola', '16-30%_legume',
#        '>30%_grain', '0-15%_legume', '16-30%_Canola', '>30%_legume']
labels = ['0-15%', '16-30%', '>30%']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Print the mean accuracies across the 20 runs
print(f"Mean Accuracy (Training): {mean_train_accuracy:.4f}")
print(f"Mean Accuracy (Validation): {mean_val_accuracy:.4f}")

# Print the hyperparameters for the best model
print("Best Model Hyperparameters:")
print(best_model.get_params())

# Plot the 20 most important features
feature_importances = pd.Series(importances[top_100_indices], index=top_100_features)
top_20_features = feature_importances.nlargest(20)

plt.figure(figsize=(10, 8))
top_20_features.plot(kind='barh')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 20 Most Important Features')
plt.show()

# Print the features that appeared most frequently in the top 20 important features
most_common_features = feature_counter.most_common()
print("Features that appeared most frequently in the top 20 important features:")
for feature, count in most_common_features:
    print(f"{feature}: {count} times")

# -

X_test

df_resampled.columns

RC_preds = best_model.predict(df_resampled.drop(["ResidueCov"], axis=1)[top_100_features])
df["rc_preds"] = pd.Series(RC_preds)
df

merged_df = pd.merge(final_df, final_df_withTillage[['pointID', 'Tillage']], on='pointID', how='left')
last_df = merged_df.dropna(subset="Tillage", how="any")
last_df

cdl_data

last_df2 = pd.merge(last_df, cdl_data, on='pointID', how='left')
last_df2

last_df2.Tillage.value_counts()
last_df2.rename(columns={'most_frequent_class': 'Croptype'}, inplace=True)
last_df2

# +
# Load your dataframe with categorical features
df = pd.read_csv('/Users/aminnorouzi/Library/CloudStorage/OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/Tillage_Mapping/Data/field_level_data/test_df.csv')

# Perform one-hot encoding for the categorical features
df = pd.get_dummies(df, columns=['Croptype', 'ResidueCov'])

# Split features and target variable
X = df.drop(['Tillage', 'pointID', 'Unnamed: 0'], axis=1)
y = df['Tillage']

# Impute missing values with the median
X = X.fillna(X.median())

# -

X

# +
df = pd.read_csv('/Users/aminnorouzi/Library/CloudStorage/OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/Tillage_Mapping/Data/field_level_data/test_df.csv')

df.loc[df["Croptype"] == "grain"] = "Grain"
df.loc[df["Croptype"] == "legume"] = "Legume"
df.loc[df["Croptype"] == "canola"] = "Canola"

df['Croptype'].value_counts()


# +
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

# -

class CustomRFC(BaseEstimator, ClassifierMixin):
    def __init__(self, a=1, n_estimators=10, max_depth=None):
        self.a = a
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.rfc = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth)

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        target_weights_dict = calculate_custom_weights(y, self.a)
        if sample_weight is not None:
            adjusted_sample_weight = sample_weight * \
                np.array([target_weights_dict[cls] for cls in y])
        else:
            adjusted_sample_weight = np.array(
                [target_weights_dict[cls] for cls in y])

        self.rfc.fit(X, y, sample_weight=adjusted_sample_weight)
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.rfc.predict(X)



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

# Load your dataframe with categorical features
df = pd.read_csv('/Users/aminnorouzi/Library/CloudStorage/OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/Tillage_Mapping/Data/field_level_data/test_df.csv')

# Perform one-hot encoding for the categorical features
df = pd.get_dummies(df, columns=['Croptype', 'ResidueCov'])

# Split features and target variable
X = df.drop(['Tillage', 'pointID', 'Unnamed: 0'], axis=1)
y = df['Tillage']

# Impute missing values with the median
X = X.fillna(X.median())

# Verify that all missing values have been imputed
print(X.isnull().sum())



# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Perform oversampling on the training data to balance the classes
# oversampler = RandomOverSampler()
# X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

# Create a new dataframe with the resampled training data
df_resampled = pd.DataFrame(X_train, columns=X_train.columns)
df_resampled['Tillage'] = y_train


# Custom weight formula function
def calculate_custom_weights(y, a):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    weight_dict = {}
    sum_weight = np.sum((1 / class_counts) ** a)
    for cls, cnt in zip(unique_classes, class_counts):
        weight_dict[cls] = (1 / cnt) ** a / sum_weight
    return weight_dict


# Calculate weights for the target variable

target_weights_dict = calculate_custom_weights(y_train, a)
target_weights = np.array([target_weights_dict[cls] for cls in y_train])

# Calculate weights for the imbalanced feature's dummy variables
dummy_cols = [col for col in X_train.columns if 'Croptype_' in col]
feature_weights = np.zeros(X_train.shape[0])
for col in dummy_cols:
    feature_weights_dict = calculate_custom_weights(X_train[col].values, a)
    # Sum up weights for each dummy feature
    feature_weights += X_train[col].map(feature_weights_dict).values


# Combine both weights (you can also consider other combinations or functions)
sample_weights = target_weights * feature_weights

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [20, 25, 30, 40, 50, 100],
    'max_depth': [5, 10, 20, 30, 35],
    'a' : [0.5, 1, 3, 10] 
}

# Perform cross-validation for 20 times and calculate accuracies
mean_accuracies = []
best_model = None
best_val_accuracy = 0
feature_counter = Counter()  # Counter to keep track of feature occurrences

# Initialize a list to store mean test scores for each hyperparameter combination
mean_test_scores = []

for _ in range(4):
    print(_)
    grid_search = GridSearchCV(
        CustomRFC(), param_grid, cv=5, return_train_score=False)
    grid_search.fit(df_resampled.drop('Tillage', axis=1),
                    df_resampled['Tillage'], sample_weight=sample_weights)

    # Store mean test scores in the list
    mean_test_scores.append(grid_search.cv_results_['mean_test_score'])

    # Get the best model and its predictions
    current_model = grid_search.best_estimator_
    y_pred = current_model.predict(X_test)  # Use the test data for prediction

    # Calculate the accuracy for the current run
    val_accuracy = accuracy_score(y_test, y_pred)
    mean_accuracies.append(val_accuracy)

    # Update the best model if the current model has a higher validation accuracy
    if val_accuracy > best_val_accuracy:
        best_model = current_model
        best_val_accuracy = val_accuracy

    # Update the feature counter with the top 20 important features of the current model
    top_20_indices = current_model.feature_importances_.argsort()[::-1][:20]
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
# -


y_test

# +
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

# Plot the confusion matrix
# labels = ['ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed']
labels = ['NoTill-DirectSeed']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


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
loss_df = pd.merge(loss_df, df[['pointID', 'Croptype']], on='pointID', how='left')

# Define a color for each Croptype, using colorblind-friendly and harmonious colors
croptype_colors = {'legume': '#E69F00', 'canola': '#56B4E9', 'grain': '#009E73'}

# Prepare data for stacked histogram and collect sample counts
croptypes = ['legume', 'canola', 'grain']
data = [loss_df[loss_df['Croptype'] == croptype]['loss'].values for croptype in croptypes]
sample_counts = [len(d) for d in data]

# Create labels with sample counts for the legend
labels_with_counts = [f"{croptype} (n = {count})" for croptype, count in zip(croptypes, sample_counts)]

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

        # Only label the bar if its height is greater than 0
        if bar_height > 0:
            cumulative_heights[j] += bar_height  # Update the cumulative height
            ax.text(bar_center_x, cumulative_heights[j], int(cumulative_heights[j]),
                    ha='center', va='bottom', color='black')

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

# # resampling croptype and tillage

last_df2['Tillage'].isna().value_counts()

last_df2.dropna(subset='Croptype', inplace=True)

df

# +
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import OneHotEncoder

# Load your dataframe with categorical features
df = last_df2  # Assuming last_df2 contains y# ... (same code as before to load data)

# Split features and target variable
X = df.drop(['Tillage', 'pointID'], axis=1)
y = df['Tillage']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a new DataFrame for train data before encoding (This is your original train_df)
train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)

# Initialize an empty DataFrame to store the resampled data
resampled_data = pd.DataFrame()

# Get unique classes of Croptype and Tillage
unique_croptypes = train_df['Croptype'].unique()
unique_tillages = train_df['Tillage'].unique()

# Perform resampling
for croptype in unique_croptypes:
    for tillage in unique_tillages:
        subset = train_df[(train_df['Croptype'] == croptype) & (train_df['Tillage'] == tillage)]
        
        # Only resample if subset has at least one sample
        if len(subset) > 0:
            resampled_subset = resample(subset, replace=True, n_samples=500, random_state=42)
            resampled_data = pd.concat([resampled_data, resampled_subset])
        else:
            print(f"No samples available for Croptype = {croptype}, Tillage = {tillage}. Skipping resampling for this group.")

# One-hot encode the resampled data
resampled_data_encoded = pd.get_dummies(resampled_data, columns=['Croptype', 'ResidueCov'])

# Split features and target variable from the resampled data
X_resampled = resampled_data_encoded.drop(['Tillage'], axis=1)
y_resampled = resampled_data_encoded['Tillage']

# One-hot encode X_test as well for model evaluation
X_test_encoded = pd.get_dummies(X_test, columns=['Croptype', 'ResidueCov'])

# Make sure X_test_encoded has the same columns as X_resampled
missing_cols = set(X_resampled.columns) - set(X_test_encoded.columns)
for c in missing_cols:
    X_test_encoded[c] = 0
X_test_encoded = X_test_encoded[X_resampled.columns]


# Create a new dataframe with the resampled training data
df_resampled = pd.DataFrame(X_resampled)
df_resampled['Tillage'] = y_resampled

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [10, 20, 25, 30, 40, 50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20, 30, 35, 40]
}

# Perform cross-validation for 20 times and calculate accuracies
mean_accuracies = []
best_model = None
best_val_accuracy = 0
feature_counter = Counter()  # Counter to keep track of feature occurrences

# Initialize a list to store mean test scores for each hyperparameter combination
mean_test_scores = []

for _ in range(20):
    print(_)
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, return_train_score=False)
    grid_search.fit(df_resampled.drop('Tillage', axis=1), df_resampled['Tillage'])

    # Store mean test scores in the list
    mean_test_scores.append(grid_search.cv_results_['mean_test_score'])

    # Get the best model and its predictions
    current_model = grid_search.best_estimator_
    y_pred = current_model.predict(X_test_encoded)  # Use the test data for prediction

    # Calculate the accuracy for the current run
    val_accuracy = accuracy_score(y_test, y_pred)
    mean_accuracies.append(val_accuracy)

    # Update the best model if the current model has a higher validation accuracy
    if val_accuracy > best_val_accuracy:
        best_model = current_model
        best_val_accuracy = val_accuracy

    # Update the feature counter with the top 20 important features of the current model
    top_20_indices = current_model.feature_importances_.argsort()[::-1][:20]
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

# Create a confusion matrix using predictions from the best model
y_pred_best = best_model.predict(X_test_encoded)
cm = confusion_matrix(y_test, y_pred_best)

# Plot the confusion matrix
labels = ['ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
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


# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set aesthetic styles for the plot
plt.style.use('ggplot')

# Compute the zero-one loss for each sample in the test set
y_pred_best = best_model.predict(X_test_encoded)
zero_one_loss = np.where(y_test != y_pred_best, 1, 0)

# Combine the zero-one loss and original features into a new DataFrame
loss_df = pd.DataFrame({'loss': zero_one_loss, 'pointID': X_test_encoded.index})
loss_df = pd.merge(loss_df, df[['pointID', 'Croptype']], on='pointID', how='left')

# Define a color for each Croptype, using colorblind-friendly and harmonious colors
croptype_colors = {'legume': '#E69F00', 'canola': '#56B4E9', 'grain': '#009E73'}

# Prepare data for stacked histogram and collect sample counts
croptypes = ['legume', 'canola', 'grain']
data = [loss_df[loss_df['Croptype'] == croptype]['loss'].values for croptype in croptypes]
sample_counts = [len(d) for d in data]

# Create labels with sample counts for the legend
labels_with_counts = [f"{croptype} (n = {count})" for croptype, count in zip(croptypes, sample_counts)]

# Plot stacked histogram
fig, ax = plt.subplots()
n, bins, patches = ax.hist(data, bins=[-0.5, 0.5, 1.5], stacked=True, 
                           color=[croptype_colors[c] for c in croptypes],
                           edgecolor='white', linewidth=1)

# Customize labels and legend
plt.xlabel('Zero-One Loss', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks([0, 1], fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Croptype', labels=labels_with_counts, title_fontsize='13', fontsize='12')

# Add title and grid
plt.title('Histogram of Loss Across Each Croptype', fontsize=16)
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()

# -

df_resampled['Croptype_canola'].value_counts(), df_resampled['Croptype_legume'].value_counts()

# +

X_resampled
# -

X_test_encoded

grid_search.cv_results_

# +
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.neural_network import MLPClassifier

# Load your dataframe with categorical features
df = last_df2

# Perform one-hot encoding for the categorical features
df_encoded = pd.get_dummies(df, columns=['rc_preds', 'predicted_label'])

# Split features and target variable
X = df_encoded.drop(['Tillage', 'pointID', 'ResidueCov'], axis=1)
y = df_encoded['Tillage']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform oversampling on the training data to balance the classes
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

# Create a new dataframe with the resampled training data
df_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
# Create a new dataframe with the resampled training data
df_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
df_resampled = pd.concat([df_resampled, pd.Series(y_resampled, name='Tillage')], axis=1)


# Define the parameter grid for hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu'],
    'alpha': [0.001, 0.01]
}

# Perform cross-validation for 20 times and calculate accuracies
mean_accuracies = []
best_model = None
best_val_accuracy = 0
feature_counter = Counter()  # Counter to keep track of feature occurrences

for _ in range(5):
    grid_search = GridSearchCV(MLPClassifier(), param_grid, cv=3)
    grid_search.fit(df_resampled.drop('Tillage', axis=1), df_resampled['Tillage'])

    # Get the best model and its predictions
    current_model = grid_search.best_estimator_
    y_pred = current_model.predict(X_test)  # Use the test data for prediction

    # Calculate the accuracy for the current run
    val_accuracy = accuracy_score(y_test, y_pred)
    mean_accuracies.append(val_accuracy)

    # Update the best model if the current model has a higher validation accuracy
    if val_accuracy > best_val_accuracy:
        best_model = current_model
        best_val_accuracy = val_accuracy

    # Update the feature counter with the top 20 important features of the current model
    top_20_indices = current_model.coefs_[-1].argsort()[::-1][:20]
    top_20_features = X.columns[top_20_indices]
    feature_counter.update(map(str, top_20_features))



# Calculate mean accuracy across the 20 runs
mean_accuracy = sum(mean_accuracies) / len(mean_accuracies)

# Print accuracies for all cross-validations
print("Accuracies for all cross-validations:")
for i, accuracy in enumerate(mean_accuracies, 1):
    print(f"Cross-Validation {i}: {accuracy:.4f}")

# Print mean accuracy
print(f"Mean Accuracy: {mean_accuracy:.4f}")

# Create a confusion matrix using predictions from the best model
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

# Plot the confusion matrix
labels = ['ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
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

