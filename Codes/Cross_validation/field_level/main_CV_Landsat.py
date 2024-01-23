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

final_df = pd.read_csv(r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\field_level_data\final_dataframe_landsat.csv")
final_df_withTillage = pd.read_csv(r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\field_level_data\finalCleaned.csv")
crop_pred = pd.read_csv(r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\crop_predictions_2021_2022.csv", index_col=0)
# -

s1 = final_df.pointID.reset_index(drop=True)
s2 = final_df_withTillage.pointID.reset_index(drop=True)

s2.isin(s1).value_counts()

final_df

# +
# Impute missing values with the median
final_df = final_df.fillna(final_df.median())

# Verify that all missing values have been imputed
print(final_df.isnull().sum())

# -

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
    'max_depth': [5, 10, 20, 30, 35, 40]
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


# +
import matplotlib.pyplot as plt
import seaborn as sns

# Increase the figure size and resolution
plt.figure(figsize=(10, 8), dpi=150)

# Plot the confusion matrix
labels = ['0-15%', '16-30%', '>30%']
ax = sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"fontsize": 18})

# Increase the font size of the numbers and labels
ax.set_xticklabels(labels, fontsize=14)
ax.set_yticklabels(labels, fontsize=14)

plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)

plt.show()


# +
import matplotlib.pyplot as plt
import seaborn as sns

# Increase the figure size and resolution
plt.figure(figsize=(10, 8), dpi=150)

# Plot the confusion matrix
labels = ['0-15%', '16-30%', '>30%']
ax = sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"fontsize": 18})

# Increase the font size of the numbers and labels
ax.set_xticklabels(labels, fontsize=14)
ax.set_yticklabels(labels, fontsize=14)

plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)

plt.show()

# -

RC_preds = best_model.predict(df_resampled.drop(["ResidueCov"], axis=1)[top_100_features])
df["rc_preds"] = pd.Series(RC_preds)
df

merged_df = pd.merge(df, final_df_withTillage[['pointID', 'Tillage']], on='pointID', how='left')
last_df = merged_df.dropna(subset="Tillage", how="any")
last_df

last_df2 = pd.merge(last_df, crop_pred, on='pointID', how='left')
last_df2[last_df2['predicted_label'] ==  'legume']

last_df2.Tillage.value_counts()

last_df2.rename({"predicted_label":"Residue_type"}, inplace=True, axis=1)
last_df2

last_df2 = last_df2.set_index("pointID")
last_df2

# +
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# Load your dataframe with categorical features
df = last_df2

# Perform one-hot encoding for the categorical features
df_encoded = pd.get_dummies(df, columns=['rc_preds', 'Residue_type'])

# Split features and target variable
X = df_encoded.drop(['Tillage', 'ResidueCov'], axis=1)
y = df_encoded['Tillage']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform oversampling on the training data to balance the classes
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

# Create a new dataframe with the resampled training data
df_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
df_resampled['Tillage'] = y_resampled

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [10, 20, 25, 30, 40, 50, 100, 200, 300],
    'max_depth': [5, 10, 20, 30, 35, 40]
}

# Perform cross-validation for 20 times and calculate accuracies
mean_accuracies = []
best_model = None
best_val_accuracy = 0
feature_counter = Counter()  # Counter to keep track of feature occurrences

for _ in range(10):
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
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

# -

cm2 = cm1 =np.array([[42,  8,  3],
                     [ 5, 25,  9],
                     [ 2,  6,  3]])

# +
import matplotlib.pyplot as plt
import seaborn as sns

# Increase the figure size and resolution
plt.figure(figsize=(10, 8), dpi=150)

# Plot the confusion matrix
labels = ['ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed']
ax = sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"fontsize": 20})

# Increase the font size of the numbers and labels
ax.set_xticklabels(labels, fontsize=14)
ax.set_yticklabels(labels, fontsize=14)

plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)

plt.show()

# -

last_df2.Residue_type.value_counts()

# y_test, y_pred_best
X_test

y_pred_best_df = pd.DataFrame(y_pred_best)
misClassified_y_best_pred = (~(pd.Series(y_pred_best) == y_test.values))[(~(pd.Series(y_pred_best) == y_test.values)) == True]
misClassified_y_best_pred.index
y_pred_best_df.loc[misClassified_y_best_pred.index].values

y_test

y_test.loc[~(y_test == y_pred_best)]

final_df_withTillage.query("pointID == 102")

y_test

last_df2.query("pointID == 102")

error_df = X_test.loc[list(y_test.loc[~(y_test == y_pred_best)].index.values)]
error_df["y_test_er"] = y_test.loc[~(y_test == y_pred_best)]
error_df["y_best_pred_er"] = y_pred_best_df.loc[misClassified_y_best_pred.index].values
error_df

X_test.Residue_type_legume.value_counts()

error_df["pointID"] = error_df.index.values
error_df.reset_index(inplace=True, drop = True)
error_df

errors_srv_df[["pointID","PriorCropT", "ResidueCov", "WhereInRan", "Tillage"]]

errors_srv_df = final_df_withTillage.loc[final_df_withTillage['pointID'].isin(error_df.pointID)]
err_df_2 = pd.merge(error_df, errors_srv_df[["pointID","PriorCropT", "ResidueCov", "WhereInRan", "Tillage"]], on="pointID", how="right")
err_df_2.iloc[:, -13:]


err_df_2.iloc[:, -13:].to_csv(r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Codes\Cross_validation\field_level\errors.csv")

print(error_df.columns)
print(errors_srv_df.columns)

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

