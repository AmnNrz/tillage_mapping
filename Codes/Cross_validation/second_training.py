# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1684808856094, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="WunwGaeGbq3h"
import pandas as pd
import numpy as np

# + id="Fi5zhNsmbz3T"
# metric_based_glcmBand = pd.read_csv("H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\spectral_data\metric_based_glcmBand.csv", index_col=0)
# metric_based_mainBand = pd.read_csv("H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\spectral_data\metric_based_mainBand.csv", index_col=0)
seasonBased_main_glcm = pd.read_csv("H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\spectral_data\seasonBased_main_glcm.csv", index_col=0)
# -

seasonBased_main_glcm.columns
seasonBased_main_glcm.drop(columns=['index', 'Unnamed: 0'], inplace=True)
seasonBased_main_glcm

seasonBased_main_glcm[seasonBased_main_glcm['pointID'] == 100]

tillage_df = pd.read_csv("H:\My Drive\P.h.D_Projects\Tillage_Mapping\Codes\Data_prep\Tillage_data.csv")
tillage_df["pointID"].unique().shape

merged_df = pd.merge(seasonBased_main_glcm, tillage_df[["pointID", "Tillage"]], on='pointID', how='left')
merged_df

merged_df["system:index"]

df = merged_df[~merged_df["Tillage"].isna()].copy()
df.drop(columns="system:index", inplace=True)
df

lt20_pxl_id = df.pointID.value_counts()[df.pointID.value_counts()>20]
lt20_pxl_id
df  = df.loc[df.pointID.isin(lt20_pxl_id.index)]
df

df.pointID.value_counts()

df

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel

# Split the data into X (spectral features) and y (target variable)
X = df.drop(['residue_%', 'pointID', 'Tillage'], axis=1)
y = df['residue_%']

# Perform feature selection using Random Forest
rf_feature_selector = RandomForestClassifier(random_state=42)
rf_feature_selector.fit(X, y)
feature_importances = rf_feature_selector.feature_importances_
feature_indices = np.argsort(feature_importances)[::-1]  # Sort feature indices in descending order of importance

# Select the top 100 most important features
top_feature_indices = feature_indices[:100]
X_selected = X.iloc[:, top_feature_indices]

# Split the selected data into train and test sets at the field level
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=df['pointID'])

# Perform oversampling on the training data
oversampler = RandomOverSampler()
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Define the classifiers
rf_classifier = RandomForestClassifier(random_state=42)
nn_classifier = MLPClassifier(random_state=42)

# Define the parameter grids for hyperparameter tuning
rf_param_grid = {'n_estimators': [20, 30], 'max_depth': [5, 10, 20]}
# nn_param_grid = {'hidden_layer_sizes': [(100,), (100, 50), (50, 50)]}
nn_param_grid = {'hidden_layer_sizes': [(64,), (64, 32)]}

# Perform hyperparameter tuning and cross-validation
rf_grid_search = GridSearchCV(rf_classifier, rf_param_grid, cv=3)
rf_grid_search.fit(X_train_resampled, y_train_resampled)
rf_best_model = rf_grid_search.best_estimator_
rf_cross_val_scores = cross_val_score(rf_best_model, X_train_resampled, y_train_resampled, cv=3)

nn_grid_search = GridSearchCV(nn_classifier, nn_param_grid, cv=3)
nn_grid_search.fit(X_train_resampled, y_train_resampled)
nn_best_model = nn_grid_search.best_estimator_
nn_cross_val_scores = cross_val_score(nn_best_model, X_train_resampled, y_train_resampled, cv=3)

# Print the mean accuracies for the best models achieved during cross-validation
print("Random Forest - Mean Accuracy:", np.mean(rf_cross_val_scores))
print("Neural Network - Mean Accuracy:", np.mean(nn_cross_val_scores))

# Evaluate the best model on the test set
y_pred = rf_best_model.predict(X_test)

# Calculate accuracy based on pixel-level classification
pixel_accuracy = accuracy_score(y_test, y_pred)

# Calculate accuracy based on field-level classification
field_accuracy = 0
field_counts = X_test['pointID'].value_counts()
for pointID, count in field_counts.items():
    field_data = y_test[X_test['pointID'] == pointID]
    field_pred = y_pred[X_test['pointID'] == pointID]
    correct_count = np.sum(field_data == field_pred)
    if correct_count / count >= 0.5:
        field_accuracy += 1

field_accuracy /= len(field_counts)

# Print accuracies
print("Pixel-level Accuracy:", pixel_accuracy)
print("Field-level Accuracy:", field_accuracy)

# Plot confusion matrix for the best model on the test set
labels = ['0-15%', '16-30%', '>30%']
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
       xticklabels=labels, yticklabels=labels, xlabel='Predicted Class', ylabel='True Class')
plt.show()

# -

# Print accuracies
print("Pixel-level Accuracy:", pixel_accuracy)
print("Field-level Accuracy:", field_accuracy)

# +
# Evaluate the best model on the test set
y_pred_rf = rf_best_model.predict(X_test)
y_pred_nn = nn_best_model.predict(X_test)

# Calculate accuracy based on pixel-level classification
pixel_accuracy_rf = accuracy_score(y_test, y_pred_rf)
pixel_accuracy_nn = accuracy_score(y_test, y_pred_nn)

print(pixel_accuracy_rf, pixel_accuracy_nn)
# -

print("Best Random Forest Model Hyperparameters:")
print("n_estimators:", rf_grid_search.best_params_['n_estimators'])
print("max_depth:", rf_grid_search.best_params_['max_depth'])


cm

# Plot confusion matrix for the best model on the test set
labels = ['0-15%', '16-30%', '>30%']
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
       xticklabels=labels, yticklabels=labels, xlabel='Predicted Class', ylabel='True Class')
plt.show()

crop_pred = pd.read_csv(r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\crop_predictions_2021_2022.csv", index_col=0)

crop_pred

df

final_df = pd.merge(df, crop_pred, on='pointID', how='left')
final_df

# +
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the dataframe
df = final_df

# Splitting the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Separate features and target variable in the training set
X_train = train_df.drop(['Tillage', 'pointID'], axis=1)
y_train = train_df['Tillage']

# Separate features and target variable in the test set
X_test = test_df.drop(['Tillage', 'pointID'], axis=1)
y_test = test_df['Tillage']

# Combine the datasets
combined = pd.concat([X_train, X_test], axis=0)

# One-hot encode the combined dataset
combined_encoded = pd.get_dummies(combined, columns=['residue_%', 'predicted_label'])

# Split the data back into train and test datasets
X_train_encoded = combined_encoded.iloc[:len(X_train), :]
X_test_encoded = combined_encoded.iloc[len(X_train):, :]

# Perform oversampling on the training set
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_encoded, y_train)

# Create a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42, max_depth=20, n_estimators=30 )

# Perform cross-validation on the resampled training set
cv_scores = cross_val_score(rf_model, X_train_resampled, y_train_resampled, cv=5)

# Print mean train and validation accuracies
print("Mean Train Accuracy:", cv_scores.mean())
print("Mean Validation Accuracy:", cv_scores.mean())

# Fit the Random Forest model on the resampled training set
rf_model.fit(X_train_resampled, y_train_resampled)


# Predict the target variable on the test set
y_pred = rf_model.predict(X_test_encoded)

# Calculate and print the accuracy of the best model on the test set
test_accuracy = (y_pred == y_test).mean()
print("Test Accuracy:", test_accuracy)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


# +
print(sorted(y_train.unique()))
class_labels = ['ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", yticklabels=class_labels, xticklabels=class_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


# +
# Find the indices of the misclassified points
misclassified_indices = y_test[y_pred != y_test].index

# Extract the misclassified points and their features from X_test
misclassified_data = X_test.loc[misclassified_indices]

misclassified_data


# +
# Find the indices of the misclassified points
misclassified_indices = y_test[y_pred != y_test].index

# Extract the misclassified points and their features from X_test_encoded
misclassified_data = X_test_encoded.loc[misclassified_indices]

# Add the true labels to the dataframe
misclassified_data['True_Labels'] = y_test.loc[misclassified_indices]

# Add the predicted labels to the dataframe
misclassified_data['Predicted_Labels'] = y_pred[y_pred != y_test]

misclassified_data

# -

misclassified_data[misclassified_data['True_Labels']=='NoTill-DirectSeed'].shape
