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

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, confusion_matrix

# Split the data into X (spectral features) and y (target variable)
X = df.drop(['residue_%', 'pointID', 'Tillage'], axis=1)
y = df['residue_%']

# Split the data into train and test sets at the field level
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=df['pointID'])

# Perform oversampling on the training data
oversampler = RandomOverSampler()
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Define the classifiers
rf_classifier = RandomForestClassifier(random_state=42)
svm_classifier = SVC(random_state=42)
nn_classifier = MLPClassifier(random_state=42)

# Define the parameter grids for hyperparameter tuning
rf_param_grid = {'n_estimators': [20, 30, 40, 50, 100, 200, 300], 'max_depth': [None, 5, 10, 20, 30, 35, 40]}
svm_param_grid = {'C': [1, 10, 100], 'gamma': ['scale', 'auto']}
nn_param_grid = {'hidden_layer_sizes': [(100,), (100, 50), (50, 50)]}

# Perform hyperparameter tuning and cross-validation
rf_grid_search = GridSearchCV(rf_classifier, rf_param_grid, cv=5)
rf_grid_search.fit(X_train_resampled, y_train_resampled)
rf_best_model = rf_grid_search.best_estimator_
rf_cross_val_scores = cross_val_score(rf_best_model, X_train_resampled, y_train_resampled, cv=5)

svm_grid_search = GridSearchCV(svm_classifier, svm_param_grid, cv=5)
svm_grid_search.fit(X_train_resampled, y_train_resampled)
svm_best_model = svm_grid_search.best_estimator_
svm_cross_val_scores = cross_val_score(svm_best_model, X_train_resampled, y_train_resampled, cv=5)

nn_grid_search = GridSearchCV(nn_classifier, nn_param_grid, cv=5)
nn_grid_search.fit(X_train_resampled, y_train_resampled)
nn_best_model = nn_grid_search.best_estimator_
nn_cross_val_scores = cross_val_score(nn_best_model, X_train_resampled, y_train_resampled, cv=5)

# Print the mean accuracies for the best models achieved during cross-validation
print("Random Forest - Mean Accuracy:", np.mean(rf_cross_val_scores))
print("Support Vector Machine - Mean Accuracy:", np.mean(svm_cross_val_scores))
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

