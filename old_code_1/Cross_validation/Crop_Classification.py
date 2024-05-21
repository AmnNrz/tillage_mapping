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
#     display_name: tillenv
#     language: python
#     name: python3
# ---

import pandas as pd 
import numpy as np

ts_19_20 = pd.read_csv("H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\smoothed_df_2019_2020_canola.csv", index_col= 0)
ts_18_19 = pd.read_csv("H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\smoothed_df_2018_2019_canola.csv", index_col= 0)
ts_17_18 = pd.read_csv("H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\smoothed_df_2017_2018_canola.csv", index_col= 0)
ts_16_17 = pd.read_csv("H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\smoothed_df_2016_2017.csv", index_col= 0)
ts_15_16 = pd.read_csv("H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\smoothed_df_2015_2016.csv", index_col= 0)
ts_14_15 = pd.read_csv("H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\smoothed_df_2014_2015.csv", index_col= 0)
ts_13_14 = pd.read_csv("H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\smoothed_df_2013_2014_canola.csv", index_col= 0)
ts_12_13 = pd.read_csv("H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\smoothed_df_2012_2013_canola.csv", index_col= 0)
ts_11_12 = pd.read_csv("H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\smoothed_df_2011_2012_canola.csv", index_col= 0)
ts_10_11 = pd.read_csv("H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\smoothed_df_2010_2011.csv", index_col= 0)
TS_merged = pd.concat([ts_10_11, ts_11_12, ts_12_13, ts_13_14, ts_14_15, 
                       ts_15_16, ts_16_17, ts_17_18, ts_18_19, ts_19_20])
TS_merged.reset_index(inplace=True, drop=True)
TS_merged.drop(columns='EVI', inplace=True)
TS_merged

df_2013_2014 = pd.read_excel(r'H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\GIS_Data\whole_WA\2014_2015_newSelection.xls')
df_2013_2014['CropType'].value_counts()

ts_15_16.pointID.value_counts().value_counts()

# +
selected_crops = ['Wheat', 'Canola', 'Wheat Fallow', 'Pea, Dry', 
                  'Barley', 'Chickpea', 'Lentil', 'Chickpea', 'Pea, Green', 'Triticale']
selectedCrops_df = TS_merged.copy().loc[TS_merged['CropType'].isin(selected_crops)]
selectedCrops_df['CropType'].value_counts()
value_mapping = {
    'Wheat': 'grain', 'Wheat Fallow':'grain', 'Pea, Dry':'legume', 
                  'Barley':'grain', 'Chickpea':'legume', 'Lentil':'legume',
                'Chickpea':'legume', 'Pea, Green':'legume', 'Triticale':'grain'
}

selectedCrops_df['CropType'] = selectedCrops_df['CropType'].replace(value_mapping)
selectedCrops_df['CropType'].value_counts()
# -



pointid_counts = selectedCrops_df.loc[selectedCrops_df["CropType"] == "Canola"]["pointID"].value_counts()
less_than_30 = pointid_counts[pointid_counts < 30].index.values
less_than_30

selectedCrops_df = selectedCrops_df.loc[~(selectedCrops_df.pointID.isin(less_than_30))]
selectedCrops_df.pointID.value_counts()

selectedCrops_df.pointID

selectedCrops_df.reset_index(inplace=True)

selectedCrops_df

selectedCrops_df['Unique_ID'] = (selectedCrops_df.index // 30) + 1

selectedCrops_df.Unique_ID.value_counts()

selectedCrops_df.iloc[:, 3:9]

selectedCrops_df.CropType

# selectedCrops_df.to_csv(r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\selectedCrops_df.csv")
df = pd.read_csv(r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\selectedCrops_df.csv")

df.CropType.value_counts()

id_counts = df['pointID'].value_counts()
id_counts[id_counts == 60].index

df.iloc[:, 4:10]

df.Unique_ID.value_counts().value_counts()

# # With overSampling

# +
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler
import os
from sklearn.metrics import confusion_matrix

# Load your dataset into a DataFrame (assuming it's named 'df')
# Ensure your DataFrame has columns for features (6 observations), target class, and ID

# Convert the target class to numeric labels
class_mapping = {'grain': 0, 'legume': 1, 'Canola': 2}
df['Target'] = df['CropType'].map(class_mapping)

# Separate features, target, and ID
features = df.iloc[:, 4:10].values  # Assuming your features are in columns 2 to 8
target = df['CropType'].values
ids = df['Unique_ID'].values

# Find unique IDs in the DataFrame
unique_ids = np.unique(ids)

# Prepare lists to store aggregated data
aggregated_features = []
aggregated_target = []

# Aggregate features and target based on unique IDs
for unique_id in unique_ids:
    id_indices = np.where(ids == unique_id)[0]
    id_features = features[id_indices]
    id_target = target[id_indices[0]]  # Assuming the target is the same for all rows with the same ID
    
    aggregated_features.append(id_features)
    aggregated_target.append(id_target)

# Convert aggregated data to numpy arrays
aggregated_features = np.array(aggregated_features)
aggregated_target = np.array([class_mapping[label] for label in aggregated_target])  # Convert target labels to numeric labels

# Reshape aggregated_features to (n_samples, n_observations)
n_samples, n_observations, n_features = aggregated_features.shape
aggregated_features_reshaped = np.reshape(aggregated_features, (n_samples, n_observations * n_features))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(aggregated_features_reshaped, aggregated_target, test_size=0.2, random_state=42)

# Perform oversampling on the training set only
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Convert target labels to categorical format
y_train_categorical = to_categorical(y_train_resampled)
y_test_categorical = to_categorical(y_test)

# Reshape the input data to match the LSTM input shape (samples, timesteps, features)
input_shape = (n_observations, n_features)
X_train_resampled = np.reshape(X_train_resampled, (X_train_resampled.shape[0], *input_shape))
X_test = np.reshape(X_test, (X_test.shape[0], *input_shape))

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=input_shape))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # Assuming you have 3 target classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the number of folds for cross-validation
num_folds = 5

# Initialize lists to store performance metrics for each fold
fold_train_losses = []
fold_train_accs = []
fold_val_losses = []
fold_val_accs = []
fold_cmats = []

# Perform k-fold cross-validation
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
for fold, (train_index, val_index) in enumerate(skf.split(X_train_resampled, np.argmax(y_train_categorical, axis=1))):
    print(f'Fold: {fold + 1}/{num_folds}')

    X_train_fold, X_val_fold = X_train_resampled[train_index], X_train_resampled[val_index]
    y_train_fold, y_val_fold = y_train_categorical[train_index], y_train_categorical[val_index]

    # Train the model
    history = model.fit(X_train_fold, y_train_fold, epochs=40, batch_size=32, validation_data=(X_val_fold, y_val_fold))

    # Evaluate the model on training and validation data for this fold
    fold_train_losses.append(history.history['loss'][-1])
    fold_train_accs.append(history.history['accuracy'][-1])
    fold_val_losses.append(history.history['val_loss'][-1])
    fold_val_accs.append(history.history['val_accuracy'][-1])

    # Make predictions on validation data for this fold
    y_val_pred_prob = model.predict(X_val_fold)
    y_val_pred = np.argmax(y_val_pred_prob, axis=1)
    cmat = confusion_matrix(np.argmax(y_val_fold, axis=1), y_val_pred)
    fold_cmats.append(cmat)

# Calculate mean performance metrics across folds
mean_train_loss = np.mean(fold_train_losses)
mean_train_acc = np.mean(fold_train_accs)
mean_val_loss = np.mean(fold_val_losses)
mean_val_acc = np.mean(fold_val_accs)

# Print mean performance metrics
print(f'Mean Train Loss: {mean_train_loss}')
print(f'Mean Train Accuracy: {mean_train_acc}')
print(f'Mean Validation Loss: {mean_val_loss}')
print(f'Mean Validation Accuracy: {mean_val_acc}')

# Find the best model based on the mean validation accuracy
best_fold = np.argmax(fold_val_accs)
best_model = model

# Evaluate the best model on the test set
test_loss, test_acc = best_model.evaluate(X_test, y_test_categorical)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

# Make predictions on the test set
y_test_pred_prob = best_model.predict(X_test)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)

# Calculate confusion matrix for the test set
test_cmat = confusion_matrix(np.argmax(y_test_categorical, axis=1), y_test_pred)
print(f'Confusion Matrix for Test Set:\n{test_cmat}')

# Save the best model
save_dir = r'H:\My Drive\P.h.D_Projects\Tillage_Mapping\Codes\Crop_Classification'  # Provide the path where you want to save the model
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
best_model.save(os.path.join(save_dir, 'best_model.h5'))


# +
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
import os
from sklearn.metrics import confusion_matrix

# Load your dataset into a DataFrame (assuming it's named 'df')
# Ensure your DataFrame has columns for features (6 observations), target class, and ID

# Convert the target class to numeric labels
class_mapping = {'grain': 0, 'legume': 1, 'Canola': 2}
df['Target'] = df['CropType'].map(class_mapping)

# Separate features, target, and ID
features = df.iloc[:, 4:10].values  # Assuming your features are in columns 2 to 8
target = df['CropType'].values
ids = df['Unique_ID'].values

# Find unique IDs in the DataFrame
unique_ids = np.unique(ids)

# Prepare lists to store aggregated data
aggregated_features = []
aggregated_target = []

# Aggregate features and target based on unique IDs
for unique_id in unique_ids:
    id_indices = np.where(ids == unique_id)[0]
    id_features = features[id_indices]
    id_target = target[id_indices[0]]  # Assuming the target is the same for all rows with the same ID
    
    aggregated_features.append(id_features)
    aggregated_target.append(id_target)

# Convert aggregated data to numpy arrays
aggregated_features = np.array(aggregated_features)
aggregated_target = np.array([class_mapping[label] for label in aggregated_target])  # Convert target labels to numeric labels

# Reshape aggregated_features to (n_samples, n_observations)
n_samples, n_observations, n_features = aggregated_features.shape
aggregated_features_reshaped = np.reshape(aggregated_features, (n_samples, n_observations * n_features))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(aggregated_features_reshaped, aggregated_target, test_size=0.2, random_state=42)

# Perform oversampling on the training set only
oversampler = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Convert target labels to categorical format
y_train_categorical = to_categorical(y_train_resampled)
y_test_categorical = to_categorical(y_test)

# Reshape the input data to match the LSTM input shape (samples, timesteps, features)
input_shape = (n_observations, n_features)
X_train_resampled = np.reshape(X_train_resampled, (X_train_resampled.shape[0], *input_shape))
X_test = np.reshape(X_test, (X_test.shape[0], *input_shape))

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=input_shape))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # Assuming you have 3 target classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the number of folds for cross-validation
num_folds = 5

# Initialize lists to store performance metrics for each fold
fold_train_losses = []
fold_train_accs = []
fold_val_losses = []
fold_val_accs = []
fold_cmats = []

# Perform k-fold cross-validation
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
for fold, (train_index, val_index) in enumerate(skf.split(X_train_resampled, np.argmax(y_train_categorical, axis=1))):
    print(f'Fold: {fold + 1}/{num_folds}')

    X_train_fold, X_val_fold = X_train_resampled[train_index], X_train_resampled[val_index]
    y_train_fold, y_val_fold = y_train_categorical[train_index], y_train_categorical[val_index]

    # Train the model
    history = model.fit(X_train_fold, y_train_fold, epochs=40, batch_size=32, validation_data=(X_val_fold, y_val_fold))

    # Evaluate the model on training and validation data for this fold
    fold_train_losses.append(history.history['loss'][-1])
    fold_train_accs.append(history.history['accuracy'][-1])
    fold_val_losses.append(history.history['val_loss'][-1])
    fold_val_accs.append(history.history['val_accuracy'][-1])

    # Make predictions on validation data for this fold
    y_val_pred_prob = model.predict(X_val_fold)
    y_val_pred = np.argmax(y_val_pred_prob, axis=1)
    cmat = confusion_matrix(np.argmax(y_val_fold, axis=1), y_val_pred)
    fold_cmats.append(cmat)

# Calculate mean performance metrics across folds
mean_train_loss = np.mean(fold_train_losses)
mean_train_acc = np.mean(fold_train_accs)
mean_val_loss = np.mean(fold_val_losses)
mean_val_acc = np.mean(fold_val_accs)

# Print mean performance metrics
print(f'Mean Train Loss: {mean_train_loss}')
print(f'Mean Train Accuracy: {mean_train_acc}')
print(f'Mean Validation Loss: {mean_val_loss}')
print(f'Mean Validation Accuracy: {mean_val_acc}')

# Find the best model based on the mean validation accuracy
best_fold = np.argmax(fold_val_accs)
best_model = model

# Evaluate the best model on the test set
test_loss, test_acc = best_model.evaluate(X_test, y_test_categorical)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

# Make predictions on the test set
y_test_pred_prob = best_model.predict(X_test)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)

# Calculate confusion matrix for the test set
test_cmat = confusion_matrix(np.argmax(y_test_categorical, axis=1), y_test_pred)
print(f'Confusion Matrix for Test Set:\n{test_cmat}')

# Save the best model
save_dir = r'H:\My Drive\P.h.D_Projects\Tillage_Mapping\Codes\Crop_Classification'  # Provide the path where you want to save the model
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
best_model.save(os.path.join(save_dir, 'best_model.h5'))


# +
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the class labels
class_labels = ['grain', 'legume', 'Canola']

# Generate the confusion matrix
confusion_mat = test_cmat  # Use the confusion matrix for the best model

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Create the heatmap
heatmap = sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                      xticklabels=class_labels, yticklabels=class_labels, ax=ax)

# Add row counts to the heatmap cells
for i in range(confusion_mat.shape[0]):
    for j in range(confusion_mat.shape[1]):
        count = confusion_mat[i, j]
        text = f'{count}\n({count / np.sum(confusion_mat[i]) * 100:.2f}%)'
        ax.text(j + 0.5, i + 0.5, text,
                ha='center', va='center', color='black', fontsize=12,
                bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))

# Set axis labels and title
ax.set_xlabel('Predicted Class')
ax.set_ylabel('True Class')
ax.set_title('Confusion Matrix')

# Rotate the x-axis labels for better visibility
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()

# -

# # More regularized

# +
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from imblearn.over_sampling import RandomOverSampler
# import os
# from sklearn.metrics import confusion_matrix

# # Load your dataset into a DataFrame (assuming it's named 'df')
# # Ensure your DataFrame has columns for features (6 observations), target class, and ID

# # Convert the target class to numeric labels
# class_mapping = {'grain': 0, 'legume': 1, 'Canola': 2}
# df['Target'] = df['CropType'].map(class_mapping)

# # Separate features, target, and ID
# features = df.iloc[:, 4:10].values  # Assuming your features are in columns 2 to 8
# target = df['CropType'].values
# ids = df['Unique_ID'].values

# # Find unique IDs in the DataFrame
# unique_ids = np.unique(ids)

# # Prepare lists to store aggregated data
# aggregated_features = []
# aggregated_target = []

# # Aggregate features and target based on unique IDs
# for unique_id in unique_ids:
#     id_indices = np.where(ids == unique_id)[0]
#     id_features = features[id_indices]
#     id_target = target[id_indices[0]]  # Assuming the target is the same for all rows with the same ID
    
#     aggregated_features.append(id_features)
#     aggregated_target.append(id_target)

# # Convert aggregated data to numpy arrays
# aggregated_features = np.array(aggregated_features)
# aggregated_target = np.array([class_mapping[label] for label in aggregated_target])  # Convert target labels to numeric labels

# # Reshape aggregated_features to (n_samples, n_observations)
# n_samples, n_observations, n_features = aggregated_features.shape
# aggregated_features_reshaped = np.reshape(aggregated_features, (n_samples, n_observations * n_features))

# # Split the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(aggregated_features_reshaped, aggregated_target, test_size=0.2, random_state=42)

# # Perform oversampling on the training set only
# oversampler = RandomOverSampler(random_state=42)
# X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# # Convert target labels to categorical format
# y_train_categorical = to_categorical(y_train_resampled)
# y_test_categorical = to_categorical(y_test)

# # Reshape the input data to match the LSTM input shape (samples, timesteps, features)
# input_shape = (n_observations, n_features)
# X_train_resampled = np.reshape(X_train_resampled, (X_train_resampled.shape[0], *input_shape))
# X_test = np.reshape(X_test, (X_test.shape[0], *input_shape))

# from tensorflow.keras.regularizers import l1, l2

# # Define the LSTM model with L1 and L2 regularization
# model = Sequential()
# model.add(LSTM(64, input_shape=input_shape, kernel_regularizer=l1(0.001), recurrent_regularizer=l2(0.001)))
# model.add(Dropout(0.5))
# model.add(Dense(3, activation='softmax'))  # Assuming you have 3 target classes

# # Compile the model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Define the number of folds for cross-validation
# num_folds = 5

# # Initialize lists to store performance metrics for each fold
# fold_train_losses = []
# fold_train_accs = []
# fold_val_losses = []
# fold_val_accs = []
# fold_cmats = []

# # Perform k-fold cross-validation
# skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
# for fold, (train_index, val_index) in enumerate(skf.split(X_train_resampled, np.argmax(y_train_categorical, axis=1))):
#     print(f'Fold: {fold + 1}/{num_folds}')

#     X_train_fold, X_val_fold = X_train_resampled[train_index], X_train_resampled[val_index]
#     y_train_fold, y_val_fold = y_train_categorical[train_index], y_train_categorical[val_index]

#     # Train the model
#     history = model.fit(X_train_fold, y_train_fold, epochs=40, batch_size=32, validation_data=(X_val_fold, y_val_fold))

#     # Evaluate the model on training and validation data for this fold
#     fold_train_losses.append(history.history['loss'][-1])
#     fold_train_accs.append(history.history['accuracy'][-1])
#     fold_val_losses.append(history.history['val_loss'][-1])
#     fold_val_accs.append(history.history['val_accuracy'][-1])

#     # Make predictions on validation data for this fold
#     y_val_pred_prob = model.predict(X_val_fold)
#     y_val_pred = np.argmax(y_val_pred_prob, axis=1)
#     cmat = confusion_matrix(np.argmax(y_val_fold, axis=1), y_val_pred)
#     fold_cmats.append(cmat)

# # Calculate mean performance metrics across folds
# mean_train_loss = np.mean(fold_train_losses)
# mean_train_acc = np.mean(fold_train_accs)
# mean_val_loss = np.mean(fold_val_losses)
# mean_val_acc = np.mean(fold_val_accs)

# # Print mean performance metrics
# print(f'Mean Train Loss: {mean_train_loss}')
# print(f'Mean Train Accuracy: {mean_train_acc}')
# print(f'Mean Validation Loss: {mean_val_loss}')
# print(f'Mean Validation Accuracy: {mean_val_acc}')

# # Find the best model based on the mean validation accuracy
# best_fold = np.argmax(fold_val_accs)
# best_model = model

# # Evaluate the best model on the test set
# test_loss, test_acc = best_model.evaluate(X_test, y_test_categorical)
# print(f'Test Loss: {test_loss}')
# print(f'Test Accuracy: {test_acc}')

# # Make predictions on the test set
# y_test_pred_prob = best_model.predict(X_test)
# y_test_pred = np.argmax(y_test_pred_prob, axis=1)

# # Calculate confusion matrix for the test set
# test_cmat = confusion_matrix(np.argmax(y_test_categorical, axis=1), y_test_pred)
# print(f'Confusion Matrix for Test Set:\n{test_cmat}')

# # Save the best model
# save_dir = r'H:\My Drive\P.h.D_Projects\Tillage_Mapping\Codes\Crop_Classification'  # Provide the path where you want to save the model
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# best_model.save(os.path.join(save_dir, 'best_model.h5'))


# +
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# predict of 2021_2022 survey fields
timeSeries_2021_2022 = pd.read_csv(r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\smoothed_df_2021_2022_survey_pols.csv", index_col=0)
timeSeries_2021_2022

# Load your new dataset into a DataFrame (assuming it's named 'new_df')
# Ensure your DataFrame has the same structure as the previous dataset, with the same column names for the features
# 'pointID' column represents the identical time-series observations of a field

# Preprocess the new dataset
new_features = timeSeries_2021_2022.iloc[:, 2:].values  # Assuming the features start from the second column
new_ids = timeSeries_2021_2022['pointID'].values

# Find unique IDs in the DataFrame
unique_ids = np.unique(new_ids)

# Prepare lists to store aggregated data
aggregated_new_features = []

# Aggregate features based on unique IDs
for unique_id in unique_ids:
    id_indices = np.where(new_ids == unique_id)[0]
    id_features = new_features[id_indices]  # Assuming the features are the same for all rows with the same ID
    
    aggregated_new_features.append(id_features)

# Convert aggregated data to numpy array
aggregated_new_features = np.array(aggregated_new_features)

# Reshape the features to match the LSTM input shape (samples, timesteps, features)
new_input_shape = (n_observations, n_features)  # Use the same input shape as the previous dataset
aggregated_new_features_reshaped = np.reshape(aggregated_new_features, (aggregated_new_features.shape[0], *new_input_shape))

# Load the best model
best_model = load_model(r'H:\My Drive\P.h.D_Projects\Tillage_Mapping\Codes\Crop_Classification\best_model.h5')  # Provide the path to the saved best model

# Make predictions on the new dataset
new_predictions = best_model.predict(aggregated_new_features_reshaped)

# Convert predictions to class labels
class_labels = ['grain', 'legume', 'Canola']
predicted_labels = [class_labels[np.argmax(prediction)] for prediction in new_predictions]

# Print the DataFrame with predicted labels
print(predicted_labels)


# +
# Create a DataFrame with predicted_label and pointID columns
df_predictions = pd.DataFrame({'predicted_label': predicted_labels, 'pointID': timeSeries_2021_2022['pointID'].unique()})

# Print the DataFrame
df_predictions

# Save the predictions 
df_predictions.to_csv(r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\crop_predictions_2021_2022.csv")

# -

timeSeries_2021_2022.pointID.unique()

# # Without overSampling

# +
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler
import os
from sklearn.metrics import confusion_matrix

# Load your dataset into a DataFrame (assuming it's named 'df')
# Ensure your DataFrame has columns for features (6 observations), target class, and ID

# Convert the target class to numeric labels
class_mapping = {'grain': 0, 'legume': 1, 'Canola': 2}
df['Target'] = df['CropType'].map(class_mapping)

# Separate features, target, and ID
features = df.iloc[:, 4:10].values  # Assuming your features are in columns 2 to 8
target = df['CropType'].values
ids = df['Unique_ID'].values

# Find unique IDs in the DataFrame
unique_ids = np.unique(ids)

# Prepare lists to store aggregated data
aggregated_features = []
aggregated_target = []

# Aggregate features and target based on unique IDs
for unique_id in unique_ids:
    id_indices = np.where(ids == unique_id)[0]
    id_features = features[id_indices]
    id_target = target[id_indices[0]]  # Assuming the target is the same for all rows with the same ID
    
    aggregated_features.append(id_features)
    aggregated_target.append(id_target)

# Convert aggregated data to numpy arrays
aggregated_features = np.array(aggregated_features)
aggregated_target = np.array([class_mapping[label] for label in aggregated_target])  # Convert target labels to numeric labels

# Reshape aggregated_features to (n_samples, n_observations)
n_samples, n_observations, n_features = aggregated_features.shape
aggregated_features_reshaped = np.reshape(aggregated_features, (n_samples, n_observations * n_features))

# # Perform oversampling to address class imbalance
# oversampler = RandomOverSampler(random_state=42)
# aggregated_features_reshaped, aggregated_target = oversampler.fit_resample(aggregated_features_reshaped, aggregated_target)

# Convert target labels to categorical format
aggregated_target_categorical = to_categorical(aggregated_target)

# Reshape the input data to match the LSTM input shape (samples, timesteps, features)
input_shape = (n_observations, n_features)
aggregated_features_reshaped = np.reshape(aggregated_features_reshaped, (aggregated_features_reshaped.shape[0], *input_shape))

# Split the data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(aggregated_features_reshaped, aggregated_target_categorical, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=input_shape))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # Assuming you have 3 target classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the number of folds for cross-validation
num_folds = 5

# Initialize lists to store performance metrics for each fold
fold_train_losses = []
fold_train_accs = []
fold_val_losses = []
fold_val_accs = []
fold_cmats = []

# Perform k-fold cross-validation
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
for fold, (train_index, val_index) in enumerate(skf.split(X_train, np.argmax(y_train, axis=1))):
    print(f'Fold: {fold + 1}/{num_folds}')

    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Train the model
    history = model.fit(X_train_fold, y_train_fold, epochs=40, batch_size=32, validation_data=(X_val_fold, y_val_fold))

    # Evaluate the model on training and validation data for this fold
    fold_train_losses.append(history.history['loss'][-1])
    fold_train_accs.append(history.history['accuracy'][-1])
    fold_val_losses.append(history.history['val_loss'][-1])
    fold_val_accs.append(history.history['val_accuracy'][-1])

    # Make predictions on validation data for this fold
    y_val_pred_prob = model.predict(X_val_fold)
    y_val_pred = np.argmax(y_val_pred_prob, axis=1)
    cmat = confusion_matrix(np.argmax(y_val_fold, axis=1), y_val_pred)
    fold_cmats.append(cmat)

# Calculate mean performance metrics across folds
mean_train_loss = np.mean(fold_train_losses)
mean_train_acc = np.mean(fold_train_accs)
mean_val_loss = np.mean(fold_val_losses)
mean_val_acc = np.mean(fold_val_accs)

# Print mean performance metrics
print(f'Mean Train Loss: {mean_train_loss}')
print(f'Mean Train Accuracy: {mean_train_acc}')
print(f'Mean Validation Loss: {mean_val_loss}')
print(f'Mean Validation Accuracy: {mean_val_acc}')

# Find the best model based on the mean validation accuracy
best_fold = np.argmax(fold_val_accs)
best_model = model

# Evaluate the best model on the test set
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

# Make predictions on the test set
y_test_pred_prob = best_model.predict(X_test)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)

# Calculate confusion matrix for the test set
test_cmat = confusion_matrix(np.argmax(y_test, axis=1), y_test_pred)
print(f'Confusion Matrix for Test Set:\n{test_cmat}')

# Save the best model
save_dir = r'H:\My Drive\P.h.D_Projects\Tillage_Mapping\Codes\Crop_Classification'  # Provide the path where you want to save the model
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
best_model.save(os.path.join(save_dir, 'best_model_withoutOverSampling.h5'))


# +
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the class labels
class_labels = ['grain', 'legume', 'Canola']

# Generate the confusion matrix
confusion_mat = test_cmat  # Use the confusion matrix for the best model

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Create the heatmap
heatmap = sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                      xticklabels=class_labels, yticklabels=class_labels, ax=ax)

# Add row counts to the heatmap cells
for i in range(confusion_mat.shape[0]):
    for j in range(confusion_mat.shape[1]):
        count = confusion_mat[i, j]
        text = f'{count}\n({count / np.sum(confusion_mat[i]) * 100:.2f}%)'
        ax.text(j + 0.5, i + 0.5, text,
                ha='center', va='center', color='black', fontsize=12,
                bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))

# Set axis labels and title
ax.set_xlabel('Predicted Class')
ax.set_ylabel('True Class')
ax.set_title('Confusion Matrix')

# Rotate the x-axis labels for better visibility
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()

