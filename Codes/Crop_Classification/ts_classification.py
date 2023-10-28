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

# + executionInfo={"elapsed": 8884, "status": "ok", "timestamp": 1684857746371, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="XnQl4e5NC8Wi"
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler

# + executionInfo={"elapsed": 2145, "status": "ok", "timestamp": 1684857748510, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="1YQSd_X0E4VL"
path = (
    "H:\My Drive\P.h.D_Projects\Tillage_Mapping"
    "\Data\Crop_type_TS\selectedCrops_df.csv"
)

df = pd.read_csv(path, index_col=0)
df

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 246, "status": "ok", "timestamp": 1684857748750, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="pHE7uJf0FeW_" outputId="3a48bc90-34fc-4904-a04b-53a37bdf291a"
df["swi2"].isna().any()
df = df.loc[~df["CropType"].isna()]
df["CropType"].isna().any()
df["CropType"].value_counts()

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 108616, "status": "ok", "timestamp": 1684857857363, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="BwrB7aF9E3Nz" outputId="75698d9b-1c49-4975-9fe9-90eeeaa9324a"


# Load your dataset into a DataFrame (assuming it's named 'df')
# Ensure your DataFrame has columns for features (6 observations), target class, and ID

# Convert the target class to numeric labels
class_mapping = {'grain': 0, 'legume': 1, 'Canola': 2}
df['CropType'] = df['CropType'].map(class_mapping)

# Separate features, target, and ID
# Assuming your features are in columns 2 to 8
features = df.iloc[:, 2:8].values
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
    if len(id_indices) >= 30:  # Only consider IDs with at least 30 rows
        id_features = features[id_indices[:30]]
        # Assuming the target is the same for all rows with the same ID
        id_target = target[id_indices[0]]

        aggregated_features.append(id_features)
        aggregated_target.append(id_target)

# Convert aggregated data to numpy arrays
aggregated_features = np.array(aggregated_features)
aggregated_target = np.array(aggregated_target)

# Reshape aggregated_features to (n_samples, n_observations)
n_samples, n_observations, n_features = aggregated_features.shape
aggregated_features_reshaped = np.reshape(
    aggregated_features, (n_samples, n_observations * n_features))

# Perform oversampling to address class imbalance
oversampler = RandomOverSampler(random_state=42)
aggregated_features_resampled, aggregated_target_resampled = oversampler.fit_resample(
    aggregated_features_reshaped, aggregated_target)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    aggregated_features_resampled, aggregated_target_resampled, test_size=0.2, random_state=42)

# Convert target labels to categorical format
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Reshape the input data to match the LSTM input shape (samples, timesteps, features)
input_shape = (n_observations, n_features)
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], *input_shape))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], *input_shape))

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=input_shape))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # Assuming you have 3 target classes

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train_categorical, epochs=10,
          batch_size=32, validation_data=(X_test_reshaped, y_test_categorical))

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 163, "status": "ok", "timestamp": 1684788281599, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="-KWwFdM9NIKD" outputId="1d452c01-630a-42e2-c141-11e24385d278"
pd.Series(y_train).isna().any()
