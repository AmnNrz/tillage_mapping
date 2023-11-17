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
import numpy as np
import pandas as pd



# def macro_accuracy(y_true, y_pred):
#     # Compute the confusion matrix
#     conf_matrix = confusion_matrix(y_true, y_pred)
    
#     # Calculate accuracy for each class
#     class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
#     # Compute the macro-averaged accuracy
#     macro_avg_accuracy = np.nanmean(class_accuracies)

#     return macro_avg_accuracy


cm = np.array([[29, 1,  1],
       [0, 33,  3],
       [ 0,  5,  15]])

# Calculate accuracy for each class
class_accuracies = cm.diagonal() / cm.sum(axis=1)

# Compute the macro-averaged accuracy
macro_avg_accuracy = np.nanmean(class_accuracies)
macro_avg_accuracy

# +
import pandas as pd

# # Read data
path_to_data = ("/Users/aminnorouzi/Library/CloudStorage/"
                "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
                "Projects/Tillage_Mapping/Data/field_level_data/FINAL_DATA/")

# path_to_data = ("/home/amnnrz/OneDrive - "
#                 "a.norouzikandelati/Ph.D/Projects/Tillage_Mapping/Data"
#                 "/field_level_data/FINAL_DATA/")

df = pd.read_csv(path_to_data + "season_finalData.csv", index_col=0)
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


# -

df.shape
