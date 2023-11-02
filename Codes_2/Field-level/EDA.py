# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: tillmap
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder


# # # Read data
# path_to_data = ("/Users/aminnorouzi/Library/CloudStorage/"
#                 "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
#                 "Projects/Tillage_Mapping/Data/field_level_data/FINAL_DATA/")

path_to_data = ("/home/amnnrz/OneDrive - "
                "a.norouzikandelati/Ph.D/Projects/Tillage_Mapping/Data"
                "/field_level_data/FINAL_DATA/")

df = pd.read_csv(path_to_data + "metric_finalData.csv", index_col=0)
df = df.dropna(subset=["Tillage", "ResidueType", "ResidueCov"])
print(df['ResidueType'].value_counts(), df['ResidueCov'].value_counts())
le_resCov = LabelEncoder()
le_resType = LabelEncoder()
df['ResidueCov'] = le_resCov.fit_transform(df['ResidueCov'])
df['ResidueType'] = le_resType.fit_transform(df['ResidueType'])
print(df['ResidueType'].value_counts(), df['ResidueCov'].value_counts())
df = df.set_index('pointID')
df


# +
from sklearn.preprocessing import StandardScaler

X = df.iloc[:, [2, 4] + list(range(7, df.shape[1]))]
l1 = list(X.index)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns, index=list(X.index))
X
# print(min(X.describe().min))
# print(max(X.describe().max))


# -

l2 = list(X.index)


