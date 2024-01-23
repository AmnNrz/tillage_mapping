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
#     language: python
#     name: python3
# ---

import pandas as pd

df = pd.read_csv("/Users/aminnorouzi/Library/CloudStorage/GoogleDrive-a.norouzikandelati@wsu.edu/My Drive/P.h.D_Projects/Tillage_Mapping/Data/Crop_type_TS/regular_dfs_2014_2015.csv")
df[['indeks', 'indeks.1', 'indeks.2', 'indeks.3', 'indeks.4', 'indeks.5']].describe()

# +
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
subset_size = 3

for i in range(0, len(my_list), subset_size):
    subset = my_list[i:i+subset_size]
    # Do something with the subset
    print(subset)

