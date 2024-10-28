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
#     display_name: gee
#     language: python
#     name: python3
# ---

import os
import numpy as np
import pandas as pd

# +
path_to_data = ("/home/a.norouzikandelati/Projects/Tillage_mapping/Data/2012_2017_2022/")

# Get a list of all files in the folder
files = os.listdir(path_to_data + "landsat_data/")

# Extract the numbers from file names
file_numbers = []
for file in files:
    if file.startswith('Landsat_metricBased_eastwa_') and file.endswith('.csv'):
        num = int(file.split('_')[-1].split('.')[0])  # Extract the number
        file_numbers.append(num)

# Find the missing numbers
all_numbers = set(range(1, 1001))  # Assuming batch numbers range from 1 to 1000
present_numbers = set(file_numbers)
missing_numbers = sorted(all_numbers - present_numbers)

print("Missing batch numbers:", missing_numbers)
len(missing_numbers)
