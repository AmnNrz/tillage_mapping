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

import numpy as np
import pandas as pd
import geopandas as gpd
import os

# +
path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
    "Ph.D/Projects/Tillage_Mapping/Data/"
)

path_to_cdl_data = path_to_data + (
    "MAPPING_DATA_2011_2012_2022/2012_2017_2022/cdl_data/"
)

path_to_landsat_data = path_to_data + (
    "MAPPING_DATA_2011_2012_2022/2012_2017_2022/landsat_data/"
)

path_to_concatenated_data = path_to_data + (
    "MAPPING_DATA_2011_2012_2022/2012_2017_2022/"
)

# +
# Folder containing the CSV files
folder_path = path_to_cdl_data

# List of files
files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]


# Function to process files in chunks
def process_in_batches(file_list, batch_size, output_file):
    # Create an empty DataFrame to store the concatenated result
    for i in range(0, len(file_list), batch_size):
        batch_files = file_list[i : i + batch_size]

        # Read and concatenate files in the current batch
        batch_df = pd.concat(
            [pd.read_csv(os.path.join(folder_path, file)) for file in batch_files],
            ignore_index=True,
        )

        # Append each batch to the final output file
        if i == 0:
            # Write the first batch, creating the output file
            batch_df.to_csv(output_file, mode="w", index=False)
        else:
            # Append to the existing output file
            batch_df.to_csv(output_file, mode="a", header=False, index=False)

        print(f"Processed batch {i // batch_size + 1}/{len(file_list) // batch_size}")


# Set batch size (adjust based on memory capacity)
batch_size = 100

# Output file for the concatenated data
output_file = path_to_concatenated_data + "concatenated_cdl_file.csv"

# Process the files in batches and save to a single file
process_in_batches(files, batch_size, output_file)

# +
cdl_12_17_22 = pd.read_csv(
    path_to_data
    + "MAPPING_DATA_2011_2012_2022/2012_2017_2022/concatenated_cdl_file.csv"
)

lsat_12_17_22 = pd.read_csv(
    path_to_data
    + "MAPPING_DATA_2011_2012_2022/2012_2017_2022/concatenated_landsat_file.csv"
)

cdl_12_17_22 = cdl_12_17_22.drop(columns=["Unnamed: 0"])
lsat_12_17_22 = lsat_12_17_22.drop(columns=["Unnamed: 0"])
# -

cdl_12_17_22["most_frequent_crop"].value_counts()

# +
cdl_12_17_22 = cdl_12_17_22.sort_values(by="pointID")
lsat_12_17_22 = lsat_12_17_22.sort_values(by="pointID")


# Rename "most_frequent_crop" to "cdl_cropType" in CDL data
cdl_12_17_22 = cdl_12_17_22.rename(columns={"most_frequent_crop": "cdl_cropType"})

to_replace = {
    23: "Grain",
    31: "Canola",
    24: "Grain",
    51: "Legume",
    53: "Legume",
    61: "Grain",
    52: "Legume",
    176: "Grassland/Pasture",
    35: "Mustard",
    21: "Grain",
    36: "Alfalfa",
    42: "Legume",
    37: "Hay, nonAlfalfa",
}

cdl_12_17_22["cdl_cropType"] = cdl_12_17_22["cdl_cropType"].replace(to_replace)
cdl_12_17_22 = cdl_12_17_22.loc[
    cdl_12_17_22["cdl_cropType"].isin(["Grain", "Legume", "Canola"])
].copy()

print(lsat_12_17_22.shape, cdl_12_17_22.shape)
lsat_12 = lsat_12_17_22.loc[lsat_12_17_22["year"] == 2012].copy()
lsat_17 = lsat_12_17_22.loc[lsat_12_17_22["year"] == 2017].copy()
lsat_22 = lsat_12_17_22.loc[lsat_12_17_22["year"] == 2022].copy()

cdl_11 = cdl_12_17_22.loc[cdl_12_17_22["year"] == 2011].copy()
cdl_16 = cdl_12_17_22.loc[cdl_12_17_22["year"] == 2016].copy()
cdl_21 = cdl_12_17_22.loc[cdl_12_17_22["year"] == 2021].copy()

# print(lsat_12.shape, lsat_17.shape, lsat_22.shape)
# print(cdl_11.shape, cdl_16.shape, cdl_21.shape)


############ Merge cdl with landsat ############
data_2012 = pd.merge(
    lsat_12, cdl_11[["pointID", "cdl_cropType"]], on="pointID", how="left"
)
data_2017 = pd.merge(
    lsat_17, cdl_16[["pointID", "cdl_cropType"]], on="pointID", how="left"
)
data_2022 = pd.merge(
    lsat_22, cdl_21[["pointID", "cdl_cropType"]], on="pointID", how="left"
)


# Rearrange the columns to place the merged column in the 4th position
def rarange(df):

    cols = list(df.columns)
    # Move the merged column to the 4th position
    cols.insert(4, cols.pop(cols.index("cdl_cropType")))

    # Reorder the DataFrame
    df = df[cols]
    # Fill rows with at least one NaN value
    df_ = df.loc[:, "B_S0_p0":]
    df_ = df_.fillna(df_.median())
    df = pd.concat([df.loc[:, :"cdl_cropType"], df_], axis=1)
    to_replace = {"Grain": 1, "Legume": 2, "Canola": 3}
    df["cdl_cropType"] = df["cdl_cropType"].replace(to_replace)
    df = df.set_index("pointID")
    df = df.loc[
        df["County"].isin(
            ["Whitman", "Spokane", "Asotin", "Walla Walla", "Garfield", "Columbia"]
        )
    ]
    df = df.dropna(how="any")
    return df


data_2012 = rarange(data_2012)
data_2017 = rarange(data_2017)
data_2022 = rarange(data_2022)

data_2012 = data_2012.reset_index()
data_2017 = data_2017.reset_index()
data_2022 = data_2022.reset_index()
# -

data_2012.to_csv(
    path_to_data + "MAPPING_DATA_2011_2012_2022/2012_2017_2022/data_2012.csv",
    index=False,
)
data_2017.to_csv(
    path_to_data + "MAPPING_DATA_2011_2012_2022/2012_2017_2022/data_2017.csv",
    index=False,
)
data_2022.to_csv(
    path_to_data + "MAPPING_DATA_2011_2012_2022/2012_2017_2022/data_2022.csv",
    index=False,
)
