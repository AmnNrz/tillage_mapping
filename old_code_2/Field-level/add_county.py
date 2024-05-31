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

import geopandas as gpd
import pandas as pd
import numpy as np

# +
path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
    "Projects/Tillage_Mapping/Data/"
)


df = pd.read_csv(path_to_data + "field_level_data/FINAL_DATA/season_finalData.csv")
Whitman_2022 = gpd.read_file(path_to_data + "GIS_Data/County_points/Whitman_2022.shp")
Whitman_2023 = gpd.read_file(path_to_data + "GIS_Data/County_points/Whitman_2023.shp")
Columbia_2022 = gpd.read_file(path_to_data + "GIS_Data/County_points/Columbia_2022.shp")
Columbia_2023 = gpd.read_file(path_to_data + "GIS_Data/County_points/Columbia_2023.shp")

# +
whitman_pointIDs = pd.concat([Whitman_2022["pointID"], Whitman_2023["fid"]])
columbia_pointIDs = pd.concat([Columbia_2022["pointID"], Columbia_2023["fid"]])
whitman_pointIDs, columbia_pointIDs

df["County"] = np.NaN
df.loc[df["pointID"].isin(columbia_pointIDs), "County"] = "Columbia"
df.loc[df["pointID"].isin(whitman_pointIDs), "County"] = "Whitman"
df["County"].isnull().value_counts()
county_col = df.pop("County")
df.insert(loc=2, column="County", value=county_col)
df.to_csv(path_to_data + "field_level_data/FINAL_DATA/season_finalData_with_county.csv")
# -

Columbia_2023.shape, Whitman_2023.shape
