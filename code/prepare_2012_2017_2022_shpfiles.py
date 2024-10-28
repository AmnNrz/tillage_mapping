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
import numpy as np
import pandas as pd

# +
path_to_data = path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
    "Ph.D/Projects/Tillage_Mapping/Data/"
)

shp_2012 = gpd.read_file(
    path_to_data + "MAPPING_DATA_2011_2012_2022/shapefiles/2012_2017_2022/WSDA_2012.shp"
)
shp_2017 = gpd.read_file(
    path_to_data + "MAPPING_DATA_2011_2012_2022/shapefiles/2012_2017_2022/WSDA_2017.shp"
)
shp_2022 = gpd.read_file(
    path_to_data + "MAPPING_DATA_2011_2012_2022/shapefiles/2012_2017_2022/WSDA_2022.shp"
)

# -

shp_2012 = shp_2012.loc[shp_2012["County"].isin(["Whitman", "Spokane", "Asotin",
                        "Walla Walla", "Garfield", "Columbia"])]
shp_2017 = shp_2017.loc[shp_2017["County"].isin(["Whitman", "Spokane", "Asotin",
                        "Walla Walla", "Garfield", "Columbia"])]
shp_2022 = shp_2022.loc[shp_2022["County"].isin(["Whitman", "Spokane", "Asotin",
                        "Walla Walla", "Garfield", "Columbia"])]

# +
shp_2012 = shp_2012.loc[:, "CropGroup":]
shp_2017 = shp_2017.loc[:, "CropGroup":]
shp_2022 = shp_2022.loc[:, "Acres":]


# Concatenate dataframes for uniform pointID assignment
combined_df = pd.concat([shp_2012, shp_2017, shp_2022], ignore_index=True)

# Create unique pointID for all rows
combined_df["pointID"] = range(1, len(combined_df) + 1)

# Split the combined dataframe back into the original dataframes and insert pointID at position 0
shp_2012["pointID"] = combined_df.loc[: len(shp_2012) - 1, "pointID"].values
shp_2017["pointID"] = combined_df.loc[
    len(shp_2012) : len(shp_2012) + len(shp_2017) - 1, "pointID"
].values
shp_2022["pointID"] = combined_df.loc[len(shp_2012) + len(shp_2017) :, "pointID"].values

# Reorder columns to place pointID at the first position
shp_2012 = shp_2012[["pointID"] + [col for col in shp_2012.columns if col != "pointID"]]
shp_2017 = shp_2017[["pointID"] + [col for col in shp_2017.columns if col != "pointID"]]
shp_2022 = shp_2022[["pointID"] + [col for col in shp_2022.columns if col != "pointID"]]
# -

shp_2012.to_file(path_to_data + "MAPPING_DATA_2011_2012_2022/shapefiles/WSDA_2012.shp")
shp_2017.to_file(path_to_data + "MAPPING_DATA_2011_2012_2022/shapefiles/WSDA_2017.shp")
shp_2022.to_file(path_to_data + "MAPPING_DATA_2011_2012_2022/shapefiles/WSDA_2022.shp")
