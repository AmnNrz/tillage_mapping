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
#     display_name: tillmap
#     language: python
#     name: python3
# ---

import geopandas as gpd
import numpy as np

# +
path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
    "Ph.D/Projects/Tillage_Mapping/Data/GIS_Data/Crop_classification/"
)

shpfile_2017 = gpd.read_file(path_to_data + "WSDA_2017/WSDACrop_2017_selection.shp")

# +
# # function to make patches of shape files
# import geopandas as gpd
# import os


# def batch_gpd(gpd_table, output_folder):
#     # Define batch size
#     batch_size = 500

#     # Split the GeoDataFrame into batches
#     batches = [
#         gpd_table.iloc[i : i + batch_size] for i in range(0, len(gpd_table), batch_size)
#     ]

#     # Save each batch as a shapefile
#     for idx, batch in enumerate(batches):
#         output_path = os.path.join(output_folder, f"shapefile_batch_{idx}.shp")
#         batch.to_file(output_path)

# +
print(shpfile_2017["CropType"].unique())

desired_crops = [
    "Wheat",
    "Wheat Fallow",
    "Pea, Dry",
    "Chickpea",
    "Barley",
    "Oat",
    "Barley Hay",
    "Lentil",
    "Canola",
    "Triticale",
]

shpfile_2017 = shpfile_2017.loc[shpfile_2017["CropType"].isin(desired_crops)]
shpfile_2017 = shpfile_2017.rename(columns={"PolygonID": "pointID"})

path_to_testfiles = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
    "Ph.D/Projects/Tillage_Mapping/Data/GIS_Data/"
)

batch_gpd(shpfile_2017, path_to_testfiles + "shapefiles_2017_map")
# -

shpfile_2017.to_file(path_to_testfiles + "shapefiles_2017_test/shp2017.shp")

# +
path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
    "Ph.D/Projects/Tillage_Mapping/Data/field_level_data/"
    "mapping_data/2012/"
)

shpfile_2012 = gpd.read_file(path_to_data + "shp2012.shp")

# +
shpfile_2012 = shpfile_2012.loc[shpfile_2012["Irrigation"] == "None"]

shpfile_2012 = shpfile_2012.loc[shpfile_2012["LastSurvey"] == "2012/12/31 00:00:00.000"]


print(shpfile_2012["CropType"].unique())

desired_crops = [
    "Wheat",
    "Wheat Fallow",
    "Pea, Dry",
    "Chickpea",
    "Barley",
    "Oat",
    "Barley Hay",
    "Lentil",
    "Canola",
    "Triticale",
]

shpfile_2012 = shpfile_2012.loc[shpfile_2012["CropType"].isin(desired_crops)]
shpfile_2012 = shpfile_2012.rename(columns={"PolygonID": "pointID"})
shpfile_2012

path_to_testfiles = ("/Users/aminnorouzi/Library/CloudStorage/"
            "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/"
            "Tillage_Mapping/Data/field_level_data/mapping_data/2012/")

shpfile_2012.to_file(path_to_testfiles + "shp2012.shp")

# +
# # function to make patches of shape files
# import geopandas as gpd
# import os


# def batch_gpd(gpd_table, output_folder):
#     # Define batch size
#     batch_size = 500

#     # Split the GeoDataFrame into batches
#     batches = [
#         gpd_table.iloc[i : i + batch_size] for i in range(0, len(gpd_table), batch_size)
#     ]

#     # Save each batch as a shapefile
#     for idx, batch in enumerate(batches):
#         output_path = os.path.join(output_folder, f"shapefile_batch_{idx}.shp")
#         batch.to_file(output_path)


# path_to_testfiles = (
#     "/Users/aminnorouzi/Library/CloudStorage/"
#     "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/"
#     "Tillage_Mapping/Data/field_level_data/mapping_data/2012/"
# )
# batch_gpd(shpfile_2012, path_to_testfiles + "batches")
