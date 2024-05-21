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

import pandas as pd 
import geopandas as gpd

merged_df

# +
path_to_data = "/home/amnnrz/OneDrive - a.norouzikandelati/Ph.D/Projects/Tillage_Mapping/Data/"

# Read survey data
survey_df = pd.read_csv(path_to_data + 'Survey_data/Tillage_data_2122_cleaned.csv')
survey_df.drop(columns='pointID.1', inplace=True)

# Read wsda shape file
wsda_pols = gpd.read_file(path_to_data + 'GIS_Data/shapefiles_2021_2022/2021_2022_polygons/WSDA_checkedForPins.shp')
merged_df = pd.merge(survey_df, wsda_pols, on='pointID', how='left')
merged_df = gpd.GeoDataFrame(merged_df, geometry='geometry')
merged_df.to_file(path_to_data + 'GIS_Data/shapefiles_2021_2022/point_polygon_joined_2122/point_polygon_joined_2122.shp')
# -

survey_df.shape, wsda_pols.shape

#
