# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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
import pandas as pd

# path_to_data = (
#     "/Users/aminnorouzi/Library/CloudStorage/"
#     "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
#     "Projects/Tillage_Mapping/Data/GIS_Data/check_for_acres/"
# )
path_to_data = (
    "/home/amnnrz/OneDrive - a.norouzikandelati/Ph.D/"
    "Projects/Tillage_Mapping/Data/GIS_Data/check_for_acres/"
)
gpd_2012 = gpd.read_file(path_to_data + "WSDA_2012_reprojected_dryland_WA.shp")
gpd_2017 = gpd.read_file(path_to_data + "WSDA_2017_reprojected_dryland_WA.shp")

gpd_2012

# +
# eastern_counties = ['Whitman', 'Columbia', 'Adams', 'Garfield', 'Asotin',
#                     'Lincoln', 'Douglas', 'Grant', 'Benton', 'Franklin', 'Spokane']
eastern_counties = ['Whitman', 'Columbia']

gpd_2017_ = gpd_2017.loc[gpd_2017['County'].isin(eastern_counties)]
gpd_2017_['CropType'].unique()

# selected_crops = ['Wheat', 'Wheat Fallow',
#        'Fallow, Idle', 'Fallow', 'Grass Hay',
#        'Pea, Green', 'Wildlife Feed', 'Alfalfa Hay', 'Corn, Field', 'Nursery, Ornamental',
#        'Alfalfa/Grass Hay', 'Rye', 'Fallow, Tilled', 'Barley', 'Chickpea', 'Pea, Dry',
#        'Barley Hay', 'Canola', 'Potato', 'Timothy',
#        'Corn Seed', 'Triticale', 'Bean, Dry', 'Sugar Beet Seed', 
#        'Bluegrass Seed', 'Oat', 'Pea Seed',
#        'Corn, Sweet', 'Sunflower', 'Oat Hay',
#        'Leek', 'Market Crops', 'Onion', 'Sorghum', 'Buckwheat',
#        'Green Manure', 'Lentil', 'Mustard', 'Pumpkin', 'Triticale Hay', 'Flax',
#        'Grass Seed, Other', 'Sudangrass', 'Cereal Grain, Unknown',
#        'Sunflower Seed', 'Legume Cover',
#        'Bromegrass Seed']
selected_crops = ['Wheat', 'Wheat Fallow',
       'Pea, Green', 'Rye', 'Barley', 'Chickpea', 'Pea, Dry',
       'Barley Hay', 'Canola', 'Triticale', 'Bean, Dry', 'Oat', 'Pea Seed', 'Oat Hay', 'Sorghum', 'Buckwheat',
         'Lentil', 'Triticale Hay', 'Cereal Grain, Unknown', 'Legume Cover'
       ]

gpd_2017_filtered = gpd_2017_.loc[gpd_2017_['CropType'].isin(selected_crops)]
gpd_2017_filtered.groupby(["County"])["ExactAcres"].sum()
gpd_2017_filtered['ExactAcres'].sum()
# -

county_crop_acres = gpd_2017.groupby(["County", "CropType"])["ExactAcres"].sum()
county_crop_acres = county_crop_acres.reset_index()
county_crop_acres.to_csv(path_to_data + 'county_crop_acres.csv')
