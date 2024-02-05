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
#     display_name: gis_env
#     language: python
#     name: python3
# ---

# +
import geopandas as gpd

path_to_shp = (
    "/media/amnnrz/New Volume/Ph.D._projects/Tillage_Mapping/Data/"
    "GIS_Data/2022_2023_survey_data/shapefile.shp"
)

gdf = gpd.read_file(path_to_shp)
print(gdf.shape)
gdf.head(5)

# +
# Filter data for 2022-2023
gdf2223 = gdf.loc[(gdf["DateTime"] >= "2022-10-13") & (gdf["DateTime"] <= "2023-05-03")]
print(gdf2223.shape)

# Filter for priorCropType and Tillage

desired_cropType = ['Canola?', 'Canola stubble', 'Legume',
                     'Canola', 'Chickpeas', 'Cickpeas', 'Garbz', 
                    'Could be garbz', 'Garbanzos', 'Not grain', 'Garbanzo',
                    'Nongrain', 'Peas', 'Canola¿', 'Garbs','Peas?']

desired_Tillage = ['Minimum Till', 'No Tilll-Direct Seed', 'MinimumTill', 'NoTill-DirectSeed']


gdf2223 = gdf.loc[(gdf["PriorCropT"].isin(desired_cropType)) | (gdf["Tillage"].isin(desired_Tillage))]
print(gdf2223["PriorCropT"].unique()) 
print(gdf2223["Tillage"].unique()) 
# -

gdf2223


