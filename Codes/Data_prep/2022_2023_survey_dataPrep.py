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
import pandas as pd

path_to_gisData = ("/home/amnnrz/GoogleDrive - a.norouzikandelati/"
                    "Ph.D._projects/Tillage_Mapping/Data/GIS_Data/"
                    "2022_2023_survey_data/")

gdf = gpd.read_file(path_to_gisData + "shapefile.shp")
print(gdf.shape)
gdf.head(5)
# -

gdf['PriorCropT'].unique()

# +
# Filter data for 2022-2023
gdf2223 = gdf.loc[((gdf["DateTime"] >= "2022-10-13") \
                   & (gdf["DateTime"] <= "2023-05-03"))]
print(gdf2223.shape)

# Filter for priorCropType and Tillage

desired_cropType = ['Canola?', 'Canola stubble', 'Legume',
                     'Canola', 'Chickpeas', 'Cickpeas', 'Garbz', 
                    'Could be garbz', 'Garbanzos', 'Not grain', 'Garbanzo',
                    'Nongrain', 'Peas', 'Canola¿', 'Garbs','Peas?']

desired_Tillage = ['Minimum Till', 'No Tilll-Direct Seed', 
                   'MinimumTill', 'NoTill-DirectSeed']


gdf2223 = gdf2223.loc[(gdf2223["PriorCropT"].isin(desired_cropType)) \
                      | (gdf2223["Tillage"].isin(desired_Tillage))]
print(gdf2223["PriorCropT"].unique()) 
print(gdf2223["Tillage"].unique())

# +
cropType_map = {
    'Canola?':'Canola', 'Canola stubble':'Canola', 'Wheat':'Grain',
      'Ww':'Grain', 'Peas and wheat':'Legume & Grain',
  'Chickpeas':'Legume', 'Cickpeas':'Legume', 'Garbz':'Legume',
    'W':'Grain', 'Old wheat':'Grain', 'Old w':'Grain',
 'Could be garbz':'Legume', 'Garbanzos':'Legume', 'Garbanzo':'Legume',
  'Peas':'Legume', 'Canola¿':'Canola', 'Garbs':'Legume', 
  'Peas?':'Legume'
}

gdf2223['PriorCropT'].replace(cropType_map, inplace=True)
gdf2223['PriorCropT'].unique()
# -

gdf2223.to_file(path_to_gisData + 'gdf2223.shp')

polID_joined_df = gpd.read_file(path_to_gisData + 'pol_ID_joined_date&crop_filtered.shp')

polID_joined_df['fid']

polID_joined_df = polID_joined_df.sort_values(by="fid")
polID_joined_df.to_csv(path_to_gisData + 'pol_ID_joined_date&crop_filtered.csv')
polID_joined_df.to_file(path_to_gisData + 'pol_ID_joined_date&crop_filtered.shp')

from shapely.geometry import Point


# +
cleaned_srvey_data = pd.read_csv(path_to_gisData + 'pol_ID_joined_date&crop_filtered_cleaned.csv')

x_y_list = list(map(lambda row: row[6:].strip('()').split(), cleaned_srvey_data['geometry']))
x_list = list(map(lambda point_list: float(point_list[0]), x_y_list))
y_list = list(map(lambda point_list: float(point_list[1]), x_y_list))
cleaned_srvey_data['x'] = pd.Series(x_list)
cleaned_srvey_data['y'] = pd.Series(y_list)
cleaned_srvey_data['y']
cleaned_srvey_data['geometry'] = cleaned_srvey_data.apply(lambda row: Point(row.x, row.y), axis=1)
cleaned_srvey_data['geometry'] 

cleaned_gpd = gpd.GeoDataFrame(cleaned_srvey_data, geometry='geometry')
cleaned_gpd.columns

cleaned_gpd.to_file(path_to_gisData + 'pol_ID_joined_date&crop_filtered_cleaned.shp')
