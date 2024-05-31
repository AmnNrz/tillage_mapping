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
#     name: python3
# ---

# + id="pVuCLkklxrzn"
import pandas as pd
import numpy as np

# !pip install geemap
# !pip install geopandas

import geemap
import geopandas as gpd

<<<<<<<=======
>>>>>>> origin/main
# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 303, "status": "ok", "timestamp": 1680801215795, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="_Ff18vB8VrTJ" outputId="46f007b1-c185-412d-fed5-631bcf736497"
print(pd.__version__)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 18357, "status": "ok", "timestamp": 1681157811898, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="DwmP9Psax2EG" outputId="c9f7e4f5-7b5a-4d5c-a0bf-7cffbb124cd7"
from google.colab import drive
drive.mount('/content/drive')

# + executionInfo={"elapsed": 4644, "status": "ok", "timestamp": 1681157816538, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="05iuBmosze_6"
shapeFile_path = "/content/drive/MyDrive/P.h.D_Projects/Tillage_Mapping/Data/GIS_Data/WSDA_survey/WSDA_checkedForPins.dbf"
WSDA_featureCol_goepd = gpd.read_file(shapeFile_path, crs='EPSG:4326')
wsda_cols = WSDA_featureCol_goepd.columns


# + [markdown] id="b-iKSWky55Vd"
# # Prepare Season-based MainBand data

# + id="pRXCQzTaJTeP"
# # Prepare Season-based MainBand data 
# folder_path = "/content/drive/MyDrive/seasonBased_Pixel_level_mainBands_TillageData"

# import os
# from pandas.errors import EmptyDataError

# fall_dataframe_list = []
# spring_dataframe_list = []
# empty_pols_list = []

# spring_files = [i for i in os.listdir(folder_path) if "spring" in i.split("_")]
# fall_files = [i for i in os.listdir(folder_path) if "fall" in i.split("_")]
# def sort_key(s):
#     # Extract the numeric part of the string using regular expressions
#     import re
#     return [int(x) if x.isdigit() else x for x in re.findall(r'\d+|\D+', s.split("_")[-1])]
# fall_files.sort(key=sort_key)
# spring_files.sort(key=sort_key)

# count = 0
# for fall_file_name, spring_file_name in zip(fall_files, spring_files):
#   # print(fall_file_name, spring_file_name)
#   fall_file_path = os.path.join(folder_path, fall_file_name)
#   spring_file_path = os.path.join(folder_path, spring_file_name)

#   # Ignore the files that are empty
#   try:
#     pd.read_csv(fall_file_path)
#     pd.read_csv(spring_file_path)
#   except EmptyDataError:
#     continue

#   fall_file = pd.read_csv(fall_file_path)
#   spring_file = pd.read_csv(spring_file_path)

#   fall_file = fall_file.iloc[:min(fall_file.shape[0], spring_file.shape[0])]
#   spring_file = spring_file.iloc[:min(fall_file.shape[0], spring_file.shape[0])]
#   print(fall_file.shape, spring_file.shape)
#   if (fall_file.shape[0] < 10) | (spring_file.shape[0] < 10):
#     count += 1
#   fall_dataframe_list += [fall_file]
#   spring_dataframe_list += [spring_file]
# print(count)
# large_fall_df = pd.concat(fall_dataframe_list, axis=0)
# large_spring_df = pd.concat(spring_dataframe_list, axis=0)
# large_fall_df.shape, large_spring_df.shape
# season_based_mainBand = pd.concat([large_fall_df, large_spring_df], axis=1)

# season_based_mainBand_cols = season_based_mainBand.columns.values
# col_to_rm = np.isin(season_based_mainBand_cols, wsda_cols)
# cols_to_keep = np.append("pointID", season_based_mainBand_cols[~col_to_rm])
# season_based_mainBand_dataframe = season_based_mainBand[cols_to_keep]
# # Delete duplicated columns
# season_based_mainBand_dataframe = season_based_mainBand_dataframe.loc[:, 
#                           ~season_based_mainBand_dataframe.columns.duplicated()]
# # Delete .geo column
# season_based_mainBand_dataframe.drop(".geo", axis=1, inplace=True)
# season_based_mainBand_dataframe.columns

# + id="H__VyWdA__a5"
# season_based_mainBand_dataframe.to_csv("season_based_mainBand.csv")

# + [markdown] id="KZJoCFYX55Vf"
# # Prepare Season-based glcmBand data 

# + id="utnDLhAUYwat"
# # Prepare Season-based glcmBand data 
# # folder_path = "/content/drive/MyDrive/seasonBased_pixel_level_glcmBands_TillageData"
# folder_path = r"H:\My Drive\seasonBased_pixel_level_glcmBands_TillageData"
# import os
# from pandas.errors import EmptyDataError

# fall_dataframe_list = []
# spring_dataframe_list = []
# empty_pols_list = []

# spring_files = [i for i in os.listdir(folder_path) if "spring" in i.split("_")]
# fall_files = [i for i in os.listdir(folder_path) if "fall" in i.split("_")]
# def sort_key(s):
#     # Extract the numeric part of the string using regular expressions
#     import re
#     return [int(x) if x.isdigit() else x for x in re.findall(r'\d+|\D+', s.split("_")[-1])]
# fall_files.sort(key=sort_key)
# spring_files.sort(key=sort_key)

# count = 0
# emptyfile_names = []

# # Pre-allocate a dataframe to put all csv files in its elements
# DF = pd.DataFrame(index=range(len(fall_files)), columns=["fall_files", "spring_files"])

# # for idx, (fall_file_name, spring_file_name) in enumerate(zip(fall_files[:len(fall_files)//2], spring_files[:len(fall_files)//2])):
# for idx, (fall_file_name, spring_file_name) in enumerate(zip(fall_files, spring_files)):
#   # print(fall_file_name, spring_file_name)
#   fall_file_path = os.path.join(folder_path, fall_file_name)
#   spring_file_path = os.path.join(folder_path, spring_file_name)

#   # Ignore the files that are empty
#   try:
#     pd.read_csv(fall_file_path)
#     pd.read_csv(spring_file_path)
#   except EmptyDataError:
#     emptyfile_names += [fall_file_name.split("_")[-1]]
#     continue
#   except OSError:
#     print("The pandas argument problem is because of this file :", fall_file_name, "or", spring_file_name)
#     continue
   

#   fall_file = pd.read_csv(fall_file_path)
#   spring_file = pd.read_csv(spring_file_path)

#   fall_file = fall_file.iloc[:min(fall_file.shape[0], spring_file.shape[0])]
#   spring_file = spring_file.iloc[:min(fall_file.shape[0], spring_file.shape[0])]

#   DF.at[idx, "fall_files"] = fall_file
#   DF.at[idx, "spring_files"] = spring_file

#   print(fall_file_name, spring_file_name)
#   if (fall_file.shape[0] < 10) | (spring_file.shape[0] < 10):
#     count += 1

# DF.dropna(how='any', inplace=True)
# DF.reset_index(inplace=True)

# fall_dataframe_list = [DF.at[i, "fall_files"] for i in range(DF.shape[0])]
# spring_dataframe_list = [DF.at[i, "spring_files"] for i in range(DF.shape[0])]

# print('number of fields with less than 10 pixel data is ', count)
# large_fall_df = pd.concat(fall_dataframe_list, axis=0)
# large_spring_df = pd.concat(spring_dataframe_list, axis=0)
# large_fall_df.shape, large_spring_df.shape
# season_based_glcmBand = pd.concat([large_fall_df, large_spring_df], axis=1)

# season_based_glcmBand_cols = season_based_glcmBand.columns.values
# col_to_rm = np.isin(season_based_glcmBand_cols, wsda_cols)
# cols_to_keep = np.append("pointID", season_based_glcmBand_cols[~col_to_rm])
# season_based_glcmBand_dataframe = season_based_glcmBand[cols_to_keep]
# # Delete duplicated columns
# season_based_glcmBand_dataframe = season_based_glcmBand_dataframe.loc[:, 
#                           ~season_based_glcmBand_dataframe.columns.duplicated()]
# # Delete .geo column
# season_based_glcmBand_dataframe.drop(".geo", axis=1, inplace=True)
# season_based_glcmBand_dataframe.columns

# + id="tzA3FwBKX2we"
# season_based_glcmBand_dataframe.to_csv(r'H:\My Drive\Codes\season_based_glcmBand.csv')

# + [markdown] id="tBeDhyeT55Vg"
# # Join Mainband and glcm bands dataframes

# + id="zvabm6Ib55Vg"
import pandas as pd
season_based_mainBand = pd.read_csv(r"H:\My Drive\season_based_mainBand.csv")
season_based_glcmBand = pd.read_csv(r"H:\My Drive\season_based_glcmBand.csv")
season_based_mainBand.shape, season_based_glcmBand.shape

# + id="wy-SG9UF55Vh"
(~(season_based_mainBand["pointID"].value_counts() == season_based_glcmBand["pointID"].value_counts())).idxmax()

# + id="FR1_Z-bS55Vi"
season_based_mainBand.loc[season_based_mainBand["pointID"] == 351]

# + id="PAhKNA5O__a8"
season_based_glcmBand.loc[season_based_glcmBand["pointID"] == 351]
season_based_glcmBand.drop(index=945572, inplace=True)
season_based_mainBand.reset_index(inplace=True)
season_based_glcmBand.reset_index(inplace=True)
season_based_mainBand.shape, season_based_glcmBand.shape

# + id="6dQWiwL-__a8"
seasonBased_main_glcm = pd.concat([season_based_mainBand, season_based_glcmBand.loc[:, "B_fall_asm":]], axis=1)
seasonBased_main_glcm

# + id="VRTdDoVP__a8"
tillData = pd.read_excel(r"H:\My Drive\Codes\Tillage_data.xlsx")
tillData.dropna(subset="ResidueCov", inplace=True)
tillData["ResidueCov"].isnull().value_counts()
merged =pd.merge(seasonBased_main_glcm, tillData, on="pointID")
merged

# + id="hUJVRZCH__a9"
# pd.Series(seasonBased_main_glcm["pointID"].unique()).isin(tillData["pointID"]).value_counts()
# tillData["pointID"].isin(pd.Series(seasonBased_main_glcm["pointID"].unique())).value_counts()

# + id="NCSMoZup__a9"
seasonBased_main_glcm.insert(3, "residue_%", merged["ResidueCov"])
# seasonBased_main_glcm.drop("residue_%", axis=1, inplace=True)
seasonBased_main_glcm.dropna(subset="residue_%", inplace=True)
seasonBased_main_glcm

# + id="rvlFY_6i__a9"
# seasonBased_main_glcm.to_csv("seasonBased_main_glcm.csv")

# + [markdown] id="P2O0LcroAI5O"
# # Prepare Distribution-based Main Bands

# + id="TGAEyzzBAncO"
# Prepare Season-based MainBand data 
folder_path = "/content/drive/MyDrive/P.h.D_Projects/Tillage_Mapping/Data/Distribution-Based_mainBands_TillageData"

import os
from pandas.errors import EmptyDataError

fall_dataframe_list = []
spring_dataframe_list = []
empty_pols_list = []

spring_files = [i for i in os.listdir(folder_path) if "spring" in i.split("_")]
fall_files = [i for i in os.listdir(folder_path) if "fall" in i.split("_")]

def sort_key(s):
  # Extract number of polygon
  polygon_n = int(s.split("_")[5].split(".")[0][7:])
  # Extract number of pth 
  P_n = int(s.split("_")[4][1:])

  return (polygon_n, P_n)
  
fall_files.sort(key=sort_key)
spring_files.sort(key=sort_key)

# Make sublists of percentiles for each polygon
def sublst(lst):
  d = {}
  for s in lst:
    polygon = s.split('_')[-1].split('.')[0] # extract the polygon_x part
    if polygon in d:
        d[polygon].append(s)
    else:
        d[polygon] = [s]

  return list(d.values())

fall_files = sublst(fall_files)
spring_files = sublst(spring_files)

count = 0
for fall_file_name, spring_file_name in zip(fall_files, spring_files):
  # print(fall_file_name[0], spring_file_name[0])
  fall_file_paths = [os.path.join(folder_path, x) for x in fall_file_name]
  spring_file_paths = [os.path.join(folder_path, x) for x in spring_file_name]

  # Ignore the files that are empty
  try:
    [pd.read_csv(file) for file in fall_file_paths]
    [pd.read_csv(file) for file in spring_file_paths]
  except EmptyDataError:
    continue

  fall_file = [pd.read_csv(fall_file_path) for fall_file_path in fall_file_paths]
  spring_file = [pd.read_csv(spring_file_path) for spring_file_path in spring_file_paths]

  fall_file = [x.iloc[:min(fall_file[0].shape[0], spring_file[0].shape[0])] for x in fall_file]
  spring_file = [x.iloc[:min(fall_file[0].shape[0], spring_file[0].shape[0])] for x in spring_file]
  print(fall_file[0].shape, spring_file[0].shape)

  if (fall_file[0].shape[0] < 10) | (spring_file[0].shape[0] < 10):
    count += 1
  fall_dataframe_list += [fall_file]
  spring_dataframe_list += [spring_file]
print(count)

fall_dataframe_List_concatinatedPths = [pd.concat(x, axis=1) for x in fall_dataframe_list]
spring_dataframe_List_concatinatedPths = [pd.concat(x, axis=1) for x in spring_dataframe_list]

large_fall_df = pd.concat(fall_dataframe_List_concatinatedPths, axis=0)
large_spring_df = pd.concat(spring_dataframe_List_concatinatedPths, axis=0)
large_fall_df.shape, large_spring_df.shape

metric_based_mainBand = pd.concat([large_fall_df, large_spring_df], axis=1)

metric_based_mainBand_cols = metric_based_mainBand.columns.values
col_to_rm = np.isin(metric_based_mainBand_cols, wsda_cols)
cols_to_keep = np.append("pointID", metric_based_mainBand_cols[~col_to_rm])
metric_based_mainBand_dataframe = metric_based_mainBand[cols_to_keep]
# Delete duplicated columns
metric_based_mainBand_dataframe = metric_based_mainBand_dataframe.loc[:, 
                          ~metric_based_mainBand_dataframe.columns.duplicated()]
# Delete .geo column
metric_based_mainBand_dataframe.drop(".geo", axis=1, inplace=True)

# Drop "doy" columns
metric_based_mainBand_dataframe.drop(columns=metric_based_mainBand_dataframe.filter(regex="^doy"), inplace=True)

# Save the Dataframe as .csv
metric_based_mainBand_dataframe.to_csv("/content/drive/MyDrive/P.h.D_Projects/Tillage_Mapping/Data/spectral_data/metricBased_mainBand.csv")

# + id="a2kWyC6pw5BU"
# Prepare Season-based MainBand data 
folder_path = "/content/drive/MyDrive/P.h.D_Projects/Tillage_Mapping/Data/Distribution-Based_mainBands_TillageData"

import os
from pandas.errors import EmptyDataError

fall_dataframe_list = []
spring_dataframe_list = []
empty_pols_list = []

spring_files = [i for i in os.listdir(folder_path) if "spring" in i.split("_")]
fall_files = [i for i in os.listdir(folder_path) if "fall" in i.split("_")]

def sort_key(s):
  # Extract number of polygon
  polygon_n = int(s.split("_")[5].split(".")[0][7:])
  # Extract number of pth 
  P_n = int(s.split("_")[4][1:])

  return (polygon_n, P_n)
  
fall_files.sort(key=sort_key)
spring_files.sort(key=sort_key)

# Make sublists of percentiles for each polygon
def sublst(lst):
  d = {}
  for s in lst:
    polygon = s.split('_')[-1].split('.')[0] # extract the polygon_x part
    if polygon in d:
        d[polygon].append(s)
    else:
        d[polygon] = [s]

  return list(d.values())

fall_files = sublst(fall_files)
spring_files = sublst(spring_files)

count = 0
for fall_file_name, spring_file_name in zip(fall_files, spring_files):
  # print(fall_file_name[0], spring_file_name[0])
  fall_file_paths = [os.path.join(folder_path, x) for x in fall_file_name]
  spring_file_paths = [os.path.join(folder_path, x) for x in spring_file_name]

  # Ignore the files that are empty
  try:
    [pd.read_csv(file) for file in fall_file_paths]
    [pd.read_csv(file) for file in spring_file_paths]
  except EmptyDataError:
    continue

  fall_file = [pd.read_csv(fall_file_path) for fall_file_path in fall_file_paths]
  spring_file = [pd.read_csv(spring_file_path) for spring_file_path in spring_file_paths]

  fall_file = [x.iloc[:min(fall_file[0].shape[0], spring_file[0].shape[0])] for x in fall_file]
  spring_file = [x.iloc[:min(fall_file[0].shape[0], spring_file[0].shape[0])] for x in spring_file]
  print(fall_file[0].shape, spring_file[0].shape)

  if (fall_file[0].shape[0] < 10) | (spring_file[0].shape[0] < 10):
    count += 1
  fall_dataframe_list += [fall_file]
  spring_dataframe_list += [spring_file]
print(count)

fall_dataframe_List_concatinatedPths = [pd.concat(x, axis=1) for x in fall_dataframe_list]
spring_dataframe_List_concatinatedPths = [pd.concat(x, axis=1) for x in spring_dataframe_list]

large_fall_df = pd.concat(fall_dataframe_List_concatinatedPths, axis=0)
large_spring_df = pd.concat(spring_dataframe_List_concatinatedPths, axis=0)
large_fall_df.shape, large_spring_df.shape

metric_based_mainBand = pd.concat([large_fall_df, large_spring_df], axis=1)

metric_based_mainBand_cols = metric_based_mainBand.columns.values
col_to_rm = np.isin(metric_based_mainBand_cols, wsda_cols)
cols_to_keep = np.append("pointID", metric_based_mainBand_cols[~col_to_rm])
metric_based_mainBand_dataframe = metric_based_mainBand[cols_to_keep]
# Delete duplicated columns
metric_based_mainBand_dataframe = metric_based_mainBand_dataframe.loc[:, 
                          ~metric_based_mainBand_dataframe.columns.duplicated()]
# Delete .geo column
metric_based_mainBand_dataframe.drop(".geo", axis=1, inplace=True)

# Drop "doy" columns
metric_based_mainBand_dataframe.drop(columns=metric_based_mainBand_dataframe.filter(regex="^doy"), inplace=True)

# Save the Dataframe as .csv
metric_based_mainBand_dataframe.to_csv("/content/drive/MyDrive/P.h.D_Projects/Tillage_Mapping/Data/spectral_data/metricBased_mainBand.csv")
