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
#     display_name: base
#     language: python
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 24814, "status": "ok", "timestamp": 1680797024513, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="2kjxDtrqLF8r" outputId="a6c9e519-6d19-43e0-c9c2-4ecd21988e12"
from google.colab import drive
drive.mount('/content/drive')

####################

# + executionInfo={"elapsed": 4414, "status": "ok", "timestamp": 1680797052166, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="szS71fJ-K5lp"
# import sys
# sys.path.append('/content/drive/MyDrive/github/Ph.D._Projects/Tillage_Mapping/Codes')


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import NASA_core as nc
# -

np.__version__

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 38677, "status": "ok", "timestamp": 1680797096751, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="xAKga-kEK5lr" outputId="f90d5f8d-3e42-46b9-eb28-125fc5064e0d"
colnames = ['pointID', 'LastSurveyDate', 'TotalAcres', 'CropType', 'NDVI', 'EVI', 'blue', 'green', 'red', 'ni', 'swi1',
            'swi2', 'system_start_time']
# directory = "/content/drive/MyDrive/P.h.D_Projects/Tillage_Mapping/Data"
directory = r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data"
L7raw = pd.read_csv(directory + "/L7_T1C2L2_timeSeries_2014-08-01_2017-07-31_WSDA_2016.csv")
L8raw = pd.read_csv(directory + "/L8_T1C2L2_timeSeries_2014-08-01_2017-07-31_WSDA_2016.csv")


L7raw.columns = colnames
L8raw.columns = colnames


print(L7raw.columns)
print(L8raw.columns)

###############################################################################
L_8_7_raw = pd.concat([L8raw, L7raw])

###############################################################################
# Remove swi2 
L_8_7_raw = L_8_7_raw.drop(columns="swi1", axis=1).copy()

# Drop na 
L_8_7_raw.dropna(how='any', subset=['NDVI', 'EVI', 'blue', 'green', 'red', 'ni', 'swi2'], inplace=True)

# Filter Dataframe based on the year's Last Survey Date
L_8_7_raw["LastSurveyDate"] = pd.to_datetime(L_8_7_raw["LastSurveyDate"])

start_date = "2016-05-05" 
end_date = "2016-12-31"  

L_8_7_raw = L_8_7_raw.loc[(L_8_7_raw["LastSurveyDate"] >= start_date) & (L_8_7_raw["LastSurveyDate"] <= end_date)]
###############################################################################

# Function to convert system_start_time to human readable format
def add_human_start_time_by_system_start_time(HDF):
    """Returns human readable time (conversion of system_start_time)
    Arguments
    ---------
    HDF : dataframe
    Returns
    -------
    HDF : dataframe
        the same dataframe with added column of human readable time.
    """
    HDF.system_start_time = HDF.system_start_time / 1000
    time_array = HDF["system_start_time"].values.copy()
    human_time_array = [time.strftime('%Y-%m-%d', time.localtime(x)) for x in time_array]
    HDF["human_system_start_time"] = human_time_array

    if type(HDF["human_system_start_time"]==str):
        HDF['human_system_start_time'] = pd.to_datetime(HDF['human_system_start_time'])
    
    """
    Lets do this to go back to the original number:
    I added this when I was working on Colab on March 30, 2022.
    Keep an eye on it and see if we have ever used "system_start_time"
    again. If we do, how we use it; i.e. do we need to get rid of the 
    following line or not.
    """
    HDF.system_start_time = HDF.system_start_time * 1000
# Convert 
add_human_start_time_by_system_start_time(L_8_7_raw)

###############################################################################
# Filter "human_system_start_time" based on the year's agricutural year (the desired time-series period)
start_date = "2015-08-01"
end_date = "2016-07-31"

L_8_7_raw = L_8_7_raw.loc[(L_8_7_raw["human_system_start_time"] >= start_date) & (L_8_7_raw["human_system_start_time"] <= end_date)]
###############################################################################
# Remove fields with less that 10 Acres
L_8_7_raw = L_8_7_raw.loc[L_8_7_raw["TotalAcres"]>=10]

###############################################################################


# -


#

# + colab={"base_uri": "https://localhost:8080/", "height": 491} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1680797096752, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="HRNyazbHKNF-" outputId="20490297-3914-4cb2-db09-64c7c277d17e"
L87raw = L_8_7_raw.sort_values(by="system_start_time").copy()
L87raw.reset_index(inplace=True, drop=True)
L87raw["pointID"] = L87raw["pointID"].astype(str)
L87raw.NDVI

# + colab={"base_uri": "https://localhost:8080/"} id="4Zbz4fe0SQoj" outputId="0a2c3909-43ff-47b5-efec-c195bbdf3b47"
# indices = ['NDVI', 'EVI', 'blue', 'green', 'red', 'ni', 'swi2']
indices = ["NDVI"]
regular_dfs = pd.DataFrame([])

indeks_list = []
for indeks in indices:
  indeks_list += [indeks]
  L87raw = L_8_7_raw.sort_values(by="system_start_time").copy()
  L87raw.reset_index(inplace=True, drop=True)
  L87raw["pointID"] = L87raw["pointID"].astype(str)
  IDs = np.sort(L87raw["pointID"].unique())
  no_outlier_df = pd.DataFrame(data = None,
                         index = np.arange(L87raw.shape[0]), 
                         columns = L87raw.columns)
  counter = 0
  row_pointer = 0
  for a_poly in IDs:
    # print(a_poly)
    # if (counter % 1000 == 0):
        # print ("counter is [{:.0f}].".format(counter))
    curr_field = L87raw[L87raw["pointID"]==a_poly].copy()
    # print(curr_field.shape)
    # small fields may have nothing in them!
    if curr_field.shape[0] > 2:
        ##************************************************
        #
        #    Set negative index values to zero.
        #
        ##************************************************
  
        # curr_field.loc[curr_field[indeks] < 0 , indeks] = 0 
        no_Outlier_TS = nc.interpolate_outliers_EVI_NDVI(outlier_input = curr_field, given_col = indeks)
        no_Outlier_TS.loc[no_Outlier_TS[indeks] < 0 , indeks] = 0 

        if len(no_Outlier_TS) > 0:
            no_outlier_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
            counter += 1
            row_pointer += curr_field.shape[0]

  # Sanity check. Will neved occur. At least should not!
  no_outlier_df.drop_duplicates(inplace=True)


  noJump_df = pd.DataFrame(data = None,
                          index = np.arange(no_outlier_df.shape[0]), 
                          columns = no_outlier_df.columns)
  counter = 0
  row_pointer = 0

  for a_poly in IDs:
    # print(a_poly)
    # if (counter % 1000 == 0):
        # print ("counter is [{:.0f}].".format(counter))
    curr_field = no_outlier_df[no_outlier_df["pointID"]==a_poly].copy()
    # print(curr_field.shape)
    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)
    
    ################################################################

    no_Outlier_TS = nc.correct_big_jumps_1DaySeries_JFD(dataTMS_jumpie = curr_field, 
                                                        give_col = indeks, 
                                                        maxjump_perDay = 0.018)

    noJump_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
    counter += 1
    row_pointer += curr_field.shape[0]
  noJump_df.dropna(how='any', inplace=True)
  noJump_df.reset_index(drop=True, inplace=True)
  noJump_df['human_system_start_time'] = pd.to_datetime(noJump_df['human_system_start_time'])

  # Sanity check. Will neved occur. At least should not!
  print("Shape of noJump_df before dropping duplicates is {}.".format(noJump_df.shape)) 
  noJump_df.drop_duplicates(inplace=True)
  print("Shape of noJump_df before dropping duplicates is {}.".format(noJump_df.shape))

  noJump_df["pointID"] = noJump_df["pointID"].astype(int)

  noJump_df["human_system_start_time"] = pd.to_datetime(noJump_df["human_system_start_time"])

  window_size = 10
  reg_cols = ['pointID', 'human_system_start_time', "indeks"]
  IDs = np.sort(noJump_df["pointID"].unique())
  startYear = noJump_df["human_system_start_time"].dt.year.max()
  endYear = noJump_df["human_system_start_time"].dt.year.min()
  numberOfdays = (startYear - endYear + 1)*366

  nsteps = int(np.ceil(numberOfdays / window_size))

  nrows = nsteps * len(IDs)
  # print('st_yr is {}.'.format(startYear))
  # print('end_yr is {}.'.format(endYear))
  # print('nrows is {}.'.format(nrows))

  regular_df = pd.DataFrame(data = None,
                         index = np.arange(nrows), 
                         columns = reg_cols)

  counter = 0
  row_pointer = 0

  for a_poly in IDs:
    # print(a_poly)
    if (counter % 1000 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = noJump_df[noJump_df["pointID"]==a_poly].copy()
    # print(curr_field.shape)
    ################################################################
    # Sort by date (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)
    
    ################################################################
    regularized_TS = nc.regularize_a_field(a_df = curr_field, \
                                          V_idks = indeks, \
                                          interval_size = window_size,\
                                          start_year = startYear, \
                                          end_year = endYear)
    
    regularized_TS = nc.fill_theGap_linearLine(a_regularized_TS = regularized_TS, V_idx = indeks)
    if (counter == 0):
        print("regular_df columns:",     regular_df.columns)
        print("regularized_TS.columns", regularized_TS.columns)
    
    ################################################################
    # row_pointer = no_steps * counter
    
    """
      The reason for the following line is that we assume all years are 366 days!
      so, the actual thing might be smaller!
    """
    # why this should not work?: It may leave some empty rows in regular_df
    # but we drop them at the end.
    # print(regularized_TS.shape[0])
    # print(regular_df[row_pointer : (row_pointer+regularized_TS.shape[0])].shape)
    regular_df[row_pointer : (row_pointer+regularized_TS.shape[0])] = regularized_TS.values
    row_pointer += regularized_TS.shape[0]

    # right_pointer = row_pointer + min(no_steps, regularized_TS.shape[0])
    # print('right_pointer - row_pointer + 1 is {}!'.format(right_pointer - row_pointer + 1))
    # print('len(regularized_TS.values) is {}!'.format(len(regularized_TS.values)))
    # try:
    #     ### I do not know why the hell the following did not work for training set!
    #     ### So, I converted this to try-except statement! hopefully, this will
    #     ### work, at least as temporary remedy! Why it worked well with 2008-2021 but not 2013-2015
    #     regular_df[row_pointer: right_pointer] = regularized_TS.values
    # except:
    #     regular_df[row_pointer: right_pointer+1] = regularized_TS.values
    counter += 1

  regular_df['human_system_start_time'] = pd.to_datetime(regular_df['human_system_start_time'])
  regular_df.drop_duplicates(inplace=True)
  regular_df.dropna(inplace=True)

  # Sanity Check
  regular_df.sort_values(by=["pointID", 'human_system_start_time'], inplace=True)
  regular_df.reset_index(drop=True, inplace=True)

  # keep the last 33 data for each pointID
  for _ in regular_df.pointID.unique():
      regular_df[regular_df["pointID"] == _] =  regular_df[regular_df["pointID"] == _].tail(20).copy()
  regular_df.dropna(subset="indeks", inplace=True)
  regular_df.reset_index(inplace=True, drop=True)
  regular_dfs = pd.concat([regular_dfs,regular_df], axis=1)
  print(f"{indeks}", "done!")
print(indeks_list)
regular_dfs

# -

path = r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\2014-08-01_2015-07-31.csv"
df = pd.read_csv(path)
df.ni.describe()

regular_dfs[regular_dfs.indeks>=1]

# + id="Ba-PGgopklf8"
regular_dfs.to_csv(r"H:\My Drive\github\Ph.D._Projects\Tillage_Mapping\Codes\regular_dfs_2015_2016.csv")

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 236, "status": "ok", "timestamp": 1680748176134, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="G0fFoAavZRX1" outputId="57912ab2-76f2-4a3c-d654-3fa6ef569240"
unique_pointID_dataframe = noJump_df.copy().drop_duplicates(subset="pointID")
regular_dfs.columns
renamed_columns = ['pointID', 'human_system_start_time', 'NDVI', 'pointID',
       'human_system_start_time', 'EVI', 'pointID',
       'human_system_start_time', 'blue', 'pointID',
       'human_system_start_time', 'green', 'pointID',
       'human_system_start_time', 'red', 'pointID',
       'human_system_start_time', 'ni', 'pointID',
       'human_system_start_time', 'swi2']
regular_dfs.columns = renamed_columns
regular_dfs = regular_dfs.loc[:, ~regular_dfs.columns.duplicated()]
regular_dfs["pointID"].astype(int)
unique_pointID_dataframe.loc["pointID"] = unique_pointID_dataframe["pointID"].astype(int)
smoothed_df = pd.merge(regular_dfs, unique_pointID_dataframe[["pointID", "LastSurveyDate", "TotalAcres", "CropType"]], on="pointID")


smoothed_df.to_csv(r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\2015-08-01_2016-07-31.csv")
# -

smoothed_df

smoothed_df["NDVI"].astype(float).describe()

# + colab={"base_uri": "https://localhost:8080/", "height": 335} executionInfo={"elapsed": 506, "status": "ok", "timestamp": 1680653204261, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="h7fn8Bh1K5l4" outputId="48727b8d-c581-4bac-de1a-125c9447d8e8"
# Plot a field timeseries
field_1 = noJump_df[noJump_df["pointID"] == 11513]

# Plot
fig, ax = plt.subplots(1, 1, figsize=(20, 3),
                       sharex='col', sharey='row',
                       # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.2, 'wspace': .05})
ax.grid(True)

ax.scatter(field_1['human_system_start_time'], field_1["NDVI"], s=40, c='#d62728')
ax.plot(field_1['human_system_start_time'], field_1["NDVI"], 
        linestyle='-',  linewidth=3.5, color="#d62728", alpha=0.8,
        label="raw NDVI")
plt.ylim([-1, 1])
ax.legend(loc="lower right")
