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
#     display_name: tillenv
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# ### Prepare raw optical data

# +
folder_path = "H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\CropClassification\Optical & Radar (2015-2020)"

csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

data_dict = {}
for csv_file in csv_files: 
    file_name = os.path.splitext(csv_file)[0]
    file_path = os.path.join(folder_path, csv_file)
    data_dict[file_name] = pd.read_csv(file_path)
list(data_dict.keys())

# +
l7_2015 = data_dict['L7_T1C2L2_timeSeries_2014-08-01_2015-07-31']
l8_2015 = data_dict['L8_T1C2L2_timeSeries_2014-08-01_2015-07-31']
l7_2016 = data_dict['L7_T1C2L2_timeSeries_2015-08-01_2016-07-31']
l8_2016 = data_dict['L8_T1C2L2_timeSeries_2015-08-01_2016-07-31']
l7_2017 = data_dict['L7_T1C2L2_timeSeries_2016-08-01_2017-07-31']
l8_2017 = data_dict['L8_T1C2L2_timeSeries_2016-08-01_2017-07-31']
l7_2018 = data_dict['L7_T1C2L2_timeSeries_2017-08-01_2018-07-31']
l8_2018 = data_dict['L8_T1C2L2_timeSeries_2017-08-01_2018-07-31']
l7_2019 = data_dict['L7_T1C2L2_timeSeries_2018-08-01_2019-07-31']
l8_2019 = data_dict['L8_T1C2L2_timeSeries_2018-08-01_2019-07-31']
l7_2020 = data_dict['L7_T1C2L2_timeSeries_2019-08-01_2020-07-31']
l8_2020 = data_dict['L8_T1C2L2_timeSeries_2019-08-01_2020-07-31']

colnames = ['pointID', 'LastSurveyDate', 'Acres', 'CropType', 'NDVI', 'blue', 'green', 'red', 'ni', 'swi1',
            'swi2', 'system_start_time']

l7_2015.columns = colnames
l8_2015.columns = colnames
l7_2016.columns = colnames
l8_2016.columns = colnames
l7_2017.columns = colnames
l8_2017.columns = colnames
l7_2018.columns = colnames
l8_2018.columns = colnames
l7_2019.columns = colnames
l8_2019.columns = colnames
l7_2020.columns = colnames
l8_2020.columns = colnames

l78_15 = pd.concat([l7_2015, l8_2015])
l78_16 = pd.concat([l7_2016, l8_2016])
l78_17 = pd.concat([l7_2017, l8_2017])
l78_18 = pd.concat([l7_2018, l8_2018])
l78_19 = pd.concat([l7_2019, l8_2019])
l78_20 = pd.concat([l7_2020, l8_2020])

l78_15.dropna(how='any', subset=['NDVI', 'blue', 'green', 'red', 'ni', 'swi1', 'swi2'], inplace=True)
l78_16.dropna(how='any', subset=['NDVI', 'blue', 'green', 'red', 'ni', 'swi1', 'swi2'], inplace=True)
l78_17.dropna(how='any', subset=['NDVI', 'blue', 'green', 'red', 'ni', 'swi1', 'swi2'], inplace=True)
l78_18.dropna(how='any', subset=['NDVI', 'blue', 'green', 'red', 'ni', 'swi1', 'swi2'], inplace=True)
l78_19.dropna(how='any', subset=['NDVI', 'blue', 'green', 'red', 'ni', 'swi1', 'swi2'], inplace=True)
l78_20.dropna(how='any', subset=['NDVI', 'blue', 'green', 'red', 'ni', 'swi1', 'swi2'], inplace=True)


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
add_human_start_time_by_system_start_time(l78_15)
add_human_start_time_by_system_start_time(l78_16)
add_human_start_time_by_system_start_time(l78_17)
add_human_start_time_by_system_start_time(l78_18)
add_human_start_time_by_system_start_time(l78_19)
add_human_start_time_by_system_start_time(l78_20)

lsat_df = pd.concat([l78_15, l78_16, l78_17,
                l78_18, l78_19, l78_20])
# -

# ### Prepare raw radar data

# +
s1_2015 = data_dict['S1_timeSeries_2014-08-01_2015-07-31']
s1_2016 = data_dict['S1_timeSeries_2015-08-01_2016-07-31']
s1_2017 = data_dict['S1_timeSeries_2016-08-01_2017-07-31']
# s1_2018 = data_dict['S1_timeSeries_2017-08-01_2018-07-31']
s1_2019 = data_dict['S1_timeSeries_2018-08-01_2019-07-31']
s1_2020 = data_dict['S1_timeSeries_2019-08-01_2020-07-31']

colnames = ['pointID', 'LastSurveyDate', 'Acres', 'CropType', 'VV_dB', 'VH_dB',
       'system_start_time']
s1_2015.columns = colnames
s1_2016.columns = colnames
s1_2017.columns = colnames
# s1_2018.columns = colnames
s1_2019.columns = colnames
s1_2020.columns = colnames

s1_2015.dropna(how='any', subset=['VV_dB', 'VH_dB'], inplace=True)
s1_2016.dropna(how='any', subset=['VV_dB', 'VH_dB'], inplace=True)
s1_2017.dropna(how='any', subset=['VV_dB', 'VH_dB'], inplace=True)
# s1_2018.dropna(how='any', subset=['VV_dB', 'VH_dB'], inplace=True)
s1_2019.dropna(how='any', subset=['VV_dB', 'VH_dB'], inplace=True)
s1_2020.dropna(how='any', subset=['VV_dB', 'VH_dB'], inplace=True)


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
add_human_start_time_by_system_start_time(s1_2015)
add_human_start_time_by_system_start_time(s1_2016)
add_human_start_time_by_system_start_time(s1_2017)
# add_human_start_time_by_system_start_time(s1_2018)
add_human_start_time_by_system_start_time(s1_2019)
add_human_start_time_by_system_start_time(s1_2020)

s1_df = pd.concat([s1_2015, s1_2016, s1_2017,
                s1_2019, s1_2020])

# +
lsat_df['CropType'].unique()
value_mapping = {'Wheat':'grain', 'Barley':'grain', 'Pea, Dry':'legume',
                'Pea, Green':'legume', 'Wheat Fallow':'grain',
                'Canola':'Canola', 'Lentil':'legume', 'Bean, Garbanzo':'legume',
                'Bean, Dry':'legume', 'Bean, Green':'legume'}
lsat_df['CropType'] = lsat_df['CropType'].replace(value_mapping)
s1_df['CropType'] = s1_df['CropType'].replace(value_mapping)

lsat_df['CropType'].unique(), s1_df['CropType'].unique()
lsat_df.sort_values(by='human_system_start_time', inplace=True)
s1_df.sort_values(by='human_system_start_time', inplace=True)
# -

# ### Plot the plots

lsat_df

# +
# Create the plot
plt.figure(figsize=(10, 6))

# Iterate through each pointID to plot its time series
lsat_df_legume = lsat_df[lsat_df['CropType'] == 'legume']
for point_id in lsat_df_legume['pointID'].unique()[:20]:
    subset_df = lsat_df_legume[lsat_df_legume['pointID'] == point_id]
    plt.plot(subset_df['human_system_start_time'], subset_df['NDVI'], label=f'Point ID {point_id}')

# Customize the plot
plt.xlabel('Date')
plt.ylabel('NDVI')
plt.title('NDVI Time Series for Multiple Point IDs')
# plt.legend()
plt.grid(True)

# Show the plot
plt.show()
# -

for point_id in lsat_df_legume['pointID'].unique()[20:22]:
    print(point_id)

# +
# Create the plot
plt.figure(figsize=(10, 6))

# Iterate through each pointID to plot its time series
lsat_df_grain = lsat_df[lsat_df['CropType'] == 'grain']
for point_id in lsat_df_grain['pointID'].unique()[1:10]:
    subset_df = lsat_df_grain[lsat_df_grain['pointID'] == point_id].copy()
    plt.plot(subset_df['human_system_start_time'], subset_df['NDVI'], label=f'Point ID {point_id}')

# Customize the plot
plt.xlabel('Date')
plt.ylabel('NDVI')
plt.title('NDVI Time Series for Multiple Point IDs')
# plt.legend()
plt.grid(True)

# Show the plot
plt.show()
# -

lsat_df

subset_df
