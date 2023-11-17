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

import pandas as pd

cdl_2223 = pd.read_csv('/Users/aminnorouzi/Library/CloudStorage/OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/Tillage_Mapping/Data/CDL crop data/most_frequent_cdl_classes_survey_polygons_2223.csv')
cdl_2223['most_frequent_class'].unique()


to_replace ={
    23: 'Grain', 
    31: 'Canola',
    24: 'Grain', 
    51: 'Legume', 
    53: 'Legume', 
    61: 'Fallow/Idle Cropland', 
    52: 'Legume', 
    176: 'Grassland/Pasture',
    35: 'Mustard', 
    21: 'Grain',
    36: 'Alfalfa'
}
cdl_2223['most_frequent_class'] = cdl_2223['most_frequent_class'].replace(to_replace)
cdl_2223['most_frequent_class'].value_counts()
cdl_2223[['PriorCropT', 'most_frequent_class']]
for i in range(cdl_2223.shape[0]): 
    print([['PriorCropT', 'most_frequent_class']].iloc[i, :])


cdl_2223.columns


cdl_filtered = cdl_2223[['fid', 'most_frequent_class']]
cdl_df = cdl_filtered.loc[cdl_filtered['most_frequent_class'].isin(['Grain', 'Canola', 'Legume'])]
cdl_df

landsat_df = pd.read_csv('/Users/aminnorouzi/Library/CloudStorage/OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/Tillage_Mapping/Data/field_level_data/2223/field_level_main_glcm_seasonBased_joined_2223.csv')
landsat_df
df = pd.merge(landsat_df, cdl_df, on= 'fid')
df_2223 = df.loc[~df['ResidueCov'].isna()]
df_2223

# +
# final_df = pd.read_csv(r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\field_level_data\final_dataframe_landsat.csv")
final_df = pd.read_csv(
    "/Users/aminnorouzi/Library/CloudStorage/GoogleDrive-a.norouzikandelati@wsu.edu/My Drive/Ph.D._projects/Tillage_Mapping/Data/field_level_data/final_dataframe_landsat.csv")
final_df_withTillage = pd.read_csv(
    "/Users/aminnorouzi/Library/CloudStorage/GoogleDrive-a.norouzikandelati@wsu.edu/My Drive/Ph.D._projects/Tillage_Mapping/Data/field_level_data/finalCleaned.csv")
# crop_pred = pd.read_csv(r"H:\My Drive\P.h.D_Projects\Tillage_Mapping\Data\Crop_type_TS\crop_predictions_2021_2022.csv", index_col=0)
cdl_data = pd.read_csv(
    '/Users/aminnorouzi/Library/CloudStorage/GoogleDrive-a.norouzikandelati@wsu.edu/My Drive/Ph.D._projects/Tillage_Mapping/Data/CDL crop data/most_frequent_cdl_classes_survey_polygons.csv')

cdl_data.most_frequent_class.value_counts().index
replacement_dict = {24: 'grain', 23: 'grain', 51: 'legume',
                    51: 'legume', 31: 'canola', 53: 'legume',
                    21: 'grain', 51: 'legume', 52: 'legume',
                    28: 'grain'}

cdl_data['most_frequent_class'] = cdl_data['most_frequent_class'].replace(
    replacement_dict)
cdl_data = cdl_data.loc[cdl_data['most_frequent_class'].isin(
    ['grain', 'legume', 'canola'])]
cdl_data['most_frequent_class'].value_counts()
cdl_data = cdl_data[['pointID', 'most_frequent_class']]
cdl_data

final_df_rc = final_df[['pointID', 'ResidueCov']]
final_df_rest = final_df.drop(labels="ResidueCov", axis=1)

# Impute missing values with the median
final_df_rest = final_df_rest.fillna(final_df_rest.median())

final_df = pd.merge(final_df_rc, final_df_rest, on="pointID", how="left")

# Verify that all missing values have been imputed
print(final_df.isnull().sum())
final_df
rc_crop_df = pd.merge(final_df, cdl_data, on='pointID', how='left')
rc_crop_df

rc_crop_df.rename({"most_frequent_class": "cropType_pred"},
                  axis=1, inplace=True)
rc_crop_df["rc_crop"] = rc_crop_df["ResidueCov"] + \
    "_" + rc_crop_df["cropType_pred"]
rc_crop_df

merged_df = pd.merge(final_df, final_df_withTillage[[
                     'pointID', 'Tillage']], on='pointID', how='left')
last_df = merged_df.dropna(subset="Tillage", how="any")
last_df
last_df2 = pd.merge(last_df, cdl_data, on='pointID', how='left')
last_df2

last_df2.Tillage.value_counts()
last_df2.rename(columns={'most_frequent_class': 'Croptype'}, inplace=True)
last_df2
# -


last_df2['Croptype']



# +
rename = {
    'most_frequent_class': 'Croptype',
    'fid':'pointID'
}

df_2223.rename(columns=rename, inplace=True)
df_2223['pointID']

# +
# Identify common columns
common_columns = df_2223.columns.intersection(last_df2.columns)

# Filter DataFrames to keep only common columns
df_2223_filtered = df_2223[common_columns]
last_df2_filtered = last_df2[common_columns]

# Concatenate along axis=0 for common columns
result_df = pd.concat([df_2223_filtered, last_df2_filtered], axis=0)
result_df

# -

result_df.to_csv(
    '/Users/aminnorouzi/Library/CloudStorage/OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/Tillage_Mapping/Data/field_level_data/test_df.csv')

