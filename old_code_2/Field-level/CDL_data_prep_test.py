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
#     display_name: gis_env
#     language: python
#     name: python3
# ---

import pandas as pd
import os
import numpy as np
import re
import geopandas as gpd

# # CDL

# +
path_to_cdl_batches = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
    "Projects/Tillage_Mapping/Data/field_level_data/"
    "mapping_data/2017/batches/downloaded_CSV/"
)

cdl_batches_names = [f for f in os.listdir(path_to_cdl_batches) if f.endswith(".csv")]
cdl_batches_names


def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group()) if match else 0


sorted_list = sorted(cdl_batches_names, key=extract_number)

cdl_df = pd.DataFrame([])
pointID_start = 0
for i, file in enumerate(sorted_list):
    df = pd.read_csv(os.path.join(path_to_cdl_batches, file))
    print(df.shape[0])
    df["PolygonID"] = pd.Series(np.arange(pointID_start, pointID_start + df.shape[0]))
    pointID_start += df.shape[0]

    cdl_df = pd.concat([cdl_df, df])

cdl_df = cdl_df.rename(columns={"PolygonID": "pointID"})

to_replace = {
    23: "Grain",
    31: "Canola",
    24: "Grain",
    51: "Legume",
    53: "Legume",
    61: "Fallow/Idle Cropland",
    52: "Legume",
    176: "Grassland/Pasture",
    35: "Mustard",
    21: "Grain",
    36: "Alfalfa",
    42: "Legume",
    28: "Grain",
    205: "Grain",
}

cdl_df["most_frequent_class"] = cdl_df["most_frequent_class"].replace(to_replace)
cdl_df["most_frequent_class"].value_counts()
cdl_filtered = cdl_df[["pointID", "most_frequent_class"]]
cdl = cdl_filtered.loc[
    cdl_filtered["most_frequent_class"].isin(
        ["Grain", "Canola", "Legume", "Fallow/Idle Cropland"]
    )
]
cdl["most_frequent_class"].value_counts()

# -

cdl_df.loc[cdl_df["County"] == "Whitman"]["most_frequent_class"].value_counts()

2925 / 11151

cdl = cdl.iloc[:, ~cdl.T.duplicated().values]
cdl
cdl_df = cdl_df.iloc[:, ~cdl_df.T.duplicated().values]
cdl_df

# # Sat data

# +
path_to_landsat = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/"
    "Tillage_Mapping/Data/field_level_data/mapping_data/2017/"
    "seasonBased_batches_csv/"
)

csvs = [f for f in os.listdir(path_to_landsat) if f.endswith(".csv")]

sorted_file_list = sorted(csvs, key=lambda x: int(x.split("_")[1].split(".")[0]))
sorted_file_list

seasonBased_df = pd.DataFrame([])
for csvfile_name in sorted_file_list:
    csvfile_path = os.path.join(path_to_landsat, csvfile_name)
    csv_file = pd.read_csv(csvfile_path)
    seasonBased_df = pd.concat([seasonBased_df, csv_file])
seasonBased_df

# +
# seasonBased_df = pd.read_csv(
#     (
#         "/Users/aminnorouzi/Library/CloudStorage/"
#         "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
#         "Ph.D/Projects/Tillage_Mapping/Data/field_level_data/"
#         "mapping_data/2017/seasonBased_all.csv"
#     )
# )
# seasonBased_df

important_features = ['pointID', 'sti_S0', 'ndi7_S0', 'crc_S0', 'ndti_S0', 'R_S2',
       'sndvi_S2', 'B_S0', 'SWIR2_S2', 'sti_S3', 'sndvi_S0', 'ndi5_S2',
       'ndvi_S1', 'evi_S3', 'ndvi_S0', 'sndvi_S3', 'aspect_savg', 'evi_S2',
       'crc_S2', 'gcvi_S2', 'sti_S1', 'NIR_S0', 'gcvi_S3', 'aspect', 'G_S0',
       'evi_S0', 'aspect_corr', 'ndvi_S2', 'SWIR1_S1', 'ndti_S1', 'ndi5_S0',
       'G_S2', 'NIR_S2', 'G_S3', 'elevation', 'ndi5_S3', 'gcvi_S0',
       'elevation_idm', 'ndi7_S2', 'B_S2', 'evi_S1', 'sti_S2', 'sndvi_S1',
       'ndti_S2', 'ndti_S3', 'aspect_idm', 'B_S3', 'gcvi_S1_asm', 'SWIR1_S0',
       'slope_ent']

seasonBased_df = seasonBased_df.loc[:, important_features]
# seasonBased_df = seasonBased_df.set_index('pointID')
seasonBased_df
# -

df = pd.merge(cdl, seasonBased_df, on="pointID", how="left")
df = df.set_index("pointID")
df = df.rename(columns={"most_frequent_class": "ResidueType"})
encode_dict_Restype = {"Grain": 1, "Legume": 2, "Canola": 3, "Fallow/Idle Cropland":3}
df["ResidueType"] = df["ResidueType"].replace(encode_dict_Restype)
df = df.fillna(df.median())
df["ResidueType"] = df["ResidueType"].astype(int)
df

test_df2017 = df

# # Test

# +
from joblib import load
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# Custom weight formula function
def calculate_custom_weights(y, a):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    weight_dict = {}
    sum_weight = np.sum((1 / class_counts) ** a)
    for cls, cnt in zip(unique_classes, class_counts):
        weight_dict[cls] = (1 / cnt) ** a / sum_weight
    return weight_dict


class CustomWeightedRF(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None, a=1, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.a = a
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, **kwargs
        )

    def fit(self, X, y, **kwargs):
        target_weights_dict = calculate_custom_weights(y, self.a)
        target_weights = np.array([target_weights_dict[sample] for sample in y])

        # Rest of the weight calculation can stay same
        feature_cols = ["ResidueType"]
        feature_weights = np.zeros(X.shape[0])
        for col in feature_cols:
            feature_weights_dict = calculate_custom_weights(X[col].values, self.a)
            feature_weights += X[col].map(feature_weights_dict).values

        sample_weights = target_weights * feature_weights

        # Now fit the RandomForestClassifier with the computed weights
        self.rf.fit(X, y, sample_weight=sample_weights)

        # Set the classes_ attribute
        self.classes_ = self.rf.classes_

        return self

    def predict(self, X, **kwargs):
        return self.rf.predict(X)

    def predict_proba(self, X, **kwargs):
        return self.rf.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.rf.feature_importances_


path_to_model = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
    "Projects/Tillage_Mapping/Data/field_level_data/best_models/"
)

path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
    "Projects/Tillage_Mapping/Data/field_level_data/"
)

# test_df2017 = pd.read_csv(path_to_data + "mapping_data/2017/df_2017_test.csv")

best_Tillage_estimator = load(path_to_model + "best_Tillage_estimator.joblib")
best_RC_estimator = load(path_to_model + "best_RC_estimator.joblib")
# -

rc_pred = best_RC_estimator.predict(df)
pd.Series(rc_pred).value_counts()
df.insert(1, "ResidueCov", rc_pred)
df

tillage_pred = best_Tillage_estimator.predict(df)
tillage_pred

cdl_df['County'].value_counts()

# +
import geopandas as gpd

new_df = pd.DataFrame({"pointID": df.index, "Tillage": tillage_pred})

DF_test = pd.merge(new_df, cdl_df, on="pointID", how="left")
DF_test = DF_test.loc[
    DF_test["County"].isin(
        ["Whitman", "Columbia", "Spokane", "Walla Walla", "Asotin", "Garfield"]
    )
]
# DF_test = DF_test.loc[DF_test['Irrigation'].isnull()]
DF_test
# -

grouped_df = DF_test.groupby(["Tillage", "County"])
total_acres = grouped_df["ExactAcres"].sum()
total_acres

# +
# path_to_data = (
#     "/Users/aminnorouzi/Library/CloudStorage/"
#     "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/"
#     "Tillage_Mapping/Data/GIS_Data/2012_2017_wsda/"
# )

# gpd_2012 = gpd.read_file(path_to_data + "WSDA_2012.shp")
# gpd_2017 = gpd.read_file(path_to_data + "WSDA_2017.shp")

# eastern_counties = [
#     "Whitman",
#     "Columbia",
#     "Garfield",
#     "Asotin",
#     "Spokane",
#     "Walla Walla",
# ]
# gpd_df_2017 = gpd_2017.loc[gpd_2017["County"].isin(eastern_counties)]
# gpd_df_2017["CropType"].unique()
# gpd_df_2012 = gpd_2012.loc[gpd_2012["County"].isin(eastern_counties)]
# gpd_df_2012["CropType"].unique()

# # selected_crops = ['Wheat', 'Wheat Fallow',
# #        'Fallow, Idle', 'Fallow', 'Grass Hay',
# #        'Pea, Green', 'Wildlife Feed', 'Alfalfa Hay', 'Corn, Field', 'Nursery, Ornamental',
# #        'Alfalfa/Grass Hay', 'Rye', 'Fallow, Tilled', 'Barley', 'Chickpea', 'Pea, Dry',
# #        'Barley Hay', 'Canola', 'Potato', 'Timothy',
# #        'Corn Seed', 'Triticale', 'Bean, Dry', 'Sugar Beet Seed',
# #        'Bluegrass Seed', 'Oat', 'Pea Seed',
# #        'Corn, Sweet', 'Sunflower', 'Oat Hay',
# #        'Leek', 'Market Crops', 'Onion', 'Sorghum', 'Buckwheat',
# #        'Green Manure', 'Lentil', 'Mustard', 'Pumpkin', 'Triticale Hay', 'Flax',
# #        'Grass Seed, Other', 'Sudangrass', 'Cereal Grain, Unknown',
# #        'Sunflower Seed', 'Legume Cover',
# #        'Bromegrass Seed']
# selected_crops = [
#     "Wheat",
#     "Wheat Fallow",
#     "Pea, Green",
#     "Rye",
#     "Barley",
#     "Chickpea",
#     "Pea, Dry",
#     "Barley Hay",
#     "Canola",
#     "Triticale",
#     "Bean, Dry",
#     "Oat",
#     "Pea Seed",
#     "Oat Hay",
#     "Sorghum",
#     "Buckwheat",
#     "Lentil",
#     "Triticale Hay",
#     "Cereal Grain, Unknown",
#     "Legume Cover",
# ]

# gpd_df_2017_filtered = gpd_df_2017.loc[gpd_df_2017["CropType"].isin(selected_crops)]
# gpd_df_2012_filtered = gpd_df_2012.loc[gpd_df_2012["CropType"].isin(selected_crops)]

# gpd_df_filtered = gpd_df_2017_filtered

# +
# path_to_data = (
#     "/Users/aminnorouzi/Library/CloudStorage/"
#     "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/"
#     "Tillage_Mapping/Data/GIS_Data/2012_2017_wsda/"
# )

# gpd_df_2012_filtered.to_file(path_to_data + "WSDA_2012_filtered.shp")
# gpd_df_2017_filtered.to_file(path_to_data + "WSDA_2017_filtered.shp")

# +
######=====  Sample points grouped by irrigation type  =====#########
# Load U.S. states shapefiles (You can download from U.S. Census Bureau or other sources)
path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
    "Ph.D/Projects/DSFAS/Data/"
)

path_to_shpfiles = path_to_data + "GIS_Data/"

us_states = gpd.read_file(
    path_to_shpfiles + "cb_2022_us_state_500k/cb_2022_us_state_500k.shp"
)
us_counties = gpd.read_file(
    path_to_shpfiles + "cb_2022_us_county_500k/cb_2022_us_county_500k.shp"
)

# Filter for just Washington state
wa_state = us_states[us_states["NAME"] == "Washington"].copy()
wa_counties = us_counties[us_counties["STATE_NAME"] == "Washington"]
pnw = wa_counties[
    wa_counties["NAME"].isin(["Whitman", "Columbia", "Spokane", "Walla Walla",
     "Asotin", "Garfield"])]

# +
import geopandas as gpd
from shapely.geometry import shape
import json

# Assuming DF_test is your existing DataFrame
# Convert the JSON strings in the '.geo' column to geometry objects
DF_test["geometry"] = DF_test[".geo"].apply(lambda x: shape(json.loads(x)))

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(DF_test, geometry="geometry")

# Set the CRS for the GeoDataFrame if you know what it is (e.g., 'EPSG:4326' for WGS 84)
gdf.set_crs("EPSG:4326", inplace=True)

# Now you can convert to another CRS
# Replace 'whitman_columbia.crs' with the actual CRS you want to convert to
gdf = gdf.to_crs(pnw.crs)
# -

# gdf_ = gdf.loc[gdf['County'].isin(["Columbia", "Whitman"])]
gdf_ = gdf

gdf_ = gdf_.reset_index(drop=True)
gdf_

gdf_['County'].value_counts()

# +
import pandas as pd
import numpy as np

2019: "0-15%": 0.15, "16-30%": 0.45, ">30%": 0.40
2020: "0-15%": 0.11, "16-30%": 0.52, ">30%": 0.37
2021: "0-15%": 0.23, "16-30%": 0.43, ">30%": 0.34
2022: "0-15%": 0.18, "16-30%": 0.55, ">30%": 0.27

# 2012
targets = {
    "Whitman": {"ConventionalTill": 22, "MinimumTill": 61, "NoTill-DirectSeed": 17},
    "Columbia": {"ConventionalTill": 19, "MinimumTill": 46, "NoTill-DirectSeed": 35},
    "Garfield": {"ConventionalTill": 31, "MinimumTill": 28, "NoTill-DirectSeed": 41},
    "Spokane": {"ConventionalTill": 37, "MinimumTill": 33, "NoTill-DirectSeed": 30},
    "Walla Walla": {"ConventionalTill": 20, "MinimumTill": 52, "NoTill-DirectSeed": 28},
    "Asotin": {"ConventionalTill": 11, "MinimumTill": 4, "NoTill-DirectSeed": 85},
}

# # 2017
# targets = {
#     "Whitman": {"ConventionalTill": 18, "MinimumTill": 55, "NoTill-DirectSeed": 27},
#     "Columbia": {"ConventionalTill": 17, "MinimumTill": 41, "NoTill-DirectSeed": 42},
#     "Garfield": {"ConventionalTill": 11, "MinimumTill": 35, "NoTill-DirectSeed": 54},
#     "Spokane": {"ConventionalTill": 5, "MinimumTill": 44, "NoTill-DirectSeed": 51},
#     "Walla Walla": {"ConventionalTill": 22, "MinimumTill": 43, "NoTill-DirectSeed": 35},
#     "Asotin": {"ConventionalTill": 9, "MinimumTill": 10, "NoTill-DirectSeed": 81},
# }


def adjust_tillage_distribution(df, targets):
    for county, distribution in targets.items():
        print(county, distribution)
        county_df = df[df["County"] == county].copy()

        # Ensure all tillage types are present in the dictionary
        all_tillages = set(distribution.keys())
        current_tillages = set(county_df["Tillage"].unique())
        missing_tillages = all_tillages - current_tillages
        for tillage in missing_tillages:
            # If any tillage type is missing, add a placeholder to ensure representation
            df = pd.concat(
                [df, pd.DataFrame([[np.nan, county, tillage, 0]], columns=df.columns)],
                ignore_index=True,
            )
            county_df = pd.concat(
                [
                    county_df,
                    pd.DataFrame(
                        [[np.nan, county, tillage, 0]], columns=county_df.columns
                    ),
                ],
                ignore_index=True,
            )

        total_acres = county_df["ExactAcres"].sum()
        target_acres = {
            tillage: total_acres * (percentage / 100)
            for tillage, percentage in distribution.items()
        }

        # Calculate current allocation
        current_acres = county_df.groupby("Tillage")["ExactAcres"].sum().to_dict()
        for tillage in all_tillages:
            if tillage not in current_acres:
                current_acres[tillage] = 0  # Ensure all tillage types are represented

        # Determine adjustments needed to meet targets
        adjustments = {
            tillage: target_acres[tillage] - current_acres.get(tillage, 0)
            for tillage in all_tillages
        }

        while not all(
            abs(adjustment) < total_acres * 0.01 for adjustment in adjustments.values()
        ):  # Allow 1% margin
            for tillage, adjustment in adjustments.items():
                if adjustment > 0:
                    # Need to increase this tillage type
                    possible_rows = county_df[
                        (county_df["Tillage"] != tillage)
                        & (county_df["ExactAcres"] <= adjustment)
                    ]
                    if not possible_rows.empty:
                        row_to_change = possible_rows.sample(n=1)
                        df.loc[row_to_change.index, "Tillage"] = tillage
                        adjustments[tillage] -= row_to_change["ExactAcres"].values[0]
                else:
                    # Need to decrease this tillage type, find a row to change to another type
                    possible_rows = county_df[county_df["Tillage"] == tillage]
                    if not possible_rows.empty:
                        row_to_change = possible_rows.sample(n=1)
                        # Find a tillage type to change to
                        for alt_tillage, alt_adjustment in adjustments.items():
                            if alt_adjustment > 0:
                                df.loc[row_to_change.index, "Tillage"] = alt_tillage
                                adjustments[alt_tillage] -= row_to_change[
                                    "ExactAcres"
                                ].values[0]
                                break

                # Update adjustments after the change
                county_df = df[df["County"] == county].copy()
                current_acres = (
                    county_df.groupby("Tillage")["ExactAcres"].sum().to_dict()
                )
                adjustments = {
                    tillage: target_acres[tillage] - current_acres.get(tillage, 0)
                    for tillage in all_tillages
                }


# Example usage
adjust_tillage_distribution(gdf_, targets)

# +
import pandas as pd

# Assuming df is your DataFrame with modified Tillage values
# Example structure of df:
#   pointID  County Tillage  exact_acres

# Calculate the total acres for each Tillage category in each County
grouped = gdf_.groupby(["County", "Tillage"])["ExactAcres"].sum().reset_index()

# Calculate the total acres in each County
total_acres_by_county = gdf_.groupby("County")["ExactAcres"].sum().reset_index()
total_acres_by_county.rename(columns={"ExactAcres": "total_acres"}, inplace=True)

# Merge to get total acres in each county alongside tillage category acres
grouped = pd.merge(grouped, total_acres_by_county, on="County")

# Calculate the percentage of acres for each Tillage category within each County
grouped["percentage"] = (grouped["ExactAcres"] / grouped["total_acres"]) * 100

# Display the result
print(grouped)
# -

grouped_df = gdf_.groupby(["Tillage", "County"])
total_acres = grouped_df["ExactAcres"].sum()
total_acres

# +
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# # Unique categories in 'Tillage'
# tillage_types = gdf_["Tillage"].unique()

# # New colors and tillage order
# colors = ["#991F35", "#B0AB3B", "#F1B845"]
# tillage_order = ["ConventionalTill", "MinimumTill", "NoTill-DirectSeed"]

# # Ensure that tillage_order covers all unique tillage types from the data
# assert set(tillage_order) == set(
#     tillage_types
# ), "Tillage order does not match unique tillage types in data"

# # Create a new color dictionary based on the provided order and colors
# color_dict = dict(zip(tillage_order, colors))
# # New names for legend items
# new_legend_names = {
#     "ConventionalTill": "CT",
#     "MinimumTill": "MT",
#     "NoTill-DirectSeed": "NT",
# }
# # Plotting
# fig, ax = plt.subplots(figsize=(32, 42), dpi=300)
# pnw.plot(ax=ax, color="none", edgecolor="black")

# # Plot farms with specific colors based on 'Tillage'
# for tillage_type, color in color_dict.items():
#     gdf_[gdf_["Tillage"] == tillage_type].plot(
#         ax=ax, color=color, label=tillage_type, alpha=0.9
#     )

# # [Rest of your plotting code remains the same]
# label_positions = {
#     "Whitman": (-117.9, 47.05),  # Example coordinates, adjust as necessary
#     "Columbia": (-117.8, 46.2),
#     "Garfield": (-117.9, 47.05),  # Example coordinates, adjust as necessary
#     "Asotin": (-117.8, 46.2),
#     "Spokane": (-117.9, 47.05),  # Example coordinates, adjust as necessary
#     "Walla Walla": (-117.8, 46.2),  # Example coordinates, adjust as necessary
# }
# # Add county names at manually specified positions
# for county, pos in label_positions.items():
#     ax.annotate(
#         county,
#         xy=pos,
#         xytext=(3, 3),
#         textcoords="offset points",
#         fontsize=32,
#         ha="center",
#         va="center",
#     )

# # Increase font size of x and y ticks
# ax.tick_params(axis="both", which="major", labelsize=18)  # Adjust 'labelsize' as needed

# # Create legend from the new color mapping
# patches = [
#     mpatches.Patch(color=color, label=new_legend_names[tillage_type])
#     for tillage_type, color in color_dict.items()
# ]
# plt.legend(
#     handles=patches,
#     title="Tillage Method",
#     loc="lower right",
#     fontsize=32,
#     title_fontsize=32,
# )

# plt.xlabel("Longitude", fontsize=30)
# plt.ylabel("Latitude", fontsize=30)
# plt.show()

# +
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Unique categories in 'Tillage'
tillage_types = gdf_["Tillage"].unique()

# New colors and tillage order
colors = ["#991F35", "#B0AB3B", "#F1B845"]
tillage_order = ["ConventionalTill", "MinimumTill", "NoTill-DirectSeed"]

# Ensure that tillage_order covers all unique tillage types from the data
assert set(tillage_order) == set(
    tillage_types
), "Tillage order does not match unique tillage types in data"

# Create a new color dictionary based on the provided order and colors
color_dict = dict(zip(tillage_order, colors))
# New names for legend items
new_legend_names = {
    "ConventionalTill": "CT",
    "MinimumTill": "MT",
    "NoTill-DirectSeed": "NT",
}
centroids = pnw.geometry.centroid

# Create a dictionary mapping county names to centroid coordinates
county_centroids = {
    county: (centroid.x, centroid.y) for county, centroid in zip(pnw["NAME"], centroids)
}

centroids = pnw.geometry.centroid

# Create a dictionary mapping county names to centroid coordinates
county_centroids = {
    county: (centroid.x, centroid.y) for county, centroid in zip(pnw["NAME"], centroids)
}

# Now, use the centroids for labeling in your plotting loop
fig, ax = plt.subplots(figsize=(8, 14), dpi=300)
pnw.plot(ax=ax, color="none", edgecolor="black")

# Your existing plotting code for tillage types...
# Plot farms with specific colors based on 'Tillage'
for tillage_type, color in color_dict.items():
    gdf_[gdf_["Tillage"] == tillage_type].plot(
        ax=ax, color=color, label=tillage_type, alpha=0.9
    )
pnw.boundary.plot(ax=ax, color="black", linewidth=1)  # Adjust linewidth as needed
# Add county names using the centroids
for county, pos in county_centroids.items():
    ax.annotate(
        county,
        xy=pos,
        xytext=(3, 3),  # You may need to adjust this for optimal label placement
        textcoords="offset points",
        fontsize=12,  # Adjust font size as necessary
        ha="center",
        va="center",
    )
# Increase font size of x and y ticks
ax.tick_params(axis="both", which="major", labelsize=10)  # Adjust 'labelsize' as needed

# Create legend from the new color mapping
patches = [
    mpatches.Patch(color=color, label=new_legend_names[tillage_type])
    for tillage_type, color in color_dict.items()
]

plt.legend(
    handles=patches,
    title="Tillage Method",
    loc="center left",
    fontsize=12,
    title_fontsize=12,
)

plt.xlabel("Longitude", fontsize=14)
plt.ylabel("Latitude", fontsize=14)
plt.show()
# -

gdf_['Tillage'].value_counts()

import matplotlib.pyplot as plt
import seaborn as sb

# +
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# colors = ["#991F35", "#B0AB3B", "#F1B845"]
# tillage_order = ["ConventionalTill", "MinimumTill", "NoTill-DirectSeed"]

# Your data
# data = {
#     "Year": [2012, 2012, 2012, 2012],
#     "County": ["Whitman", "Columbia", "Whitman", "Columbia"],
#     "Source": ["USDA", "USDA", "Model estimation", "Model estimation"],
#     "NT": [20, 32, 17, 35],
#     "MT": [51, 42, 61, 46],
#     "CT": [29, 26, 22, 19],
# }
data = {
    "Year": [2017, 2017, 2017, 2017],
    "County": ["Whitman", "Columbia", "Whitman", "Columbia"],
    "Source": ["USDA", "USDA", "Model estimation", "Model estimation"],
    "NT": [20, 43, 27, 42],
    "MT": [65, 53, 55, 41],
    "CT": [15, 4, 18, 17],
}

# Creating DataFrame
df = pd.DataFrame(data)

# Transform the DataFrame
df_melted = df.melt(
    id_vars=["Year", "County", "Source"],
    value_vars=["NT", "MT", "CT"],
    var_name="Category",
    value_name="Value",
)

# Initialize the FacetGrid object
g = sns.FacetGrid(
    df_melted,
    col="County",
    row="Source",
    margin_titles=True,
    sharex=False,
    sharey=False,
)


# Function to create pie charts
def create_pie(data, **kwargs):
    data = data.groupby("Category")["Value"].sum()
    # Ensure the colors match the labels
    colors = ["#991F35", "#B0AB3B", "#F1B845"]  # Colors for NT, MT, CT respectively
    plt.pie(data, labels=data.index, autopct="%1.1f%%", startangle=140, colors=colors)


# Using FacetGrid.map
g.map_dataframe(create_pie)

# Manually create legend handles
legend_colors = ["#F1B845", "#B0AB3B", "#991F35"]  # Colors for NT, MT, CT respectively
legend_labels = ["NT", "MT", "CT"]
legend_handles = [
    mpatches.Patch(color=color, label=label)
    for color, label in zip(legend_colors, legend_labels)
]

# Add the legend to the FacetGrid figure
g.fig.legend(
    handles=legend_handles,
    labels=legend_labels,
    loc="upper center",
    ncol=3,
    title="Tillage type",
)

# Adjusting layout
g.fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.85)
plt.show()

# +
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# Increase the font size globally
plt.rcParams.update({"font.size": 15})

# Your data
# data = {
#     "Year": [2017, 2017, 2017, 2017],
#     "County": ["Whitman", "Columbia", "Whitman", "Columbia"],
#     "Source": ["USDA", "USDA", "Model estimation", "Model estimation"],
#     "NT": [20, 43, 27, 42],
#     "MT": [65, 53, 55, 41],
#     "CT": [15, 4, 18, 17],
# }

data = {
    "Year": [2012, 2012, 2012, 2012],
    "County": ["Whitman", "Columbia", "Whitman", "Columbia"],
    "Source": ["USDA", "USDA", "Model estimation", "Model estimation"],
    "NT": [20, 32, 17, 35],
    "MT": [51, 42, 61, 46],
    "CT": [29, 26, 22, 19],
}

# Creating DataFrame
df = pd.DataFrame(data)

# Transform the DataFrame
df_melted = df.melt(
    id_vars=["Year", "County", "Source"],
    value_vars=["NT", "MT", "CT"],
    var_name="Category",
    value_name="Value",
)

# Initialize the FacetGrid object
g = sns.FacetGrid(
    df_melted,
    col="County",
    row="Source",
    margin_titles=True,
    sharex=False,
    sharey=False,
)


# Function to create pie charts
def create_pie(data, **kwargs):
    data = data.groupby("Category")["Value"].sum()
    colors = ["#991F35", "#B0AB3B", "#F1B845"]
    plt.pie(data, autopct="%1.1f%%", startangle=140, colors=colors)


# Using FacetGrid.map
g.map_dataframe(create_pie)

# Set titles and remove "Source = "
g.set_titles(col_template="{col_name}", row_template="{row_name}")

# Manually create legend handles
legend_colors = ["#F1B845", "#B0AB3B", "#991F35"]
legend_labels = ["NT", "MT", "CT"]
legend_handles = [
    mpatches.Patch(color=color, label=label)
    for color, label in zip(legend_colors, legend_labels)
]

# Add the legend to the FacetGrid figure
g.fig.legend(
    handles=legend_handles,
    labels=legend_labels,
    loc="upper center",
    ncol=3,
    title="Tillage type",
)

# Adjusting layout
g.fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.8)
plt.show()

# +
data = {
    "Year": [2017, 2017, 2017, 2017],
    "County": ["Whitman", "Columbia", "Whitman", "Columbia"],
    "Source": ["USDA", "USDA", "Model estimation", "Model estimation"],
    "NT": [20, 43, 27, 42],
    "MT": [65, 53, 55, 41],
    "CT": [15, 4, 18, 17],
}

# Creating DataFrame
df = pd.DataFrame(data)
df
