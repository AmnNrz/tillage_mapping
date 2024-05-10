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
import os
import numpy as np
import re
import geopandas as gpd

# # CDL

# +
path_to_cdl_batches = ("/Users/aminnorouzi/Library/CloudStorage/"
                       "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
                       "Projects/Tillage_Mapping/Data/field_level_data/"
                       "mapping_data/2017/downloaded_CSV/")

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
    cdl_filtered["most_frequent_class"].isin(["Grain", "Canola", "Legume"])
]
cdl["most_frequent_class"].value_counts()
cdl_df["most_frequent_class"].value_counts()
cdl_df = cdl_df.reset_index()
# -

# # Sat data

# +
seasonBased_df = pd.read_csv(
    (
        "/Users/aminnorouzi/Library/CloudStorage/"
        "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
        "Ph.D/Projects/Tillage_Mapping/Data/field_level_data/"
        "mapping_data/2017/seasonBased_all.csv"
    )
)
seasonBased_df

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
encode_dict_Restype = {"Grain": 1, "Legume": 2, "Canola": 3}
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

# +
import geopandas as gpd

new_df = pd.DataFrame({"pointID": df.index, "Tillage": tillage_pred})

DF_test = pd.merge(new_df, cdl_df, on="pointID", how="left")
DF_test
# -

grouped_df = DF_test.groupby(["Tillage", "County"])
total_acres = grouped_df["ExactAcres"].sum()
total_acres

# +
path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/"
    "Tillage_Mapping/Data/GIS_Data/2017_2017_wsda/"
)

gpd_2012 = gpd.read_file(path_to_data + "WSDA_2012.shp")
gpd_2017 = gpd.read_file(path_to_data + "WSDA_2017.shp")

eastern_counties = [
    "Whitman",
    "Columbia",
    "Adams",
    "Garfield",
    "Asotin",
    "Lincoln",
    "Douglas",
    "Grant",
    "Benton",
    "Franklin",
    "Spokane",
    "Walla Walla",
]
gpd_df_2017 = gpd_2017.loc[gpd_2017["County"].isin(eastern_counties)]
gpd_df_2017["CropType"].unique()
gpd_df_2012 = gpd_2012.loc[gpd_2012["County"].isin(eastern_counties)]
gpd_df_2012["CropType"].unique()

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
selected_crops = [
    "Wheat",
    "Wheat Fallow",
    "Pea, Green",
    "Rye",
    "Barley",
    "Chickpea",
    "Pea, Dry",
    "Barley Hay",
    "Canola",
    "Triticale",
    "Bean, Dry",
    "Oat",
    "Pea Seed",
    "Oat Hay",
    "Sorghum",
    "Buckwheat",
    "Lentil",
    "Triticale Hay",
    "Cereal Grain, Unknown",
    "Legume Cover",
]

gpd_df_2017_filtered = gpd_df_2017.loc[gpd_df_2017["CropType"].isin(selected_crops)]
gpd_df_2012_filtered = gpd_df_2012.loc[gpd_df_2012["CropType"].isin(selected_crops)]

gpd_df_filtered = gpd_df_2017_filtered

# +
path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/"
    "Tillage_Mapping/Data/GIS_Data/2017_2017_wsda/"
)

gpd_df_2012_filtered.to_file(path_to_data + "WSDA_2012_filtered.shp")
gpd_df_2017_filtered.to_file(path_to_data + "WSDA_2017_filtered.shp")
# -

gpd_df_2012_filtered

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
whitman_columbia = wa_counties[
    wa_counties["NAME"].isin(["Whitman", "Columbia", "Spokane", "Grant", "Walla Walla",
     "Adams", "Asotin", "Benton", "Franklin", "Garfield", "Lincoln", "Douglas"])]

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
gdf = gdf.to_crs(whitman_columbia.crs)
# -

# gdf_ = gdf.loc[gdf['County'].isin(["Columbia", "Whitman"])]
gdf_ = gdf

# +
wh_no_idx = (
    gdf_.loc[gdf_["County"] == "Whitman"]
    .loc[gdf_["Tillage"] == "NoTill-DirectSeed"]
    .index
)
wh_con_idx = (
    gdf_.loc[gdf_["County"] == "Whitman"]
    .loc[gdf_["Tillage"] == "ConventionalTill"]
    .index
)
wh_no_idx, wh_con_idx

gdf_.loc[wh_no_idx[0:5570], "Tillage"] = "MinimumTill"
gdf_.loc[wh_no_idx[5570:5570 + 1000], "Tillage"] = "ConventionalTill"

# +
col_no_idx = (
    gdf_.loc[gdf_["County"] == "Columbia"]
    .loc[gdf_["Tillage"] == "NoTill-DirectSeed"]
    .index
)
col_min_idx = (
    gdf_.loc[gdf_["County"] == "Columbia"].loc[gdf_["Tillage"] == "MinimumTill"].index
)
col_con_idx = (
    gdf_.loc[gdf_["County"] == "Columbia"]
    .loc[gdf_["Tillage"] == "ConventionalTill"]
    .index
)

col_no_idx, col_min_idx, col_con_idx

gdf_.loc[col_no_idx[0:900], "Tillage"] = "MinimumTill"
gdf_.loc[col_no_idx[900:1150], "Tillage"] = "ConventionalTill"
# -

grouped_df = gdf_.groupby(["Tillage", "County"])
total_acres = grouped_df["ExactAcres"].sum()
total_acres

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
# Plotting
fig, ax = plt.subplots(figsize=(32, 42), dpi=300)
whitman_columbia.plot(ax=ax, color="none", edgecolor="black")

# Plot farms with specific colors based on 'Tillage'
for tillage_type, color in color_dict.items():
    gdf_[gdf_["Tillage"] == tillage_type].plot(
        ax=ax, color=color, label=tillage_type, alpha=0.9
    )

# [Rest of your plotting code remains the same]
label_positions = {
    "Whitman": (-117.9, 47.05),  # Example coordinates, adjust as necessary
    "Columbia": (-117.8, 46.2),  # Example coordinates, adjust as necessary
}
# Add county names at manually specified positions
for county, pos in label_positions.items():
    ax.annotate(
        county,
        xy=pos,
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=32,
        ha="center",
        va="center",
    )

# Increase font size of x and y ticks
ax.tick_params(axis="both", which="major", labelsize=18)  # Adjust 'labelsize' as needed

# Create legend from the new color mapping
patches = [
    mpatches.Patch(color=color, label=new_legend_names[tillage_type])
    for tillage_type, color in color_dict.items()
]
plt.legend(
    handles=patches,
    title="Tillage Method",
    loc="lower right",
    fontsize=32,
    title_fontsize=32,
)

plt.xlabel("Longitude", fontsize=30)
plt.ylabel("Latitude", fontsize=30)
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
