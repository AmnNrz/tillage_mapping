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
#     display_name: tillmap
#     language: python
#     name: python3
# ---

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

# test_df2017 = pd.read_csv(path_to_data + "mapping_data/2017/seasonBased_all.csv")
test_df2017 = pd.read_csv(path_to_data + "mapping_data/2012/seasonBased_all.csv")

best_Tillage_estimator = load(path_to_model + "best_Tillage_estimator.joblib")
best_RC_estimator = load(path_to_model + "best_RC_estimator.joblib")
# -

test_df2017

# +
important_features = ['pointID', 'sti_S0', 'ndi7_S0', 'crc_S0', 'ndti_S0', 'R_S2',
       'sndvi_S2', 'B_S0', 'SWIR2_S2', 'sti_S3', 'sndvi_S0', 'ndi5_S2',
       'ndvi_S1', 'evi_S3', 'ndvi_S0', 'sndvi_S3', 'aspect_savg', 'evi_S2',
       'crc_S2', 'gcvi_S2', 'sti_S1', 'NIR_S0', 'gcvi_S3', 'aspect', 'G_S0',
       'evi_S0', 'aspect_corr', 'ndvi_S2', 'SWIR1_S1', 'ndti_S1', 'ndi5_S0',
       'G_S2', 'NIR_S2', 'G_S3', 'elevation', 'ndi5_S3', 'gcvi_S0',
       'elevation_idm', 'ndi7_S2', 'B_S2', 'evi_S1', 'sti_S2', 'sndvi_S1',
       'ndti_S2', 'ndti_S3', 'aspect_idm', 'B_S3', 'gcvi_S1_asm', 'SWIR1_S0',
       'slope_ent']

test_df2017 = test_df2017.loc[:, important_features]
test_df2017 = test_df2017.set_index('pointID')
encode_dict_Restype = {"Grain": 1, "Legume": 2, "Canola": 3}
# test_df2017["ResidueType"] = test_df2017["ResidueType"].replace(encode_dict_Restype)
test_df2017 = test_df2017.fillna(test_df2017.median())
test_df2017['ResidueType'] = test_df2017['ResidueType'].astype(int)
test_df2017

# +
# Estimate RC

rc_pred = best_RC_estimator.predict(test_df2017)
pd.Series(rc_pred).value_counts()
test_df2017.insert(1, "ResidueCov", rc_pred)
# -

test_df2017.columns

tillage_pred = best_Tillage_estimator.predict(test_df2017)
tillage_pred

# +
import geopandas as gpd

df = pd.DataFrame({"pointID": test_df2017.index, "Tillage": tillage_pred})

path_to_shp = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
    "Ph.D/Projects/Tillage_Mapping/Data/GIS_Data/Crop_classification/"
)

shpfile_2017 = gpd.read_file(path_to_shp + "WSDA_2017/WSDACrop_2017_selection.shp")
shpfile_2017 = shpfile_2017.rename(columns={"PolygonID": "pointID"})
df = pd.merge(df, shpfile_2017, on="pointID", how="left")
df
# -

grouped_df = df.groupby(["Tillage", "County"])
total_acres = grouped_df["ExactAcres"].sum()
total_acres

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
whitman_columbia = wa_counties[wa_counties["NAME"].isin(["Whitman", "Columbia"])]
# -

gdf = gpd.GeoDataFrame(df, geometry="geometry")
gdf = gdf.to_crs(whitman_columbia.crs)

# +
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Unique categories in 'Tillage'
tillage_types = gdf["Tillage"].unique()

# Assign a color to each category
colors = plt.cm.viridis(np.linspace(0, 1, len(tillage_types)))
color_dict = dict(zip(tillage_types, colors))

# Plotting
fig, ax = plt.subplots(figsize=(32,42), dpi=300)
whitman_columbia.plot(ax=ax, color="none", edgecolor="black")

# Plot farms with specific colors based on 'Tillage'
for tillage_type, color in color_dict.items():
    gdf[gdf["Tillage"] == tillage_type].plot(
        ax=ax, color=color, label=tillage_type, alpha=0.9
    )

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

# Create legend from color mapping
patches = [
    mpatches.Patch(color=color, label=tillage_type)
    for tillage_type, color in color_dict.items()
]
plt.legend(
    handles=patches,
    title="Tillage Method",
    loc="lower right",
    fontsize=32,
    title_fontsize=32,
)
# Map of Tillage in Whitman and Columbia Counties of Washington State (2017)

plt.xlabel("Longitude", fontsize=30)
plt.ylabel("Latitude", fontsize=30)
plt.show()

# +
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Unique categories in 'Tillage'
tillage_types = gdf["Tillage"].unique()

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
    gdf[gdf["Tillage"] == tillage_type].plot(
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

color_dict

# +
import geopandas as gpd
import matplotlib.pyplot as plt

# Filter for just Washington state
washington_state = us_states[us_states["NAME"] == "Washington"].copy()
wa_counties = us_counties[us_counties["STATE_NAME"] == "Washington"]
whitman_columbia = wa_counties[wa_counties["NAME"].isin(["Whitman", "Columbia"])]


# Filter for Whitman and Columbia counties in Washington state
counties_to_highlight = wa_counties[wa_counties["NAME"].isin(["Whitman", "Columbia"])]

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the state boundary
washington_state.plot(ax=ax, color="white", edgecolor="black")

# Highlight Whitman and Columbia counties with a red line
counties_to_highlight.plot(ax=ax, color="red", edgecolor="red", linewidth=2)

# Remove axis for better presentation
ax.axis("off")

plt.show()
