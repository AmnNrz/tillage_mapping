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

import numpy as np
import pandas as pd
import geopandas as gpd
import os

path_to_data = ('/Users/aminnorouzi/Library/CloudStorage/'
                'OneDrive-WashingtonStateUniversity(email.wsu.edu)/'
                'Ph.D/Projects/Tillage_Mapping/Data/')

# # Read data

# +
data_2012 = pd.read_csv(
    path_to_data
    + "MAPPING_DATA_2011_2012_2022/mapped_data/dat/2012_data.csv"
)

data_2017 = pd.read_csv(
    path_to_data
    + "MAPPING_DATA_2011_2012_2022/mapped_data/dat/2017_data.csv"
)

data_2022 = pd.read_csv(
    path_to_data
    + "MAPPING_DATA_2011_2012_2022/mapped_data/dat/2022_data.csv"
)
# -

# # Predict fr

# +
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

# Load best fr classifer
fr_classifier = joblib.load(path_to_data + "best_models/best_fr_classifier.pkl")

# Load the saved scaler for fr
scaler = joblib.load(path_to_data + "best_models/fr_scaler_model.pkl")

# Apply PCA
# Load the PCA object used during training
pca = joblib.load(path_to_data + "best_models/fr_pca_model.pkl")

def pred_fr(df):
    x_imagery = df.loc[:, "B_S0_p0":]
    x_imagery_scaled = scaler.transform(x_imagery)

    x_imagery_pca = pca.transform(x_imagery_scaled)
    x_imagery_pca = pd.DataFrame(x_imagery_pca)
    x_imagery_pca.set_index(x_imagery.index, inplace=True)

    X = pd.concat(
        [
            df["cdl_cropType"],
            x_imagery_pca,
        ],
        axis=1,
    )
    X.columns = X.columns.astype(str)
    y_preds = fr_classifier.predict(X)
    df["fr_pred"] = y_preds
    cols = list(df.columns)
    # Move the merged column to the 4th position
    cols.insert(3, cols.pop(cols.index("fr_pred")))
    df = df[cols]
    return df
data_2022 = pred_fr(data_2022)
data_2017 = pred_fr(data_2017)
data_2012 = pred_fr(data_2012)
# -

# # Predict tillage

# +
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# Custom weight formula function
def calculate_custom_weights(y, a):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    weight_dict = {}
    sum_weight = np.sum((1 / class_counts) ** a)
    for cls, cnt in zip(unique_classes, class_counts):
        weight_dict[cls] = (1 / cnt) ** a / sum_weight
    return weight_dict


class CustomWeightedRF(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        a=1,
        max_features=None,
        min_samples_split=None,
        min_samples_leaf=None,
        bootstrap=None,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.min_samples_split = min_samples_split
        self.a = a
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, **kwargs
        )

    def fit(self, X, y, **kwargs):
        
        # Calculate the target weights based on 'a'
        target_weights_dict = calculate_custom_weights(y, self.a)
        target_weights = np.array([target_weights_dict[sample] for sample in y])
        if self.a == 0:
            X_mod = X.drop(columns=["cdl_cropType"])
            feature_weights = np.ones(X_mod.shape[0])
        else:
            X_mod = X.copy()
            feature_cols = ["cdl_cropType"]
            feature_weights = np.zeros(X_mod.shape[0])
            for col in feature_cols:
                feature_weights_dict = calculate_custom_weights(
                    X_mod[col].values, self.a
                )
                feature_weights += X_mod[col].map(feature_weights_dict).values

        # Calculate sample weights by combining target and feature weights
        sample_weights = target_weights * feature_weights

        # Fit the RandomForestClassifier with the computed weights and modified dataset
        self.rf.fit(X_mod, y, sample_weight=sample_weights)

        # Set the classes_ attribute
        self.classes_ = self.rf.classes_

        return self

    def predict(self, X, **kwargs):
        if self.a == 0:
            X_mod = X.drop(columns=["cdl_cropType"])
        else:
            X_mod = X.copy()
        return self.rf.predict(X_mod)

    def predict_proba(self, X, **kwargs):
        if self.a == 0:
            X_mod = X.drop(columns=["cdl_cropType"])
        else:
            X_mod = X.copy()
        return self.rf.predict_proba(X_mod)

    @property
    def feature_importances_(self):
        return self.rf.feature_importances_


# +
# Load best fr classifer
tillage_classifier = joblib.load(
    path_to_data + "best_models/best_tillage_classifier_config2.pkl"
)

# Load the saved scaler for fr
scaler = joblib.load(path_to_data + "best_models/tillage_scaler_model.pkl")

# Apply PCA
# Load the PCA object used during training
pca = joblib.load(path_to_data + "best_models/tillage_pca_model.pkl")

def pred_tillage(df):
    x_imagery = df.loc[:, "B_S0_p0":]
    x_imagery_scaled = scaler.transform(x_imagery)
    x_imagery_pca = pca.transform(x_imagery_scaled)
    x_imagery_pca = pd.DataFrame(x_imagery_pca)
    x_imagery_pca.set_index(x_imagery.index, inplace=True)

    X = pd.concat(
        [
            df["cdl_cropType"],
            df["fr_pred"],
            x_imagery_pca,
        ],
        axis=1,
    )

    to_replace = {"0-15%": 1, "16-30%": 2, ">30%": 3}
    X["fr_pred"] = X["fr_pred"].replace(to_replace)
    X.columns = X.columns.astype(str)
    y_preds = tillage_classifier.predict(X)
    df["Tillage"] = y_preds
    cols = list(df.columns)
    # Move the merged column to the 4th position
    cols.insert(2, cols.pop(cols.index("Tillage")))
    df = df[cols]
    return df

mapped_2022 = pred_tillage(data_2022)
mapped_2017 = pred_tillage(data_2017)
mapped_2012 = pred_tillage(data_2012)

# +
mapped_2022.to_csv(
    path_to_data + "MAPPING_DATA_2011_2012_2022/mapped_data/mapped_2022.csv",
      index=False)

mapped_2017.to_csv(
    path_to_data + "MAPPING_DATA_2011_2012_2022/mapped_data/mapped_2017.csv",
    index=False,
)
mapped_2012.to_csv(
    path_to_data + "MAPPING_DATA_2011_2012_2022/mapped_data/mapped_2012.csv",
    index=False,
)
# -

shpfile_2022 = gpd.read_file(
    path_to_data + "MAPPING_DATA_2011_2012_2022/shapefiles/2012_2017_2022/WSDA_2022.shp"
)
shpfile_2017 = gpd.read_file(
    path_to_data + "MAPPING_DATA_2011_2012_2022/shapefiles/2012_2017_2022/WSDA_2017.shp"
)
shpfile_2012 = gpd.read_file(
    path_to_data + "MAPPING_DATA_2011_2012_2022/shapefiles/2012_2017_2022/WSDA_2012.shp"
)


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
    wa_counties["NAME"].isin(
        ["Whitman", "Columbia", "Spokane", "Walla Walla", "Asotin", "Garfield"]
    )
]
# -

# # Plot map of tillage for each county

# +
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

path_to_plots = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
    "Projects/Tillage_Mapping/plots/"
)

def map_county(mapped_data, shapfile):
    # Merge mapped_2012 with shpfile_2012 geometry column
    mapped_data = mapped_data.merge(
        shapfile[["pointID", "geometry"]], on="pointID", how="left"
    )

    # Convert to GeoDataFrame after merging
    mapped_data = gpd.GeoDataFrame(mapped_data, geometry="geometry")

    mapped_data.head(2)

    gdf = mapped_data
    # Unique categories in 'Tillage'
    tillage_types = gdf["Tillage"].unique()

    # New colors and tillage order
    colors = ["#233D4D", "#FE7F2D", "#A1C181"]
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
        gdf[gdf["Tillage"] == tillage_type].plot(
            ax=ax, color=color, label=tillage_type, alpha=0.9
        )
    pnw.boundary.plot(ax=ax, color="black", linewidth=1)  # Adjust linewidth as needed

    # Increase font size of x and y ticks
    ax.tick_params(axis="both", which="major", labelsize=12)  # Adjust 'labelsize' as needed

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

    year = mapped_data["year"].iloc[0]
    # Save plot
    plt.savefig(path_to_plots + f"mapping/fig_{year}_map.png", dpi=500, bbox_inches="tight")

    plt.show()


map_county(mapped_2022, shpfile_2022)

map_county(mapped_2017, shpfile_2017)

map_county(mapped_2012, shpfile_2012)
# -

# # Paneled map

# +
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Function to plot a single map
def plot_map(
    gdf,
    ax,
    pnw,
    color_dict,
    new_legend_names,
    year,
    county_centroids,
    county_label_settings,
):
    # Plot farms with specific colors based on 'Tillage'
    for tillage_type, color in color_dict.items():
        gdf[gdf["Tillage"] == tillage_type].plot(
            ax=ax, color=color, label=tillage_type, alpha=0.9
        )
    pnw.boundary.plot(ax=ax, color="black", linewidth=1)  # Adjust linewidth as needed

    # Add county names with custom placement and rotation
    for county, pos in county_centroids.items():
        label_settings = county_label_settings.get(county, {})
        xytext = label_settings.get("xytext", (3, 3))  # Default offset for labels
        rotation = label_settings.get("rotation", 0)  # Default rotation angle
        ax.annotate(
            county,
            xy=pos,
            xytext=xytext,
            textcoords="offset points",
            fontsize=28,  # Larger font size for county labels
            ha="center",
            va="center",
            rotation=rotation,  # Apply rotation
        )

    # Add legend
    patches = [
        mpatches.Patch(color=color, label=new_legend_names[tillage_type])
        for tillage_type, color in color_dict.items()
    ]
    ax.legend(
        handles=patches,
        title="Tillage Method",
        loc="upper left",
        fontsize=28,
        title_fontsize=28,
    )

    # Set title with the year
    ax.set_title(f"{year}", fontsize=32)
    ax.set_xlabel("Longitude", fontsize=24)
    ax.set_ylabel("Latitude", fontsize=24)
    # Increase font size of latitude and longitude tick labels
    ax.tick_params(
        axis="both", which="major", labelsize=24
    )  # Adjust 'labelsize' as needed


# New colors and tillage order
colors = ["#233D4D", "#FE7F2D", "#A1C181"]
tillage_order = ["ConventionalTill", "MinimumTill", "NoTill-DirectSeed"]

# Ensure that tillage_order covers all unique tillage types from the data
color_dict = dict(zip(tillage_order, colors))

# New names for legend items
new_legend_names = {
    "ConventionalTill": "CT",
    "MinimumTill": "MT",
    "NoTill-DirectSeed": "NT",
}

# Create county centroids (pre-calculated from earlier)
centroids = pnw.geometry.centroid
county_centroids = {
    county: (centroid.x, centroid.y) for county, centroid in zip(pnw["NAME"], centroids)
}

# Create custom settings for each county label (position and rotation)
county_label_settings = {
    "Whitman": {"xytext": (-130, -20), "rotation": 45},
    "Columbia": {"xytext": (20, -45), "rotation": -30},
    "Spokane": {"xytext": (40, 100), "rotation": 0},
    "Walla Walla": {"xytext": (-65, 50), "rotation": 50},
    "Asotin": {"xytext": (0, -30), "rotation": 0},
    "Garfield": {"xytext": (5, -80), "rotation": 90},
}

# Create figure and axes for subplots
fig, axes = plt.subplots(1, 3, figsize=(32, 32), dpi=300)  # 1 row, 3 columns

# Plot maps for 2012, 2017, and 2022
datasets = [mapped_2012, mapped_2017, mapped_2022]
shapefiles = [shpfile_2012, shpfile_2017, shpfile_2022]
years = [2012, 2017, 2022]

for i, (mapped_df, shfile_df, year) in enumerate(zip(datasets, shapefiles, years)):
    # Merge dataset with geometry from shapefile
    mapped_df = mapped_df.merge(
        shfile_df[["pointID", "geometry"]], on="pointID", how="left"
    )
    mapped_df = gpd.GeoDataFrame(mapped_df, geometry="geometry")

    # Plot the map for each year
    plot_map(
        mapped_df,
        axes[i],
        pnw,
        color_dict,
        new_legend_names,
        year,
        county_centroids,
        county_label_settings,
    )

# # Save and display the final figure
# plt.tight_layout()
# plt.savefig(
#     path_to_plots + "mapping/fig_maps_121722.png",
#     dpi=300,
#     bbox_inches="tight",
# )

plt.show()
# -

# # Compare mapped tillage statistics with USDA Census data

# +
path_to_data = path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
    "Ph.D/Projects/Tillage_Mapping/Data/"
)

usda_stats = pd.read_csv(path_to_data + "accuracy_assessment_data/USDA_stats.csv")

# Combine datasets into one DataFrame for processing
datasets = [mapped_2012, mapped_2017, mapped_2022]
combined_df = pd.concat(datasets, ignore_index=True)

# Rename tillage types to match USDA format
combined_df["Tillage"] = combined_df["Tillage"].replace(
    {"NoTill-DirectSeed": "NT", "MinimumTill": "MT", "ConventionalTill": "CT"}
)

# Step 1: Calculate total acres for each county per year
total_acres = combined_df.groupby(["County", "year"])["ExactAcres"].sum().reset_index()
total_acres = total_acres.rename(columns={"ExactAcres": "TotalAcres"})

# Step 2: Merge the total acres back to the main dataframe
combined_df = combined_df.merge(total_acres, on=["County", "year"], how="left")

# Step 3: Calculate relative area (ExactAcres / TotalAcres)
combined_df["Relative_area"] = combined_df["ExactAcres"] / combined_df["TotalAcres"]

# Step 4: Create the new dataframe in the same structure as usda_stats
mapped_stats = (
    combined_df.groupby(["County", "Tillage", "year"])
    .agg({"Relative_area": "sum"})  # Sum of relative areas by tillage practice
    .reset_index()
)

# Step 5: Add Source column
mapped_stats["Source"] = "Mapped"  # Since this data is from your mapped dataset

# Rename columns to match usda_stats structure
mapped_stats = mapped_stats.rename(columns={"year": "Year"})

# Set the Tillage order to be NT, MT, CT using categorical type
tillage_order = ["NT", "MT", "CT"]
mapped_stats["Tillage"] = pd.Categorical(
    mapped_stats["Tillage"], categories=tillage_order, ordered=True
)


# Reorder columns
mapped_stats = mapped_stats[["County", "Source", "Relative_area", "Year", "Tillage"]]

# Sort for easier comparison
mapped_stats = mapped_stats.sort_values(by=["County", "Year", "Tillage"]).reset_index(drop=True)

# +
#######
## Plot relationship between relative areas
#######

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# Step 1: Merge USDA stats with mapped stats for comparison
comparison_df = pd.merge(
    usda_stats,
    mapped_stats,
    on=["County", "Year", "Tillage"],
    suffixes=("_usda", "_mapped"),
)

# Define custom colors for counties
custom_colors = {
    "Asotin": "#88b9d9",
    "Columbia": "#e07a5f",
    "Garfield": "#3d405b",
    "Spokane": "#5f797b",
    "Walla Walla": "#81b29a",
    "Whitman": "#f2cc8f",
}

# Define custom markers for Tillage types
custom_markers = {
    "CT": "^",
    "MT": "s",
    "NT": "X",
}

# Your existing data preparation code...
# [Assuming 'comparison_df' is already prepared]

# Create a scatter plot for each year
years = [2012, 2017, 2022]

# Increase figure size significantly
fig, axes = plt.subplots(1, 3, figsize=(32, 10), sharex=True, sharey=True)

# Set the overall font size
plt.rcParams.update({"font.size": 32})

# Loop over each year to create scatter plots
for i, year in enumerate(years):
    # Filter data for the current year
    year_df = comparison_df[comparison_df["Year"] == year]

    # Scatter plot with county as hue (color) and tillage as style (shape)
    sns.scatterplot(
        data=year_df,
        x="Relative_area_usda",
        y="Relative_area_mapped",
        hue="County",
        style="Tillage",
        ax=axes[i],
        s=500,  # Size of points
        hue_order=sorted(comparison_df["County"].unique()),
        style_order=sorted(comparison_df["Tillage"].unique()),
        markers=custom_markers,
        palette=custom_colors,
    )

    # Set titles and labels
    axes[i].set_title(f"{year}", fontsize=40)
    axes[i].set_xlabel("USDA Relative Area", fontsize=40)
    axes[i].set_ylabel("Mapped Relative Area", fontsize=40 if i == 0 else 0)

    # Add 45-degree line
    lims = [0, 1]
    axes[i].plot(lims, lims, "--", color="gray", linewidth=2)

    # Set the same ticks for x and y axes (every 0.2)
    axes[i].set_xticks(np.arange(0, 1.2, 0.2))
    axes[i].set_yticks(np.arange(0, 1.2, 0.2))

    # Calculate Pearson correlation
    corr, _ = pearsonr(year_df["Relative_area_usda"], year_df["Relative_area_mapped"])

    # Display the Pearson correlation on the plot
    axes[i].annotate(
        f"Pearson r = {corr:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=40,
        ha="left",
        va="top",
    )

# Remove individual legends from each subplot
for ax in axes:
    ax.legend_.remove()

# Create custom legend handles for Counties (colors)
county_handles = [
    Patch(color=custom_colors[county], label=county)
    for county in sorted(custom_colors.keys())
]

# Create custom legend handles for Tillage types (markers)
tillage_handles = [
    Line2D(
        [0],
        [0],
        marker=custom_markers[tillage],
        color="black",
        linestyle="",
        markersize=20,  # Adjust marker size as needed
        label=tillage,
    )
    for tillage in sorted(custom_markers.keys())
]

# Add the County legend
legend1 = fig.legend(
    handles=county_handles,
    title="County",
    loc="upper right",
    bbox_to_anchor=(0.94, 0.9),
    fontsize=32,
    title_fontsize=36,
    markerscale=2,
)

# Add the Tillage legend
legend2 = fig.legend(
    handles=tillage_handles,
    title="Tillage",
    loc="upper right",
    bbox_to_anchor=(0.91, 0.4),
    fontsize=32,
    title_fontsize=36,
    markerscale=1,
)

# Make the legend titles bold
legend1.get_title().set_fontweight("bold")
legend2.get_title().set_fontweight("bold")

# Adjust layout to give space for the legends
plt.subplots_adjust(right=0.8)

# # Save the plot as a high-resolution image
# plt.savefig(
#     path_to_plots + "mapping/fig_mapped_usda_correlation.png",
#     dpi=200,
#     bbox_inches="tight",
# )

# Show the plot
plt.show()
# -

# # Compare trends of change in tillage

# +
# Plot trends

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties

# Define bold font properties for legend titles
bold_font = FontProperties(weight="bold", size=24)

# Adjust the path
df = comparison_df

# Reshape the data
df_usda = df[["County", "Year", "Tillage", "Relative_area_usda"]].copy()
df_usda["Source"] = "USDA"
df_usda.rename(columns={"Relative_area_usda": "Relative_area"}, inplace=True)

df_mapped = df[["County", "Year", "Tillage", "Relative_area_mapped"]].copy()
df_mapped["Source"] = "Mapped"
df_mapped.rename(columns={"Relative_area_mapped": "Relative_area"}, inplace=True)

# Combine both datasets
data = pd.concat([df_usda, df_mapped])

# Adjustments for visual representation
hatches_adjusted = {"2012": " ", "2017": ".", "2022": "x"}
colors = {"CT": "#233D4D", "MT": "#FE7F2D", "NT": "#A1C181"}

# Ensure Year is string for consistent handling and sort data
data["Year"] = data["Year"].astype(str)
data["Source"] = pd.Categorical(data["Source"], ["Mapped", "USDA"])

# Create the plot
fig, ax = plt.subplots(figsize=(20, 10))
bar_width = 0.4
space_within_years = 0.05 
space_between_sources = 0.5  
space_between_counties = 2  

current_position = 0
last_county = None

x_ticks = []
x_tick_labels = []

for county, county_data in data.groupby("County"):
    if last_county and county != last_county:
        current_position += space_between_counties  
    last_county = county

    for source in ["Mapped", "USDA"]:
        for year in sorted(data["Year"].unique()):
            data_subset = county_data[
                (county_data["Source"] == source) & (county_data["Year"] == year)
            ]
            bottom_height = 0
            for tillage, color in colors.items():
                tillage_area = data_subset[data_subset["Tillage"] == tillage][
                    "Relative_area"
                ].sum()
                if tillage_area > 0:
                    ax.bar(
                        current_position,
                        tillage_area,
                        bottom=bottom_height,
                        color=color,
                        hatch=hatches_adjusted[year],
                        width=bar_width,
                        edgecolor="black",
                    )
                    bottom_height += tillage_area

            current_position += bar_width + space_within_years

        # Add the x-tick in the center of the group for this source
        x_ticks.append(current_position - ((bar_width + space_within_years) * 1.5))
        x_tick_labels.append(f"{county} ({source})")  # Removed year from label

        current_position += space_between_sources - space_within_years

# Adjust x_ticks and labels to only include county and source, removing year
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels, rotation=90, ha="center")

ax.tick_params(axis="x", labelsize="large")  # Increase x-axis tick label font size
ax.tick_params(axis="y", labelsize="large")  # Increase y-axis tick label font size
ax.grid(False)

# Change x-axis tick size
ax.tick_params(axis="x", labelsize=24)  # Adjust the number for desired size

# Change y-axis tick size
ax.tick_params(axis="y", labelsize=25)  # Adjust the number for desired size

# Create and add legends for Tillage Type and Year
tillage_legend_elements = [
    Patch(facecolor=colors[tillage], edgecolor="black", label=tillage)
    for tillage in colors
]
tillage_legend = ax.legend(
    handles=tillage_legend_elements,
    title="Tillage",
    loc="upper left",
    bbox_to_anchor=(1, 1),  
    fontsize=24,
    title_fontproperties=bold_font,  
)

year_legend_elements = [
    Patch(
        facecolor="white", edgecolor="black", label=year, hatch=hatches_adjusted[year]
    )
    for year in sorted(data["Year"].unique())
]
year_legend = ax.legend(
    handles=year_legend_elements,
    title="Year",
    loc="upper left",
    bbox_to_anchor=(1, 0.55),  
    fontsize=24,
    title_fontproperties=bold_font,  
)

# Add the tillage legend back to the plot so both appear
ax.add_artist(tillage_legend)

plt.tight_layout()
plt.subplots_adjust(right=0.85)


# # Save the plot as a high-resolution image
# plt.savefig(
#     path_to_plots + "mapping/fig_map_trends.png",
#     dpi=200,
#     bbox_inches="tight",
# )

plt.show()
