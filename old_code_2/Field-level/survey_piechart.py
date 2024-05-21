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
#     display_name: tillmap
#     language: python
#     name: python3
# ---

import pandas as pd 
import geopandas as gpd

path_to_survery = "/Users/aminnorouzi/Library/CloudStorage/OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/Tillage_Mapping/Data/GIS_Data/shapefiles_2021_2022/point_polygon_joined_2122/point_polygon_joined_2122.shp"
df = gpd.read_file(path_to_survery)
df
# gpd = gpd.read_file(path_)
# gpd
df.groupby(["County"])["Tillage"]
df['Tillage'].value_counts()
Whitman  = df.loc[df['County'] == 'Columbia']
Whitman['Tillage'].value_counts()

# +
data = {
    "County": ["Whitman", "Columbia"],
    "NT": [0.51, 0.09],
    "MT": [0.37, 0.48],
    "CT": [0.51, 0.43],
}

# Creating DataFrame
df = pd.DataFrame(data)
df

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
    "County": ["Whitman", "Columbia"],
    "Source": ["Ground-truth", "Ground-truth"],
    "Year": [2223, 2223],
    "NT": [0.51, 0.09],
    "MT": [0.37, 0.48],
    "CT": [0.51, 0.43],
}

# Creating DataFrame
df = pd.DataFrame(data)
df

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
g.fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.60)
plt.show()
