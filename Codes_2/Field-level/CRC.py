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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import NASA_core as nc

# +
path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
    "Ph.D/Projects/Tillage_Mapping/Data/field_level_data/CRC/"
)

df_2122_l7 = pd.read_csv(
    path_to_data + "L7_T1C2L2_timeSeries_2021-08-01_2022-07-31.csv"
)

df_2223_l7 = pd.read_csv(
    path_to_data + "L7_T1C2L2_timeSeries_2022-08-01_2023-07-31.csv"
)

df_2122_l8 = pd.read_csv(
    path_to_data + "L8_T1C2L2_timeSeries_2021-08-01_2022-07-31.csv"
)

df_2223_l8 = pd.read_csv(
    path_to_data + "L8_T1C2L2_timeSeries_2022-08-01_2023-07-31.csv"
)

df_2122 = pd.concat([df_2122_l7, df_2122_l8])
df_2223 = pd.concat([df_2223_l7, df_2223_l8])

df_2122 = df_2122.dropna(subset=["NDVI", "NDTI", "ResidueCov"])
df_2223 = df_2223.dropna(subset=["NDVI", "NDTI", "ResidueCov"])
# -

df_2122.head(3)


# +
def num_to_date(df):
    # Convert to string and format
    df["formatted_system_start_time"] = (
        df["formatted_system_start_time"]
        .astype(str)
        .str.replace(".", "")
        .str.slice(0, 8)
    )
    df["formatted_system_start_time"] = df["formatted_system_start_time"].apply(
        lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}"
    )

    # Optionally convert to datetime
    df["formatted_system_start_time"] = pd.to_datetime(
        df["formatted_system_start_time"], format="%Y-%m-%d"
    )

    return df


df_2122 = num_to_date(df_2122)
df_2223 = num_to_date(df_2223)
df_2123 = pd.concat([df_2122, df_2223])


# +
#  Pick a field
VI_idx = "NDTI"
a_field = df_2122[df_2122.pointID == df_2122.pointID.unique()[0]].copy()
a_field.sort_values(
    by="formatted_system_start_time", axis=0, ascending=True, inplace=True
)

# Plot
fig, ax = plt.subplots(
    1,
    1,
    figsize=(12, 3),
    sharex="col",
    sharey="row",
    # sharex=True, sharey=True,
    gridspec_kw={"hspace": 0.2, "wspace": 0.05},
)
ax.grid(axis="y", which="both")

ax.scatter(a_field["formatted_system_start_time"], a_field[VI_idx], s=40, c="#d62728")
ax.plot(
    a_field["formatted_system_start_time"],
    a_field[VI_idx],
    linestyle="-",
    linewidth=3.5,
    color="#d62728",
    alpha=0.8,
    label=f"raw {VI_idx}",
)
plt.ylim([-0.5, 1.2])
ax.legend(loc="lower right")

# +
df = df_2123
res_1_idx = np.random.choice(
    df.loc[df["ResidueCov"] == "0-15%"]["pointID"].unique(), size=1, replace=False
)
# res_2_idx = np.random.choice(
#     df.loc[df["ResidueCov"] == "16-30%"].index, size=1, replace=False
# )
# res_3_idx = np.random.choice(
#     df.loc[df["ResidueCov"] == ">30%"].index, size=1, replace=False
# )
res_1_idx

# [res_1_idx, res_2_idx, res_3_idx]

# +
import pandas as pd
import matplotlib.pyplot as plt

# Load your DataFrame here
df = df_2223

res_1_idx = np.random.choice(
    df.loc[df["ResidueCov"] == "0-15%"]["pointID"].unique(), size=1, replace=False
)
res_2_idx = np.random.choice(
    df.loc[df["ResidueCov"] == "16-30%"]["pointID"].unique(), size=1, replace=False
)
res_3_idx = np.random.choice(
    df.loc[df["ResidueCov"] == ">30%"]["pointID"].unique(), size=1, replace=False
)

df = df.loc[df['pointID'].isin([res_1_idx[0], res_2_idx[0], res_3_idx[0]])]

df["DateTime"] = df["formatted_system_start_time"]
# Convert 'DateTime' to datetime objects
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Create a color map for ResidueCov types
ResidueCov_types = df["ResidueCov"].unique()
colors = plt.cm.get_cmap("tab10", len(ResidueCov_types))
color_map = {ResidueCov: colors(i) for i, ResidueCov in enumerate(ResidueCov_types)}

# Plotting
fig, ax = plt.subplots(figsize=(30, 5))

# Iterate through each pointID
for point_id in df["pointID"].unique():
    df_point = df[df["pointID"] == point_id].sort_values(by="DateTime")
    ResidueCov_type = df_point["ResidueCov"].iloc[0]  # Get ResidueCov type
    ax.plot(
        df_point["DateTime"],
        df_point["NDTI"],
        color=color_map[ResidueCov_type],
        label=ResidueCov_type,
    )

# Remove duplicate labels in the legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(
    by_label.values(),
    by_label.keys(),
    loc="best",
    bbox_to_anchor=(1.05, 1.0),
    title="ResidueCov Type",
)

ax.set_xlabel("Date")
ax.set_ylabel("NDTI")
ax.set_title("NDTI vs Date for Different PointIDs, Colored by ResidueCov Type")

# To prevent labels from overlapping
plt.tight_layout()

# Show the plot
plt.show()
