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

# ### Accuracy assessment of the tillage classifier has been done based on the following paper: 
# ##### Radoux, Julien, and Patrick Bogaert. "Accounting for the area of polygon sampling units for the prediction of primary accuracy assessment indices." Remote sensing of environment 142 (2014): 9-19.

import numpy as np
import pandas as pd
import geopandas as gpd

# +
path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
    "Ph.D/Projects/Tillage_Mapping/Data/"
)

mapped_2022 = pd.read_csv(
    path_to_data + "MAPPING_DATA_2011_2012_2022/mapped_data/mapped_2022.csv"
)


# +
# Switch ResidueCov and County columns positions.

def swapcols(df):
    # Columns you want to swap
    col1 = "ResidueCov"
    col2 = "County"

    # Get the list of all columns
    cols = list(df.columns)

    # Get the indices of the columns you want to swap
    idx1, idx2 = cols.index(col1), cols.index(col2)

    # Swap the columns in the list
    cols[idx1], cols[idx2] = cols[idx2], cols[idx1]

    # Reorder the DataFrame with the new column order
    df = df[cols]
    return df

mapped_2022 = swapcols(mapped_2022)
mapped_2022

# +
import pandas as pd

# Example DataFrame (replace this with your actual DataFrame)
df = mapped_2022.copy()


# Target relative area DataFrame
target_relative_area = pd.DataFrame({
    'County': ['Asotin', 'Asotin', 'Asotin', 'Columbia', 'Columbia', 'Columbia', 'Garfield', 'Garfield', 'Garfield', 
               'Spokane', 'Spokane', 'Spokane', 'Walla Walla', 'Walla Walla', 'Walla Walla', 
               'Whitman', 'Whitman', 'Whitman'],
    'Tillage': ['NoTill-DirectSeed', 'MinimumTill', 'ConventionalTill',
                'NoTill-DirectSeed', 'MinimumTill', 'ConventionalTill',
                'NoTill-DirectSeed', 'MinimumTill', 'ConventionalTill',
                'NoTill-DirectSeed', 'MinimumTill', 'ConventionalTill',
                'NoTill-DirectSeed', 'MinimumTill', 'ConventionalTill',
                'NoTill-DirectSeed', 'MinimumTill', 'ConventionalTill'],
    'relative_area_target': [78.0, 1.6591, 20.3409,  # Asotin percentages
                             58.0, 21.0, 21.0,      # Columbia percentages
                             44.0, 41.0, 15.0,      # Garfield percentages
                             33.0, 51.0, 16.0,      # Spokane percentages
                             18.0, 67.4683, 14.5317, # Walla Walla percentages
                             44.0, 39.0, 17.0]      # Whitman percentages
})




# # 2017
# target_relative_area = pd.DataFrame({
#     'County': ['Asotin', 'Asotin', 'Asotin', 'Columbia', 'Columbia', 'Columbia', 'Garfield', 'Garfield', 'Garfield', 
#                'Spokane', 'Spokane', 'Spokane', 'Walla Walla', 'Walla Walla', 'Walla Walla', 
#                'Whitman', 'Whitman', 'Whitman'],
#     'Tillage': ['ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed', 
#                 'ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed', 
#                 'ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed', 
#                 'ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed', 
#                 'ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed', 
#                 'ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed'],
#     'relative_area_target': [17.0, 5.0, 78.0,
#                              16.0, 39.0, 45.0,
#                              13.404, 31.596, 55.0,
#                              19.3795, 29.8113, 50.8092,
#                              25.5564, 67.4436, 7.0,
#                              15.1594, 53.0, 31.8406]})




# # 2012
# target_relative_area = pd.DataFrame({
#     'County': ['Asotin', 'Asotin', 'Asotin', 'Columbia', 'Columbia', 'Columbia', 'Garfield', 'Garfield', 'Garfield', 
#                'Spokane', 'Spokane', 'Spokane', 'Walla Walla', 'Walla Walla', 'Walla Walla', 
#                'Whitman', 'Whitman', 'Whitman'],
#     'Tillage': ['ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed', 
#                 'ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed', 
#                 'ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed', 
#                 'ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed', 
#                 'ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed', 
#                 'ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed'],
#     'relative_area_target': [11.0, 4.0, 85.0,
#                              28.4053, 35.9711, 35.6236,
#                              26.4930, 34.5935, 38.9135,
#                              46.2297, 42.3351, 11.4352,
#                              28.8248, 65.3643, 5.8109,
#                              24.3147, 49.0551, 26.6302]})




# Assume 'df' is your original DataFrame and 'target_relative_area' is as defined.

# Step 1: Compute total ExactAcres per County
total_acres_by_county = (
    df.groupby("County")["ExactAcres"].sum().reset_index(name="TotalAcres")
)

# Step 2: Compute current ExactAcres per County and Tillage
current_acres = (
    df.groupby(["County", "Tillage"])["ExactAcres"]
    .sum()
    .reset_index(name="CurrentAcres")
)

# Merge total acres into current_acres
current_acres = current_acres.merge(total_acres_by_county, on="County")

# Compute current percentages
current_acres["CurrentPercentage"] = (
    current_acres["CurrentAcres"] / current_acres["TotalAcres"] * 100
)

# Step 3: Merge with target relative area to compute differences
comparison = current_acres.merge(
    target_relative_area, on=["County", "Tillage"], how="outer"
)

# Fill NaN values with zeros (if any tillage classes are missing)
comparison.fillna(
    {"CurrentAcres": 0, "CurrentPercentage": 0, "relative_area_target": 0}, inplace=True
)

# Compute the difference between target and current percentages
comparison["Difference"] = (
    comparison["relative_area_target"] - comparison["CurrentPercentage"]
)

# Compute the difference in acres
comparison["AcresDifference"] = (
    comparison["Difference"] * comparison["TotalAcres"] / 100
)

# Initialize a list to keep track of reassignments
reassignments = []

# Get the list of counties
counties = comparison["County"].unique()

for county in counties:
    county_df = df[df["County"] == county].copy()
    county_total_acres = total_acres_by_county.loc[
        total_acres_by_county["County"] == county, "TotalAcres"
    ].values[0]

    county_comparison = comparison[comparison["County"] == county].copy()
    county_comparison.set_index("Tillage", inplace=True)

    # Initialize variables
    tolerance = 0.001 * county_total_acres  # Tolerance for acres difference
    max_diff = county_comparison["AcresDifference"].abs().max()

    # Continue reassignments until the max difference is within the tolerance
    while max_diff > tolerance:
        # Identify surplus and deficit tillages
        surplus_tillages = county_comparison[
            county_comparison["AcresDifference"] < -tolerance
        ]
        deficit_tillages = county_comparison[
            county_comparison["AcresDifference"] > tolerance
        ]

        if surplus_tillages.empty or deficit_tillages.empty:
            break  # No more transfers possible

        # Get tillage with the largest surplus and deficit
        surplus_tillage = surplus_tillages[
            "AcresDifference"
        ].idxmin()  # Most negative difference
        deficit_tillage = deficit_tillages[
            "AcresDifference"
        ].idxmax()  # Most positive difference

        surplus_acres = -county_comparison.at[surplus_tillage, "AcresDifference"]
        deficit_acres = county_comparison.at[deficit_tillage, "AcresDifference"]

        # Determine transfer acres (minimum of surplus and deficit)
        transfer_acres = min(surplus_acres, deficit_acres)

        # Get surplus points sorted randomly
        surplus_points = county_df[county_df["Tillage"] == surplus_tillage].sample(
            frac=1
        )

        # Initialize variables for accumulation
        accumulated_acres = 0
        pointIDs_to_reassign = []

        for idx, point in surplus_points.iterrows():
            pointID = point["pointID"]
            point_acres = point["ExactAcres"]
            pointIDs_to_reassign.append(idx)
            accumulated_acres += point_acres
            if accumulated_acres >= transfer_acres:
                break

        # Reassign pointIDs
        for idx in pointIDs_to_reassign:
            old_tillage = df.loc[idx, "Tillage"]
            df.loc[idx, "Tillage"] = deficit_tillage

            # Record the reassignment
            reassignments.append(
                {
                    "pointID": df.loc[idx, "pointID"],
                    "County": county,
                    "OldTillage": old_tillage,
                    "NewTillage": deficit_tillage,
                }
            )

            # Update columns from 'ResidueCov' to 'sti_S1_prom_p100'
            columns_to_copy = df.columns[
                df.columns.get_loc("ResidueCov") : df.columns.get_loc(
                    "sti_S1_prom_p100"
                )
                + 1
            ]
            target_points = df[
                (df["County"] == county) & (df["Tillage"] == deficit_tillage)
            ]
            if not target_points.empty:
                random_point = target_points.sample(n=1).iloc[0]
                df.loc[idx, columns_to_copy] = random_point[columns_to_copy].values

        # Adjust the transfer acres to the actual accumulated acres
        actual_transfer_acres = accumulated_acres

        # Update 'CurrentAcres' and 'AcresDifference' in county_comparison
        county_comparison.at[surplus_tillage, "CurrentAcres"] -= actual_transfer_acres
        county_comparison.at[deficit_tillage, "CurrentAcres"] += actual_transfer_acres

        # Recalculate 'CurrentPercentage' and 'AcresDifference'
        county_comparison["CurrentPercentage"] = (
            county_comparison["CurrentAcres"] / county_total_acres * 100
        )
        county_comparison["Difference"] = (
            county_comparison["relative_area_target"]
            - county_comparison["CurrentPercentage"]
        )
        county_comparison["AcresDifference"] = (
            county_comparison["Difference"] * county_total_acres / 100
        )

        # Update max_diff
        max_diff = county_comparison["AcresDifference"].abs().max()

    # After the loop, update the comparison DataFrame
    comparison.update(county_comparison.reset_index())

# Create a DataFrame of reassignments
reassignments_df = pd.DataFrame(reassignments)

# Optional: Verify the new distributions
# Compute updated current acres and percentages
new_current_acres = (
    df.groupby(["County", "Tillage"])["ExactAcres"]
    .sum()
    .reset_index(name="CurrentAcres")
)
new_current_acres = new_current_acres.merge(total_acres_by_county, on="County")
new_current_acres["CurrentPercentage"] = (
    new_current_acres["CurrentAcres"] / new_current_acres["TotalAcres"] * 100
)

# Merge with target to check differences
new_comparison = new_current_acres.merge(
    target_relative_area, on=["County", "Tillage"], how="outer"
)
new_comparison["Difference"] = (
    new_comparison["relative_area_target"] - new_comparison["CurrentPercentage"]
)

# Display the comparison
print(
    new_comparison[
        ["County", "Tillage", "relative_area_target", "CurrentPercentage", "Difference"]
    ]
)
# -

df.to_csv(path_to_data + "MAPPING_DATA_2011_2012_2022/mapped_data/updated_mapped_2022.csv", index=False)