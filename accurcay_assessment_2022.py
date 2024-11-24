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
import geopandas as gpd

# +
path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
    "Ph.D/Projects/Tillage_Mapping/Data/"
)

mapped_2022 = pd.read_csv(
    path_to_data + "MAPPING_DATA_2011_2012_2022/mapped_data/updated_mapped_2022.csv"
)


shpfile_filtered_2022 = gpd.read_file(
    path_to_data + "GIS_Data/acc_assessment/WSDA_2022_groundtruth_filtered.shp"
)

test_df_1 = pd.read_csv(path_to_data + "accuracy_assessment_data/X_test_y_test_pred_1.csv")
test_df_2 = pd.read_csv(path_to_data + "accuracy_assessment_data/X_test_y_test_pred_2.csv")
test_df_3 = pd.read_csv(path_to_data + "accuracy_assessment_data/X_test_y_test_pred_3.csv")

shpfile_2122 = gpd.read_file(path_to_data + "GIS_Data/final_shpfiles/final_shp_2122.shp")
shpfile_2223 = gpd.read_file(path_to_data + "GIS_Data/final_shpfiles/final_shp_2223.shp")
ground_merged = pd.concat([shpfile_2122, shpfile_2223])


mapped_2022 = mapped_2022.loc[
    mapped_2022["pointID"].isin(shpfile_filtered_2022["pointID"])
]

test_df_1 = pd.merge(
    test_df_1, ground_merged[["pointID", "ExactAcres"]], on="pointID", how="left"
)
test_df_2 = pd.merge(
    test_df_2, ground_merged[["pointID", "ExactAcres"]], on="pointID", how="left"
)
test_df_3 = pd.merge(
    test_df_3, ground_merged[["pointID", "ExactAcres"]], on="pointID", how="left"
)

# +
import pandas as pd
import numpy as np

# Load the sampled and non-sampled dataframes
sampled_df = test_df_2

non_sampled_df = mapped_2022

# Define classes
classes = ["MinimumTill", "NoTill-DirectSeed", "ConventionalTill"]

# Initialize accumulators for overall accuracy calculation
correct_area = 0  # Sum of S_i * correctly classified polygons
total_area = 0  # Sum of all areas

# Step 1: Calculate the first part of the overall accuracy (sampled data)
for i, row in sampled_df.iterrows():
    if row["y_test"] == row["y_pred"]:  # If correctly classified
        correct_area += row["ExactAcres"] if not np.isnan(row["ExactAcres"]) else 0
    total_area += row["ExactAcres"] if not np.isnan(row["ExactAcres"]) else 0

# Step 2: Calculate the second part (non-sampled data)
p_hat = {}  # Store p_j estimates for each class
for cls in classes:
    # Estimate p_j
    beta_sum = (
        sampled_df["y_pred"] == cls
    ).sum()  # Count how many polygons were predicted as cls
    alpha_beta_sum = (
        (sampled_df["y_test"] == cls) & (sampled_df["y_pred"] == cls)
    ).sum()  # Correctly classified
    if beta_sum > 0:
        p_hat[cls] = alpha_beta_sum / beta_sum
    else:
        p_hat[cls] = 0

# Sum over non-sampled polygons
for i, row in non_sampled_df.iterrows():
    predicted_class = row["Tillage"]
    correct_area += p_hat[predicted_class] * row["ExactAcres"]  # Use p_hat estimate
    total_area += row["ExactAcres"]

# Overall accuracy
overall_accuracy = correct_area / total_area


# Display the results
overall_accuracy

# +
# Initialize a dictionary to store user's accuracy for each class
users_accuracy = {}

# Iterate over each class to compute User's Accuracy for each class
for cls in classes:
    # Step 1: Sampled data (i = 1 to n)
    alpha_beta_sum_sampled = 0
    beta_sum_sampled = 0
    for i, row in sampled_df.iterrows():
        # If the polygon belongs to class `cls`
        if row["y_test"] == cls and row["y_pred"] == cls:
            alpha_beta_sum_sampled += (
                row["ExactAcres"] if not np.isnan(row["ExactAcres"]) else 0
            )
        if row["y_pred"] == cls:
            beta_sum_sampled += (
                row["ExactAcres"] if not np.isnan(row["ExactAcres"]) else 0
            )

    # Step 2: Non-sampled data (i = n+1 to N)
    alpha_beta_sum_non_sampled = 0
    beta_sum_non_sampled = 0
    for i, row in non_sampled_df.iterrows():
        if not np.isnan(row["ExactAcres"]):
            # If the polygon is predicted to be in class `cls`
            if row["Tillage"] == cls:
                alpha_beta_sum_non_sampled += p_hat[cls] * row["ExactAcres"]
                beta_sum_non_sampled += row["ExactAcres"]

    # Numerator: Sum of sampled + non-sampled correctly classified areas
    numerator = alpha_beta_sum_sampled #+ alpha_beta_sum_non_sampled

    # Denominator: Sum of all areas predicted as `cls`
    denominator = beta_sum_sampled #+ beta_sum_non_sampled

    # User's accuracy for class `cls`
    users_accuracy[cls] = numerator / denominator if denominator > 0 else 0

# Display the User's Accuracy for each class
users_accuracy

# +
# Initialize a dictionary to store producer's accuracy for each class
producers_accuracy = {}

# Iterate over each class to compute Producer's Accuracy
for cls in classes:
    # # Step 1: Numerator is the same as user's accuracy (correctly classified areas)
    alpha_beta_sum_sampled = 0
    for i, row in sampled_df.iterrows():
        # If the polygon belongs to class `cls`
        if row["y_test"] == cls and row["y_pred"] == cls:
            alpha_beta_sum_sampled += (
                row["ExactAcres"] if not np.isnan(row["ExactAcres"]) else 0
            )
    numerator = alpha_beta_sum_sampled
    # Step 2: Denominator
    # Part 1: Sampled data (i = 1 to n)
    alpha_sum_sampled = 0
    for i, row in sampled_df.iterrows():
        # If the actual class is `cls`
        if row["y_test"] == cls:
            alpha_sum_sampled += (
                row["ExactAcres"] if not np.isnan(row["ExactAcres"]) else 0
            )

    # Denominator: Total area that actually belongs to class `cls`
    denominator = alpha_sum_sampled #+ alpha_sum_non_sampled

    # Producer's accuracy for class `cls`
    producers_accuracy[cls] = numerator / denominator if denominator > 0 else 0

# Display the Producer's Accuracy for each class
producers_accuracy

# +
# Initialize total area for both sampled and non-sampled polygons
total_area = sampled_df["ExactAcres"].sum() + non_sampled_df["ExactAcres"].sum()

# Calculate the numerator for variance (squared difference term)
variance_numerator = 0

# Iterate over non-sampled data
for i, row in non_sampled_df.iterrows():
    predicted_class = row["Tillage"]

    # Estimated accuracy for the class
    p_hat_j = p_hat[predicted_class]

    # Actual correct classification for the non-sampled data
    alpha_beta_sum_non_sampled = 1 if row["Tillage"] == predicted_class else 0

    # Compute the difference and square it
    diff = (p_hat_j * row["ExactAcres"]) - (
        alpha_beta_sum_non_sampled * row["ExactAcres"]
    )
    variance_numerator += diff**2

# Variance of overall accuracy
variance_overall_accuracy = variance_numerator / total_area

# Display the variance of overall accuracy
variance_overall_accuracy

# +
import numpy as np

# Step 1: Calculate the standard error (SE)

overall_accuracy = round(overall_accuracy, 2) 

standard_error = np.sqrt(variance_overall_accuracy) * 0.01

# Step 2: Determine the critical value (for 95% confidence interval)
z_alpha_over_2 = 1.96  # For 95% confidence

# Step 3: Compute the confidence interval
lower_bound = overall_accuracy - z_alpha_over_2 * standard_error
upper_bound = overall_accuracy + z_alpha_over_2 * standard_error

print("standard_error", standard_error)
# Display the confidence interval
confidence_interval = (lower_bound, upper_bound)
print("Confidence interval:", confidence_interval)
# -


