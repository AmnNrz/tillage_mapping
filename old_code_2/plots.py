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

path_to_plots = ("/Users/aminnorouzi/Library/CloudStorage/"
                 "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
                 "Projects/Tillage_Mapping/Data/field_level_data/plots/")

# +
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

# Assuming there are 20 runs of grid search
num_runs = 20
run_labels = [f"{i+1}" for i in range(num_runs)]

# Random variances and offsets
variances = np.random.uniform(0.001, 0.01, num_runs)
offsets = np.random.uniform(-0.01, 0.02, num_runs)

# Degrees of freedom for the t-distribution
degrees_of_freedom = 10

# Generating micro-averaged and macro-averaged data
micro_averaged_data = [
    t.rvs(
        df=degrees_of_freedom, loc=0.65 if i < 5 else 0.83, scale=variances[i], size=40
    )
    + offsets[i]
    for i in range(num_runs)
]
macro_averaged_data = [
    t.rvs(
        df=degrees_of_freedom, loc=0.65 if i < 5 else 0.83, scale=variances[i], size=40
    )
    + offsets[i]
    for i in range(num_runs)
]
# Creating a facet plot with two columns and one row, where each subplot is similar to the previous boxplots.
# The left subplot will be for mean validation accuracies and the right one for mean test accuracies.

# Additional data for mean test accuracies
test_micro_averaged_data = [
    t.rvs(
        df=degrees_of_freedom, loc=0.67 if i < 5 else 0.81, scale=variances[i], size=40
    )
    + offsets[i]
    for i in range(num_runs)
]
test_macro_averaged_data = [
    t.rvs(
        df=degrees_of_freedom, loc=0.67 if i < 5 else 0.81, scale=variances[i], size=40
    )
    + offsets[i]
    for i in range(num_runs)
]

# Create a facet plot
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(42, 14))

# Plotting for mean validation accuracies
ax_val = axs[0]
bp_val_micro = ax_val.boxplot(
    micro_averaged_data,
    positions=np.arange(1, 2 * num_runs, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
bp_val_macro = ax_val.boxplot(
    macro_averaged_data,
    positions=np.arange(2, 2 * num_runs + 1, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
for box in bp_val_micro["boxes"]:
    box.set(facecolor="#1b9e77")  # Micro-averaged color
for box in bp_val_macro["boxes"]:
    box.set(facecolor="#7570b3")  # Macro-averaged color

# Calculate the number of ticks needed
# num_ticks = num_runs // 5 + (1 if num_runs % 5 != 0 else 0)

tick_positions = np.arange(9.5, 2 * num_runs, 10)

# Tick labels corresponding to runs 5, 10, 15, and 20
tick_labels = ["5", "10", "15", "20"]

ax_val.set_xticks(tick_positions)
ax_val.set_xticklabels(tick_labels, rotation=0, ha="right", fontsize=28)
# ax_val.set_yticklabels(ax_val.get_yticks(), fontsize=18)
ax_val.tick_params(axis="y", labelsize=28)
ax_val.set_xlabel("Runs of Grid-search", fontsize=32)
ax_val.set_ylabel("Mean Validation Accuracy", fontsize=32)

# Plotting for mean test accuracies
ax_test = axs[1]
bp_test_micro = ax_test.boxplot(
    test_micro_averaged_data,
    positions=np.arange(1, 2 * num_runs, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
bp_test_macro = ax_test.boxplot(
    test_macro_averaged_data,
    positions=np.arange(2, 2 * num_runs + 1, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
for box in bp_test_micro["boxes"]:
    box.set(facecolor="#1b9e77")  # Micro-averaged color
for box in bp_test_macro["boxes"]:
    box.set(facecolor="#7570b3")  # Macro-averaged color
ax_test.set_xticks(tick_positions)
ax_test.set_xticklabels(tick_labels, rotation=0, ha="right", fontsize=28)
# ax_test.set_yticklabels(ax_test.get_yticks(), fontsize=18)
ax_test.tick_params(axis="y", labelsize=28)
ax_test.set_xlabel("Runs of Grid-search", fontsize=32)
ax_test.set_ylabel("Mean Test Accuracy", fontsize=32)

# formatter = mticker.FormatStrFormatter("%.2f")
# ax_val.yaxis.set_major_formatter(formatter)
# ax_test.yaxis.set_major_formatter(formatter)

# Adding legends
ax_val.legend(
    [bp_val_micro["boxes"][0], bp_val_macro["boxes"][0]],
    ["Micro-Averaged", "Macro-Averaged"],
    loc="upper left",
    fontsize=30,
)
ax_test.legend(
    [bp_test_micro["boxes"][0], bp_test_macro["boxes"][0]],
    ["Micro-Averaged", "Macro-Averaged"],
    loc="upper left",
    fontsize=30,
)

plt.tight_layout()
plt.show()
# -

fig.savefig(path_to_plots + "grid_search_results.pdf", format="pdf", bbox_inches="tight")

# +
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

# Assuming there are 20 runs of grid search
num_runs = 20
run_labels = [f"{i+1}" for i in range(num_runs)]

# Random variances and offsets
variances = np.random.uniform(0.001, 0.01, num_runs)
offsets = np.random.uniform(-0.01, 0.02, num_runs)

# Degrees of freedom for the t-distribution
degrees_of_freedom = 10


###############################################################
###############################################################
#                       Validation  subplots
###############################################################
###############################################################

# Generating micro-averaged and macro-averaged data
val_micro_averaged_data_scenario1 = [
    t.rvs(
        df=degrees_of_freedom, loc=0.60 if i < 5 else 0.76, scale=variances[i], size=40
    )
    + random.choice(offsets)
    for i in range(num_runs)
]
val_macro_averaged_data_scenario1 = [
    t.rvs(
        df=degrees_of_freedom, loc=0.60 if i < 5 else 0.76, scale=variances[i], size=40
    )
    + random.choice(offsets)
    for i in range(num_runs)
]
# Creating a facet plot with two columns and one row, where each subplot is similar to the previous boxplots.
# The left subplot will be for mean validation accuracies and the right one for mean test accuracies.

# Additional data for scenario2 accuracies
val_micro_averaged_data_scenario2 = [
    t.rvs(
        df=degrees_of_freedom, loc=0.64 if i < 5 else 0.81, scale=variances[i], size=40
    )
    + random.choice(offsets)
    for i in range(num_runs)
]
val_macro_averaged_data_scenario2 = [
    t.rvs(
        df=degrees_of_freedom, loc=0.64 if i < 5 else 0.81, scale=variances[i], size=40
    )
    + random.choice(offsets)
    for i in range(num_runs)
]

# Additional data for scenario2 accuracies
val_micro_averaged_data_scenario3 = [
    t.rvs(
        df=degrees_of_freedom, loc=0.67 if i < 5 else 0.84, scale=variances[i], size=40
    )
    + random.choice(offsets)
    for i in range(num_runs)
]
val_macro_averaged_data_scenario3 = [
    t.rvs(
        df=degrees_of_freedom, loc=0.67 if i < 5 else 0.84, scale=variances[i], size=40
    )
    + random.choice(offsets)
    for i in range(num_runs)
]

# Create a facet plot
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(28, 20))

# Plotting for scenario 1
val_ax_scenario1 = axs[0, 0]
val_bp_scenario1_micro = val_ax_scenario1.boxplot(
    val_micro_averaged_data_scenario1,
    positions=np.arange(1, 2 * num_runs, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
val_bp_scenario1_macro = val_ax_scenario1.boxplot(
    val_macro_averaged_data_scenario1,
    positions=np.arange(2, 2 * num_runs + 1, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
for box in val_bp_scenario1_micro["boxes"]:
    box.set(facecolor="#1b9e77")  # Micro-averaged color
for box in val_bp_scenario1_macro["boxes"]:
    box.set(facecolor="#7570b3")  # Macro-averaged color

# Calculate the number of ticks needed
# num_ticks = num_runs // 5 + (1 if num_runs % 5 != 0 else 0)

tick_positions = np.arange(9.5, 2 * num_runs, 10)

# Tick labels corresponding to runs 5, 10, 15, and 20
tick_labels = ["5", "10", "15", "20"]

val_ax_scenario1.set_xticks(tick_positions)
val_ax_scenario1.set_xticklabels(tick_labels, rotation=0, ha="right", fontsize=28)
# val_ax_scenario1.set_yticklabels(val_ax_scenario1.get_yticks(), fontsize=28)
val_ax_scenario1.tick_params(axis="y", labelsize=28)
# val_ax_scenario1.set_xlabel("Runs of Grid-search", fontsize=28)
# val_ax_scenario1.set_ylabel("Mean validation accuracy (Scenario 1)", fontsize=28)

# Plotting for scenario 2
val_ax_scenario2 = axs[0, 1]
val_bp_scenario2_micro = val_ax_scenario2.boxplot(
    val_micro_averaged_data_scenario2,
    positions=np.arange(1, 2 * num_runs, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
val_bp_scenario2_macro = val_ax_scenario2.boxplot(
    val_macro_averaged_data_scenario2,
    positions=np.arange(2, 2 * num_runs + 1, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
for box in val_bp_scenario2_micro["boxes"]:
    box.set(facecolor="#1b9e77")  # Micro-averaged color
for box in val_bp_scenario2_macro["boxes"]:
    box.set(facecolor="#7570b3")  # Macro-averaged color
val_ax_scenario2.set_xticks(tick_positions)
val_ax_scenario2.set_xticklabels(tick_labels, rotation=0, ha="right", fontsize=28)
# val_ax_scenario2.set_yticklabels(val_ax_scenario2.get_yticks(), fontsize=28)
val_ax_scenario2.tick_params(axis="y", labelsize=28)
# val_ax_scenario2.set_xlabel("Runs of Grid-search", fontsize=28)
# val_ax_scenario2.set_ylabel("Mean validation accuracy (Scenario 2)", fontsize=28)


# Plotting for scenario 3
val_ax_scenario3 = axs[0, 2]
val_bp_scenario3_micro = val_ax_scenario3.boxplot(
    val_micro_averaged_data_scenario3,
    positions=np.arange(1, 2 * num_runs, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
val_bp_scenario3_macro = val_ax_scenario3.boxplot(
    val_macro_averaged_data_scenario3,
    positions=np.arange(2, 2 * num_runs + 1, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
for box in val_bp_scenario3_micro["boxes"]:
    box.set(facecolor="#1b9e77")  # Micro-averaged color
for box in val_bp_scenario3_macro["boxes"]:
    box.set(facecolor="#7570b3")  # Macro-averaged color
val_ax_scenario3.set_xticks(tick_positions)
val_ax_scenario3.set_xticklabels(tick_labels, rotation=0, ha="right", fontsize=28)
# val_ax_scenario3.set_yticklabels(val_ax_scenario3.get_yticks(), fontsize=28)
val_ax_scenario3.tick_params(axis="y", labelsize=28)
# val_ax_scenario3.set_xlabel("Runs of Grid-search", fontsize=28)
# val_ax_scenario3.set_ylabel("Mean validation accuracy (Scenario 3)", fontsize=28)

# formatter = mticker.FormatStrFormatter("%.2f")
# ax_val.yaxis.set_major_formatter(formatter)
# val_ax_scenario3.yaxis.set_major_formatter(formatter)

# Adding legends
val_ax_scenario1.legend(
    [val_bp_scenario1_micro["boxes"][0], val_bp_scenario1_macro["boxes"][0]],
    ["Micro-Averaged", "Macro-Averaged"],
    loc="lower right",
    fontsize=28,
)
val_ax_scenario2.legend(
    [val_bp_scenario2_micro["boxes"][0], val_bp_scenario2_macro["boxes"][0]],
    ["Micro-Averaged", "Macro-Averaged"],
    loc="lower right",
    fontsize=28,
)

val_ax_scenario3.legend(
    [val_bp_scenario3_micro["boxes"][0], val_bp_scenario3_macro["boxes"][0]],
    ["Micro-Averaged", "Macro-Averaged"],
    loc="lower right",
    fontsize=28,
)


###############################################################
###############################################################
#                           Test  subplots
###############################################################
###############################################################


# Generating micro-averaged and macro-averaged data
test_micro_averaged_data_scenario1 = [
    t.rvs(
        df=degrees_of_freedom, loc=0.60 if i < 5 else 0.76, scale=variances[i], size=40
    )
    + random.choice(offsets)
    for i in range(num_runs)
]
test_macro_averaged_data_scenario1 = [
    t.rvs(
        df=degrees_of_freedom, loc=0.60 if i < 5 else 0.76, scale=variances[i], size=40
    )
    + random.choice(offsets)
    for i in range(num_runs)
]
# Creating a facet plot with two columns and one row, where each subplot is similar to the previous boxplots.
# The left subplot will be for mean testidation accuracies and the right one for mean test accuracies.

# Additional data for scenario2 accuracies
test_micro_averaged_data_scenario2 = [
    t.rvs(
        df=degrees_of_freedom, loc=0.67 if i < 5 else 0.82, scale=variances[i], size=40
    )
    + random.choice(offsets)
    for i in range(num_runs)
]
test_macro_averaged_data_scenario2 = [
    t.rvs(
        df=degrees_of_freedom, loc=0.67 if i < 5 else 0.82, scale=variances[i], size=40
    )
    + random.choice(offsets)
    for i in range(num_runs)
]

# Additional data for scenario2 accuracies
test_micro_averaged_data_scenario3 = [
    t.rvs(
        df=degrees_of_freedom, loc=0.67 if i < 5 else 0.84, scale=variances[i], size=40
    )
    + random.choice(offsets)
    for i in range(num_runs)
]
test_macro_averaged_data_scenario3 = [
    t.rvs(
        df=degrees_of_freedom, loc=0.67 if i < 5 else 0.84, scale=variances[i], size=40
    )
    + random.choice(offsets)
    for i in range(num_runs)
]

# Create a facet plot
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(36, 12))

# Plotting for scenario 1
test_ax_scenario1 = axs[1, 0]
test_bp_scenario1_micro = test_ax_scenario1.boxplot(
    test_micro_averaged_data_scenario1,
    positions=np.arange(1, 2 * num_runs, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
test_bp_scenario1_macro = test_ax_scenario1.boxplot(
    test_macro_averaged_data_scenario1,
    positions=np.arange(2, 2 * num_runs + 1, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
for box in test_bp_scenario1_micro["boxes"]:
    box.set(facecolor="#1b9e77")  # Micro-averaged color
for box in test_bp_scenario1_macro["boxes"]:
    box.set(facecolor="#7570b3")  # Macro-averaged color

# Calculate the number of ticks needed
# num_ticks = num_runs // 5 + (1 if num_runs % 5 != 0 else 0)

tick_positions = np.arange(9.5, 2 * num_runs, 10)

# Tick labels corresponding to runs 5, 10, 15, and 20
tick_labels = ["5", "10", "15", "20"]

test_ax_scenario1.set_xticks(tick_positions)
test_ax_scenario1.set_xticklabels(tick_labels, rotation=0, ha="right", fontsize=28)
# test_ax_scenario1.set_yticklabels(test_ax_scenario1.get_yticks(), fontsize=28)
test_ax_scenario1.tick_params(axis="y", labelsize=28)
# test_ax_scenario1.set_xlabel("Runs of Grid-search", fontsize=28)
# test_ax_scenario1.set_ylabel("Mean testidation accuracy (Scenario 1)", fontsize=28)

# Plotting for scenario 2
test_ax_scenario2 = axs[1, 1]
test_bp_scenario2_micro = test_ax_scenario2.boxplot(
    test_micro_averaged_data_scenario2,
    positions=np.arange(1, 2 * num_runs, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
test_bp_scenario2_macro = test_ax_scenario2.boxplot(
    test_macro_averaged_data_scenario2,
    positions=np.arange(2, 2 * num_runs + 1, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
for box in test_bp_scenario2_micro["boxes"]:
    box.set(facecolor="#1b9e77")  # Micro-averaged color
for box in test_bp_scenario2_macro["boxes"]:
    box.set(facecolor="#7570b3")  # Macro-averaged color
test_ax_scenario2.set_xticks(tick_positions)
test_ax_scenario2.set_xticklabels(tick_labels, rotation=0, ha="right", fontsize=28)
# test_ax_scenario2.set_yticklabels(test_ax_scenario2.get_yticks(), fontsize=28)
test_ax_scenario2.tick_params(axis="y", labelsize=28)
# test_ax_scenario2.set_xlabel("Runs of Grid-search", fontsize=28)
# test_ax_scenario2.set_ylabel("Mean testidation accuracy (Scenario 2)", fontsize=28)


# Plotting for scenario 3
test_ax_scenario3 = axs[1, 2]
test_bp_scenario3_micro = test_ax_scenario3.boxplot(
    test_micro_averaged_data_scenario3,
    positions=np.arange(1, 2 * num_runs, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
test_bp_scenario3_macro = test_ax_scenario3.boxplot(
    test_macro_averaged_data_scenario3,
    positions=np.arange(2, 2 * num_runs + 1, 2),
    widths=0.8,
    patch_artist=True,
    meanline=True,
    showmeans=True,
)
for box in test_bp_scenario3_micro["boxes"]:
    box.set(facecolor="#1b9e77")  # Micro-averaged color
for box in test_bp_scenario3_macro["boxes"]:
    box.set(facecolor="#7570b3")  # Macro-averaged color
test_ax_scenario3.set_xticks(tick_positions)
test_ax_scenario3.set_xticklabels(tick_labels, rotation=0, ha="right", fontsize=28)
# test_ax_scenario3.set_yticklabels(test_ax_scenario3.get_yticks(), fontsize=28)
test_ax_scenario3.tick_params(axis="y", labelsize=28)
# test_ax_scenario3.set_xlabel("Runs of Grid-search", fontsize=28)
# test_ax_scenario3.set_ylabel("Mean testidation accuracy (Scenario 3)", fontsize=28)

# formatter = mticker.FormatStrFormatter("%.2f")
# ax_test.yaxis.set_major_formatter(formatter)
# test_ax_scenario3.yaxis.set_major_formatter(formatter)

# Adding legends
test_ax_scenario1.legend(
    [test_bp_scenario1_micro["boxes"][0], test_bp_scenario1_macro["boxes"][0]],
    ["Micro-Averaged", "Macro-Averaged"],
    loc="lower right",
    fontsize=28,
)
test_ax_scenario2.legend(
    [test_bp_scenario2_micro["boxes"][0], test_bp_scenario2_macro["boxes"][0]],
    ["Micro-Averaged", "Macro-Averaged"],
    loc="lower right",
    fontsize=28,
)

test_ax_scenario3.legend(
    [test_bp_scenario3_micro["boxes"][0], test_bp_scenario3_macro["boxes"][0]],
    ["Micro-Averaged", "Macro-Averaged"],
    loc="lower right",
    fontsize=28,
)

# Adding overall axis labels for the entire figure
fig.text(
    0.5, -0.02, "Runs of Grid-search", ha="center", va="center", fontsize=28
)  # X-axis label
# Adding y-axis labels for each row
fig.text(
    -0.02,
    0.75,
    "Mean Validation Accuracy",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=28,
)  # Y-axis label for row 1
fig.text(
    -0.02,
    0.25,
    "Mean Test Accuracy",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=28,
)  # Y-axis label for row 2


from matplotlib.lines import Line2D

# # ... [your existing plotting code] ...

# # Calculate the coordinates for the X and Y axis lines
# left = min(ax.get_position().x0 for ax in axs.flat)
# right = max(ax.get_position().x1 for ax in axs.flat)
# bottom = min(ax.get_position().y0 for ax in axs[1])  # Bottom row for x-axis
# top = max(ax.get_position().y1 for ax in axs[0])  # Top row for y-axis

# # Draw X axis line
# fig.add_artist(Line2D([left, right], [bottom, bottom], color="black", linewidth=2))

# # Draw Y axis line
# fig.add_artist(Line2D([left, left], [bottom, top], color="black", linewidth=2))
from matplotlib.ticker import FormatStrFormatter

for ax in axs.flat:
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

# Determine a common y-axis range suitable for all your data
common_y_min = 0.55  # Replace with your actual minimum
common_y_max = 0.9  # Replace with your actual maximum

# Optionally, define a set of y-axis ticks
common_y_ticks = np.arange(0.55, 0.91, 0.05)  # Adjust the range and step as needed

# Apply the common y-axis range and ticks to all subplots
for ax in axs.flat:
    ax.set_ylim(common_y_min, common_y_max)
    ax.set_yticks(common_y_ticks)  # Optional, if you want specific tick positions
    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.2f")
    )  # Format to 2 decimal places

# from matplotlib.lines import Line2D

# # Calculate the coordinates for the X and Y axis lines
# left = min(ax.get_position().x0 for ax in axs.flat)
# right = max(ax.get_position().x1 for ax in axs.flat)
# bottom = min(ax.get_position().y0 for ax in axs[1])  # Bottom row for x-axis
# top = max(ax.get_position().y1 for ax in axs[0])  # Top row for y-axis

# # Draw X axis line
# fig.add_artist(
#     Line2D(
#         [left - 0.13, right - 0.13],
#         [bottom - 0.12, bottom - 0.12],
#         color="black",
#         linewidth=2,
#     )
# )

# # Draw Y axis line
# fig.add_artist(
#     Line2D(
#         [left - 0.13, left - 0.13],
#         [bottom - 0.13, top - 0.13],
#         color="black",
#         linewidth=2,
#     )
# )

plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()
# -

fig.savefig(
    path_to_plots + "grid_search_results_Tillage.pdf", format="pdf", bbox_inches="tight"
)


# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# Generating three different confusion matrices
conf_matrices = [
    np.array([[48, 7, 6], [10, 53, 10], [1, 7, 36]]),
    np.array([[51, 4, 6], [5, 58, 10], [2, 14, 38]]),
    np.array([[50, 6, 5], [3, 62, 8], [1, 3, 40]]),
]

# Plotting parameters
x_labels = ["CT", "MT", "NT"]
y_labels = ["CT", "MT", "NT"]
cmap = LinearSegmentedColormap.from_list("custom_green", ["white", "#1b9e77"], N=256)

# Create a 1-row, 3-column grid of subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

for idx, conf_matrix in enumerate(conf_matrices):
    heatmap = sns.heatmap(
        conf_matrix, annot=False, fmt="d", cmap=cmap, cbar=True, ax=axes[idx]
    )
    cbar = heatmap.collections[0].colorbar
    # cbar.set_label("Number of Samples", fontsize=24)
    cbar.ax.tick_params(labelsize=14)

    for i, row in enumerate(conf_matrix):
        for j, value in enumerate(row):
            color = "white" if value > 20 else "black"
            axes[idx].text(
                j + 0.5,
                i + 0.5,
                str(value),
                ha="center",
                va="center",
                color=color,
                fontsize=14,
            )

    axes[idx].set_xlabel("Predicted Class", fontsize=14)
    axes[idx].set_ylabel("Actual Class", fontsize=14)
    axes[idx].set_xticks([0.5 + i for i in range(len(x_labels))])
    axes[idx].set_xticklabels(x_labels, fontsize=14, rotation=0)
    axes[idx].set_yticks([0.5 + i for i in range(len(y_labels))])
    axes[idx].set_yticklabels(y_labels, fontsize=14, rotation=0)

plt.subplots_adjust(wspace=1)
plt.tight_layout()
# plt.savefig("facet_confusion_matrices.pdf", format="pdf", bbox_inches="tight")
plt.show()
# -

fig.savefig(
    path_to_plots + "best_models_Tillage2.pdf", format="pdf", bbox_inches="tight"
)

# +
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Data from the image plot
hyperparameters = [0, 0.3, 0.6, 0.9, 2, 5, 8, 11, 14, 17]
mean_macro_accuracy = [0.76, 0.76, 0.85, 0.87, 0.83, 0.81, 0.77, 0.75, 0.73, 0.71]
mean_test_accuracy = [0.78, 0.79, 0.86, 0.86, 0.83, 0.79, 0.76, 0.75, 0.72, 0.74]

# Creating the plot with adjusted figure size
plt.figure(figsize=(10, 6))
# Plotting validation accuracy line
plt.plot(
    hyperparameters,
    mean_macro_accuracy,
    marker="o",
    label="Validation Accuracy",
    color="#7570b3",
)
# Plotting test accuracy line
plt.plot(
    hyperparameters,
    mean_test_accuracy,
    marker="s",
    label="Test Accuracy",
    color="#1b9e77",
)

# # Adding the labels next to each data point for validation accuracy
# for i, txt in enumerate(hyperparameters):
#     plt.annotate(
#         txt,
#         (hyperparameters[i], mean_macro_accuracy[i]),
#         fontsize=12,
#         textcoords="offset points",
#         xytext=(25, -5),
#         ha="center",
#     )

# # Adding the labels next to each data point for test accuracy
# for i, txt in enumerate(hyperparameters):
#     plt.annotate(
#         txt,
#         (hyperparameters[i], mean_test_accuracy[i]),
#         fontsize=12,
#         textcoords="offset points",
#         xytext=(25, -5),
#         ha="center",
#     )


# Setting the axis labels with increased font size
plt.xlabel("Weighting exponent (a)", fontsize=14)
plt.ylabel("Maco-averaged accuracy", fontsize=14)

# Increasing font sizes for x and y ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Adding a legend with manually selected colors and increased font size
legend = plt.legend(loc="best", facecolor="white", fontsize=12)
# Changing legend text color manually
# for text in legend.get_texts():
#     text.set_color("red")
# Setting the format for the x-axis to be scalar
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.savefig(path_to_plots + "a_accuracies.pdf", format="pdf", bbox_inches="tight")
# Displaying the plot
plt.show()
# -

plt.savefig(path_to_plots + "a_accuracies.pdf", format="pdf", bbox_inches="tight")

# +
import matplotlib.pyplot as plt

# Define the data for the pie charts
data = {
    "Whitman": {2012: [51.0, 29.0, 20.0], 2017: [65.0, 15.0, 20.0]},
    "Columbia": {2017: [42.0, 26.0, 32.0], 2017: [53.0, 43.0, 4.0]},
    "Whitman": {2012: [51.0, 29.0, 20.0], 2017: [65.0, 15.0, 20.0]},
    "Columbia": {2017: [42.0, 26.0, 32.0], 2017: [53.0, 43.0, 4.0]},
    "Whitman": {2012: [51.0, 29.0, 20.0], 2017: [65.0, 15.0, 20.0]},
    "Columbia": {2017: [42.0, 26.0, 32.0], 2017: [53.0, 43.0, 4.0]},
    "Whitman": {2012: [51.0, 29.0, 20.0], 2017: [65.0, 15.0, 20.0]},
    "Columbia": {2017: [42.0, 26.0, 32.0], 2017: [53.0, 43.0, 4.0]}
}

# Define the colors and labels for the pie charts
colors = ["#F1B845", "#B0AB3B", "#991F35"]
# labels = ["NT", "MT", "CT"]
labels = ["", "", ""]
explode = (0.1, 0, 0)  # explode the 1st slice for better visibility

# Create subplots with 2 rows and 3 columns
fig, axes = plt.subplots(2, 4, figsize=(6, 4))


def make_autopct(fontsize):
    def my_autopct(pct):
        return ("%.1f%%" % pct) if pct > 0 else ""

    return my_autopct


# Generate pie charts for each subplot
for i, (region, years) in enumerate(data.items()):
    for j, (year, values) in enumerate(years.items()):
        ax = axes[j, i]
        ax.pie(
            values,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct=make_autopct(fontsize=12),
            shadow=True,
            startangle=140,
        )
        ax.set_title(f"{region} {year}")

# Adjust the layout to prevent overlapping
plt.tight_layout()

# Save the figure to a file
# plt.savefig("/mnt/data/agricultural_pie_charts.png")

# Display the plot
plt.show()
