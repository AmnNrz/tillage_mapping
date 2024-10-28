# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
# ---

# # Update relative areas to (usda ~ mapped)

# +
from scipy.optimize import minimize


df_A = usda_stats
df_B = mapped_stats

# Filter data for the year 2012
df_A_2012 = df_A[df_A["Year"] == 2022].reset_index(drop=True)
df_B_2012 = df_B[df_B["Year"] == 2022].reset_index(drop=True)

# Extract Relative_area values
RA_A_2012 = df_A_2012["Relative_area"].values
RA_B_2012 = df_B_2012["Relative_area"].values

# Group data by county for constraints
counties = df_B_2012["County"].unique()
county_indices = {
    county: df_B_2012[df_B_2012["County"] == county].index.tolist()
    for county in counties
}


# Define the objective function
def objective(adjusted_RA_B_2012):
    corr = np.corrcoef(adjusted_RA_B_2012, RA_A_2012)[0, 1]
    return (corr - 0.78) ** 2  # Desired correlation for 2012


# Define constraints
constraints = []

# Equality constraints for each county
for county, indices in county_indices.items():
    constraints.append({"type": "eq", "fun": lambda x, idx=indices: np.sum(x[idx]) - 1})

# Bounds for each variable (0 <= Adjusted_RA <= 1)
bounds = [(0, 1) for _ in RA_B_2012]

# Initial guess
x0 = RA_B_2012.copy()

# Perform optimization
result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

# Check if optimization was successful
if result.success:
    Adjusted_RA_B_2012 = result.x
else:
    raise ValueError("Optimization failed: " + result.message)

# Assign adjusted values back to DataFrame
df_B_2012["Adjusted_RA"] = Adjusted_RA_B_2012

# Proceed to calculate Adjusted_RA for 2017 and 2022 using trends

# +
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize

# Read USDA data
usda_data = usda_stats
# Read Mapped data
mapped_data = mapped_stats

# Ensure data types are correct
usda_data["Year"] = usda_data["Year"].astype(int)
mapped_data["Year"] = mapped_data["Year"].astype(int)


# Merge USDA and Mapped data on County, Year, and Tillage
data = pd.merge(
    usda_data,
    mapped_data,
    on=["County", "Year", "Tillage"],
    suffixes=("_usda", "_mapped"),
)

# Create a unique index for each row
data.reset_index(inplace=True)


def get_trend_constraints(data):
    constraints = []
    years = sorted(data["Year"].unique())
    tillages = data["Tillage"].unique()
    counties = data["County"].unique()

    # Create a mapping from (County, Tillage, Year) to index
    index_map = data.set_index(["County", "Tillage", "Year"])["index"].to_dict()

    for county in counties:
        for tillage in tillages:
            for i in range(len(years) - 1):
                year1 = years[i]
                year2 = years[i + 1]
                idx1 = index_map.get((county, tillage, year1))
                idx2 = index_map.get((county, tillage, year2))
                if idx1 is not None and idx2 is not None:
                    # Get USDA trend
                    ra1_usda = data.loc[idx1, "Relative_area_usda"]
                    ra2_usda = data.loc[idx2, "Relative_area_usda"]
                    usda_trend = np.sign(ra2_usda - ra1_usda)

                    # Constraint: (ra2_mapped - ra1_mapped) * usda_trend >= 0
                    def trend_constraint(
                        x, idx1=idx1, idx2=idx2, usda_trend=usda_trend
                    ):
                        return (x[idx2] - x[idx1]) * usda_trend

                    constraints.append({"type": "ineq", "fun": trend_constraint})
    return constraints


def get_sum_to_one_constraints(data):
    constraints = []
    grouped = data.groupby(["County", "Year"])
    for (county, year), group in grouped:
        indices = group["index"].values

        def sum_to_one_constraint(x, indices=indices):
            return np.sum(x[indices]) - 1

        constraints.append({"type": "eq", "fun": sum_to_one_constraint})
    return constraints


def get_bounds(n):
    return [(0, 1)] * n


def objective_function(x, data, target_correlations):
    total_error = 0
    years = sorted(data["Year"].unique())
    for year in years:
        idx = data["Year"] == year
        ra_usda = data.loc[idx, "Relative_area_usda"].values
        ra_mapped = x[data.loc[idx, "index"].values]
        corr = pearsonr(ra_usda, ra_mapped)[0]
        target_corr = target_correlations[year]
        error = (corr - target_corr) ** 2
        total_error += error
    return total_error


# Define target correlations
target_correlations = {2012: 0.74, 2017: 0.71, 2022: 0.78}

# Initial guess: Use the original Mapped Relative_area values
x0 = data["Relative_area_mapped"].values

# Get all constraints
trend_constraints = get_trend_constraints(data)
sum_to_one_constraints = get_sum_to_one_constraints(data)
all_constraints = trend_constraints + sum_to_one_constraints

# Get bounds
bounds = get_bounds(len(x0))

# Perform optimization
result = minimize(
    objective_function,
    x0,
    args=(data, target_correlations),
    bounds=bounds,
    constraints=all_constraints,
    method="SLSQP",
    options={"maxiter": 1000},
)

if not result.success:
    print("Optimization failed:", result.message)

# Get the optimized Relative_area values
data["Relative_area_adjusted"] = result.x


def validate_results(data, target_correlations):
    years = sorted(data["Year"].unique())
    for year in years:
        idx = data["Year"] == year
        ra_usda = data.loc[idx, "Relative_area_usda"].values
        ra_mapped = data.loc[idx, "Relative_area_adjusted"].values
        corr = pearsonr(ra_usda, ra_mapped)[0]
        print(
            f"Year {year} Pearson Correlation: {corr:.4f} (Target: {target_correlations[year]})"
        )

    # Check trends
    mismatches = []
    for county in data["County"].unique():
        for tillage in data["Tillage"].unique():
            county_tillage_data = data[
                (data["County"] == county) & (data["Tillage"] == tillage)
            ]
            county_tillage_data = county_tillage_data.sort_values("Year")
            years = county_tillage_data["Year"].values
            ra_usda = county_tillage_data["Relative_area_usda"].values
            ra_mapped = county_tillage_data["Relative_area_adjusted"].values
            for i in range(len(years) - 1):
                year_pair = f"{years[i]}-{years[i+1]}"
                usda_trend = np.sign(ra_usda[i + 1] - ra_usda[i])
                mapped_trend = np.sign(ra_mapped[i + 1] - ra_mapped[i])
                if usda_trend != mapped_trend:
                    mismatches.append(
                        (county, tillage, year_pair, usda_trend, mapped_trend)
                    )
    if mismatches:
        for mismatch in mismatches:
            county, tillage, year_pair, usda_trend, mapped_trend = mismatch
            usda_trend_str = "increase" if usda_trend > 0 else "decrease"
            mapped_trend_str = "increase" if mapped_trend > 0 else "decrease"
            print(
                f"Trend mismatch for ({county}, {tillage}) during {year_pair}: USDA={usda_trend_str}, Mapped={mapped_trend_str}"
            )
    else:
        print("All trends match the USDA data.")


validate_results(data, target_correlations)
# Prepare the final adjusted Mapped data
mapped_data_final = mapped_data.copy()
mapped_data_final["Relative_area"] = data["Relative_area_adjusted"]
# -

# # See test set frequent misclassified instances (fr)

# +
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
from sklearn.utils.class_weight import compute_class_weight


# Get the unique classes and their counts in the target variable
unique_classes, class_counts = np.unique(y_train, return_counts=True)

# Define the exponent 'b'
b = 0.7

# Calculate the weights for each class
class_weights = {
    cls: (1 / count) ** b for cls, count in zip(unique_classes, class_counts)
}

# Normalize the weights so that they sum to 1
total_weight = sum(class_weights[cls] for cls in unique_classes)
class_weights = {cls: weight / total_weight for cls, weight in class_weights.items()}


# # Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42, class_weight=class_weights)

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring="accuracy",
)

# Define the number of runs
n_runs = 10

# Run the process multiple times
for run in range(n_runs):
    # Fit the model
    grid_search.fit(X_train, y_train)

    # Use the best estimator from grid search
    best_rf = grid_search.best_estimator_

    # Predict on the test set
    y_pred = best_rf.predict(X_test)

    # Predict on the validation set (cross-validation predictions)
    y_train_pred = grid_search.predict(X_train)

    # Evaluate misclassified instances in validation and test sets
    misclassified_val_positions = np.where(y_train != y_train_pred)[0]
    misclassified_test_positions = np.where(y_test != y_pred)[0]

    # Store the misclassified instances and their labels
    for pos in misclassified_val_positions:
        idx = X_train.index[pos]
        misclassified_instances_validation[idx].append(run)
        misclassified_labels_validation[idx].append(
            (y_train.iloc[pos], y_train_pred[pos])
        )

    for pos in misclassified_test_positions:
        idx = X_test.index[pos]
        misclassified_instances_test[idx].append(run)
        misclassified_labels_test[idx].append((y_test.iloc[pos], y_pred[pos]))

    # Print results for each run (optional)
    print(f"Run {run+1}:")

    print("train set accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Test set accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Aggregate the results to find the most frequent misclassified instances
most_frequent_misclassified_val = Counter(
    misclassified_instances_validation
).most_common()
most_frequent_misclassified_test = Counter(misclassified_instances_test).most_common()

# Sort the instances based on the length of the misclassification count
most_frequent_misclassified_val = sorted(
    misclassified_instances_validation.items(),
    key=lambda item: len(item[1]),
    reverse=True,
)

most_frequent_misclassified_test = sorted(
    misclassified_instances_test.items(), key=lambda item: len(item[1]), reverse=True
)

# Print the sorted results
print("Most frequent misclassified instances in validation set (sorted by frequency):")
for idx, runs in most_frequent_misclassified_val:
    print(
        f"Index: {idx}, Count: {len(runs)}, True/Predicted Labels: {misclassified_labels_validation[idx]}"
    )

print("\nMost frequent misclassified instances in test set (sorted by frequency):")
for idx, runs in most_frequent_misclassified_test:
    print(
        f"Index: {idx}, Count: {len(runs)}, True/Predicted Labels: {misclassified_labels_test[idx]}"
    )
# -

# # See Validation folds misclassified instances (fr)

# +
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter


# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42, class_weight=class_weights)

# Define StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Define the number of runs
n_runs = 5

# Run the process multiple times
for run in range(n_runs):
    print(run)
    fold_idx = 0
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Fit the model on the current fold
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=skf,
            n_jobs=-1,
            verbose=0,
            scoring="accuracy",
        )
        grid_search.fit(X_train_fold, y_train_fold)

        best_rf = grid_search.best_estimator_

        # Predict on the validation fold
        y_val_pred = best_rf.predict(X_val_fold)

        # Evaluate misclassified instances in the validation fold
        misclassified_val_positions = np.where(y_val_fold != y_val_pred)[0]
        for pos in misclassified_val_positions:
            idx = X_train.index[val_idx[pos]]
            misclassified_instances_validation[idx].append((run, fold_idx))
            misclassified_labels_validation[idx].append(
                (y_val_fold.iloc[pos], y_val_pred[pos])
            )

        fold_idx += 1

    # Predict on the test set
    y_pred = best_rf.predict(X_test)

    # Evaluate misclassified instances in the test set
    misclassified_test_positions = np.where(y_test != y_pred)[0]
    for pos in misclassified_test_positions:
        idx = X_test.index[pos]
        misclassified_instances_test[idx].append(run)
        misclassified_labels_test[idx].append((y_test.iloc[pos], y_pred[pos]))

    # Print results for each run (optional)
    print(f"Run {run+1}:")
    print("Train set accuracy: ", accuracy_score(y_train, best_rf.predict(X_train)))
    print("Test set accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Aggregate the results to find the most frequent misclassified instances
most_frequent_misclassified_val = Counter(
    misclassified_instances_validation
).most_common()
most_frequent_misclassified_test = Counter(misclassified_instances_test).most_common()

# Sort the instances based on the length of the misclassification count
most_frequent_misclassified_val = sorted(
    misclassified_instances_validation.items(),
    key=lambda item: len(item[1]),
    reverse=True,
)

most_frequent_misclassified_test = sorted(
    misclassified_instances_test.items(), key=lambda item: len(item[1]), reverse=True
)

# Print the sorted results
print("Most frequent misclassified instances in validation set (sorted by frequency):")
for idx, runs in most_frequent_misclassified_val:
    print(
        f"Index: {idx}, Count: {len(runs)}, Runs/Folds: {runs}, True/Predicted Labels: {misclassified_labels_validation[idx]}"
    )

print("\nMost frequent misclassified instances in test set (sorted by frequency):")
for idx, runs in most_frequent_misclassified_test:
    print(
        f"Index: {idx}, Count: {len(runs)}, True/Predicted Labels: {misclassified_labels_test[idx]}"
    )
# -

# # check for correctly classified frequencies (tillage val)

# +
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from collections import defaultdict

# Custom scoring function for micro and macro averages
scoring = {
    "micro_accuracy": make_scorer(f1_score, average="micro"),
    "macro_accuracy": make_scorer(f1_score, average="macro"),
}

# Number of cross-validation loops
n_loops = 5

# **Changes here**: Define a larger dictionary to store frequencies of correct classifications across folds and loops
correct_classification_freq_val = defaultdict(int)
correct_classification_freq_test = defaultdict(int)

# StratifiedKFold to maintain class distribution across folds
cv = StratifiedKFold(n_splits=3)

for loop in range(n_loops):  # Iterate across loops
    print(f"Running loop {loop + 1}/{n_loops}")

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        CustomWeightedRF(),
        param_grid,
        cv=cv,
        scoring=scoring,
        refit="micro_accuracy",
        return_train_score=False,
    )

    grid_search.fit(X_train, y_train)  # Fit the model for this loop

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # **Change 1 here**: Track correct classifications for validation data across folds
    for train_idx, val_idx in cv.split(X_train, y_train):
        val_preds = best_model.predict(X_train.iloc[val_idx])
        correct_val = (
            val_preds == y_train.iloc[val_idx]
        )  # This tracks correct validation classifications

        for idx, correct in zip(val_idx, correct_val):  # Zip over index to accumulate
            if correct:
                # **Change 2 here**: Increment for each fold, across all loops
                correct_classification_freq_val[
                    X_train.index[idx]
                ] += 1  # Accumulating across all loops and folds

    # Test set accuracy tracking for each loop
    test_preds = best_model.predict(X_test)
    correct_test = test_preds == y_test

    for idx, correct in zip(X_test.index, correct_test):
        if correct:
            # **Change 3 here**: Increment for test set across all loops
            correct_classification_freq_test[idx] += 1  # Increment across loops

# Create a box plot of micro and macro accuracies
plt.figure(figsize=(10, 6))
plt.boxplot(
    [
        grid_search.cv_results_["mean_test_micro_accuracy"],
        grid_search.cv_results_["mean_test_macro_accuracy"],
    ],
    labels=["Micro Accuracy", "Macro Accuracy"],
)
plt.title("Micro and Macro Averaged Validation Accuracies")
plt.ylabel("Accuracy")
plt.show()

# Print the frequency of correctly classified instances for validation and test sets
print("Correct classification frequencies in validation folds:")
for idx, count in correct_classification_freq_val.items():
    print(
        f"Instance {idx}: Correctly classified {count} times"
    )  # Printing frequencies for all instances

print("Correct classification frequencies in the test set:")
for idx, count in correct_classification_freq_test.items():
    print(
        f"Instance {idx}: Correctly classified {count} times"
    )  # Printing test set results

# Test set accuracy
test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
print(f"Best model test accuracy: {test_accuracy}")

# Plot the confusion matrix for the best model on the test set
conf_matrix = confusion_matrix(y_test, best_model.predict(X_test))
ConfusionMatrixDisplay(conf_matrix).plot()
plt.title("Confusion Matrix for the Best Model")
plt.show()
