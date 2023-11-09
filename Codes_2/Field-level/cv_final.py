# ---
# jupyter:
#   jupytext:
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
        self.rf = RandomForestClassifier(n_estimators=self.n_estimators,
                                         max_depth=self.max_depth, **kwargs)
    
    def fit(self, X, y, **kwargs):
        target_weights_dict = calculate_custom_weights(y, self.a)
        target_weights = np.array([target_weights_dict[sample] for sample in y])

        # Rest of the weight calculation can stay same
        feature_cols = ['ResidueType']
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
    

# # Read data
path_to_data = ("/Users/aminnorouzi/Library/CloudStorage/"
                "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
                "Projects/Tillage_Mapping/Data/field_level_data/FINAL_DATA/")

# path_to_data = ("/home/amnnrz/OneDrive - "
#                 "a.norouzikandelati/Ph.D/Projects/Tillage_Mapping/Data"
#                 "/field_level_data/FINAL_DATA/")

df = pd.read_csv(path_to_data + "season_finalData.csv", index_col=0)
df = df.dropna(subset=["Tillage", "ResidueType", "ResidueCov"])
########################################################################
########################################################################
########################################################################
# df_Opt = df.iloc[:,
#     list(np.arange(0, 1205))]






########################################################################
########################################################################
########################################################################
# Split df into two dataframes. It is important that each category
# in columns "Tillage", "ResidueType", "ResidueCov" has roughly equal counts
# in both dataframes.

# We split it based on Tillage and see if it works for the two features also:
def split_dataframe(df, column):
    unique_values = df[column].unique()
    dfs1 = []
    dfs2 = []

    for value in unique_values:
        temp_df = df[df[column] == value].sample(frac=1) \
        .reset_index(drop=True) # Shuffle
        midpoint = len(temp_df) // 2
        dfs1.append(temp_df.iloc[:midpoint])
        dfs2.append(temp_df.iloc[midpoint:])

    df1 = pd.concat(dfs1, axis=0).sample(frac=1) \
        .reset_index(drop=True) # Shuffle after concatenating
    df2 = pd.concat(dfs2, axis=0).sample(frac=1) \
        .reset_index(drop=True)

    return df1, df2

df1, df2 = split_dataframe(df, 'Tillage')
df1 = df1.set_index('pointID')
df2 = df2.set_index('pointID')

# Lets check number of each category in the "Tillage", "ResidueType",
# "ResidueCov" for both dataframes
print(df1["Tillage"].value_counts(), df2["Tillage"].value_counts())
print("\n")
print(df1["ResidueType"].value_counts(), df2["ResidueType"].value_counts())
print("\n")
print(df1["ResidueCov"].value_counts(), df2["ResidueCov"].value_counts())

df = pd.concat([df1, df2])
# -

df__ = df[['Tillage', 'ResidueCov', 'ResidueType']]
df__ = df__.reset_index(drop=True)
df__

# +
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define a custom function for autopct to display the count
def absolute_value(val, allvals):
    absolute = int(np.round(val/100.*allvals.sum()))
    return f"{absolute:d}" if val > 1 else ' '

# Create the FacetGrid with empty axes
g = sns.FacetGrid(df__, row='ResidueType', col='ResidueCov', margin_titles=True, despine=False,
                  col_order=['0-15%', '16-30%', '>30%'])
g.fig.set_size_inches(14, 8)  # Adjust the figure size as needed


# Define custom colors for the Tillage categories
colors = ['#7c646e', '#94bba9', '#3e8245']  # Replace with your preferred colors
tillage_order = ['ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed']

# Plot the pie charts on each FacetGrid axis
for (row_val, col_val), ax in g.axes_dict.items():
    # Filter the dataframe for this subset
    subset = df__[(df__['ResidueType'] == row_val) & (df__['ResidueCov'] == col_val)]
    # Get the value counts of the 'Tillage' column for this subset
    tillage_counts = subset['Tillage'].value_counts().reindex(tillage_order, fill_value=0)
    # If there are no counts, continue to the next subplot
    if tillage_counts.sum() == 0:
        continue
    # Plot pie chart on the current axis
    ax.pie(tillage_counts, labels=None, autopct=lambda pct: absolute_value(pct, tillage_counts),
            colors=colors, startangle=90, textprops={'size': 18, 'fontweight': 'bold'})

# Add a legend
legend_patches = [Patch(color=colors[i], label=tillage_order[i]) for i in range(len(tillage_order))]
plt.legend(handles=legend_patches, title='Tillage', loc='center left', bbox_to_anchor=(1.3, 0.5), fontsize='large')

# Set titles and adjust layout
g.set_titles(col_template='{col_name}', row_template='{row_name}', size = 18)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle('Proportion of Tillage Categories by Residue Type and Coverage', size=16)

# Display the plot
plt.show()

# -

df.iloc[:, [2, 4, 5] + list(np.arange(7, df.shape[1]))].iloc[0:20, 0:20]


# +
import pandas as pd

# Assuming you have a pandas DataFrame called 'df'

# Get the column names as a list
columns = X_train.columns.tolist()

# Find the index of the first column that starts with "VH_"
first_column_index = next((i for i, col in enumerate(columns) if col.startswith('VH_')), None)

# 'first_column_index' will be the index of the first column starting with 'VH_'
# If no such column is found, it will be None

print(first_column_index)


# +
# X_train.shape, X_test.shape
# X_train_Opt = X_train.iloc[:,
#     list(np.arange(0, 1199))]

# X_test_Opt = X_test.iloc[:,
#     list(np.arange(0, 1199))]

# +
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
# from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict


# df1 = df1.drop(columns='ResidueCov')

# Load your dataframe with categorical features
df = df

# # # Perform one-hot encoding for "Residue Cover" features
# df_encoded = pd.get_dummies(df, columns=['ResidueCov'])
df_encoded = df

# Encode "ResidueType"
encode_dict_Restype = {
    'grain': 1,
    'legume': 2,
    'canola': 3
}
df_encoded['ResidueType'] = df_encoded['ResidueType'].replace(encode_dict_Restype)

# Encode "ResidueCov"
encode_dict_ResCov = {
    '0-15%': 1,
    '16-30%': 2,
    '>30%': 3
}
df_encoded['ResidueCov'] = df_encoded['ResidueCov'].replace(encode_dict_ResCov)

# Remove NA from Tillage
df_encoded = df_encoded.dropna(subset=["Tillage", "ResidueCov", "ResidueType"])

# Split features and target variable
X = df_encoded.iloc[:,
    [2,4] + list(np.arange(7, df_encoded.shape[1]))]
    
y = df_encoded['Tillage']

# Impute missing values with the median
X = X.fillna(X.median())



# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 300],
    # 'n_estimators': [30],
    'max_depth': [5, 40, 55],
    # 'a': list(np.arange(-10, 10, 0.5))
    'a': list(np.concatenate((np.arange(0,1,0.3), np.arange(2,12,3))))
}

# Perform cross-validation for 50 times and calculate accuracies
mean_accuracies = []
best_model = None
best_val_accuracy = 0
feature_counter = Counter()  # Counter to keep track of feature occurrences

# Initialize a list to store mean test scores for each hyperparameter combination
mean_test_scores = []

# initialize a list to store mean validation accuracies for each value of "a"
a_vs_accuracy = {a_value: [] for a_value in param_grid['a']}
a_cm = []
for _ in range(2):

    if _ == 1:  # After the first three loops
        top_50_features = [feature[0]
            for feature in feature_counter.most_common(50)]
        selected_features = top_50_features
        # Adjust training and test sets to include only these 50 features
        selected_features = ['ResidueType'] + \
        list(X_train.iloc[:, np.array(top_50_features)].columns)
        selected_features
        list_without_duplicates = list(OrderedDict.fromkeys(selected_features))
        
        X_train_selected = X_train[list_without_duplicates]
        X_test_selected = X_test[list_without_duplicates]



    grid_search = GridSearchCV(
        CustomWeightedRF(), param_grid, cv=3, return_train_score=False)
    grid_search.fit(X_train, y_train)

    print(grid_search.cv_results_['mean_test_score'].shape)

    # Update the a_vs_accuracy dictionary with the mean validation accuracies 
    # for each value of "a"
    for i, a_value in enumerate(param_grid['a']):
        a_vs_accuracy[a_value].append(grid_search.cv_results_[
            'mean_test_score'][i::len(param_grid['a'])].mean())
        
        current_model = grid_search.best_estimator_
        y_pred = current_model.predict(X_test)
        a_cm += [confusion_matrix(y_test, y_pred)]


    # Store mean test scores in the list
    mean_test_scores.append(grid_search.cv_results_['mean_test_score'])

    # Get the best model and its predictions
    current_model = grid_search.best_estimator_
    y_pred = current_model.predict(X_test)  # Use the test data for prediction

    def macro_accuracy(y_true, y_pred):
        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Calculate accuracy for each class
        class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        
        # Compute the macro-averaged accuracy
        macro_avg_accuracy = np.nanmean(class_accuracies)
    
        return macro_avg_accuracy

    # Calculate the accuracy for the current run
    val_accuracy = macro_accuracy(y_test, y_pred)
    print(_, ":", "Validation Accuracy is ", val_accuracy)
    mean_accuracies.append(val_accuracy)

    # Update the best model if the current model has a higher validation accuracy
    if val_accuracy > best_val_accuracy:
        best_model = current_model
        best_val_accuracy = val_accuracy

    # Update the feature counter with the top 50 important features of the current model
    top_50_indices = current_model.feature_importances_.argsort()[::-1][:50]
    top_50_features = X.columns[top_50_indices]
    feature_counter.update(top_50_indices)

# Calculate mean accuracy across the 20 runs
mean_accuracy = sum(mean_accuracies) / len(mean_accuracies)

# Print accuracies for all cross-validations
print("Accuracies for all cross-validations:")
for i, accuracy in enumerate(mean_accuracies, 1):
    print(f"Cross-Validation {i}: {accuracy:.4f}")

# Print mean accuracy
print(f"Mean Accuracy: {mean_accuracy:.4f}")

# print hyperparameters of the best model 
print("Best hyperparameters for the model:", grid_search.best_params_)

# -




# +
param_grid = {
    'n_estimators': [50, 100, 300],
    # 'n_estimators': [30],
    'max_depth': [5, 40, 55],
    # 'a': list(np.arange(-10, 10, 0.5))
    'a': list(np.concatenate((np.arange(0,1,0.3), np.arange(2,12,3))).round(2))
}

# Perform cross-validation for 50 times and calculate accuracies
mean_accuracies = []
best_model = None
best_val_accuracy = 0
feature_counter = Counter()  # Counter to keep track of feature occurrences

# Initialize a list to store mean test scores for each hyperparameter combination
mean_test_scores = []

# initialize a list to store mean validation accuracies for each value of "a"
all_a_vs_accuracies = []
a_vs_accuracy = {a_value: [] for a_value in param_grid['a']}

# Initialize dictionaries to store predictions and confusion matrices for each value of 'a'
a_predictions = {a_value: [] for a_value in param_grid['a']}
a_cm = {a_value: [] for a_value in param_grid['a']}

best_models = []
for _ in np.arange(1):
    for a_value in param_grid['a']:
        print(f'a is {a_value}')
        # Create a new param grid with only the current value of 'a'
        current_param_grid = {
            'n_estimators': param_grid['n_estimators'],
            'max_depth': param_grid['max_depth'],
            'a': [a_value]
        }
        grid_search = GridSearchCV(
            CustomWeightedRF(), current_param_grid, cv=3,
              return_train_score=False)
        grid_search.fit(X_train_selected, y_train)

        print(grid_search.cv_results_['mean_test_score'].shape)

        # Get the best model for the current 'a' value
        current_model = grid_search.best_estimator_
        
        # Make predictions and store them
        y_pred = current_model.predict(X_test_selected)
        a_predictions[a_value].append(y_pred)
        
        # Compute the confusion matrix for the current 'a' value and append it
        cm = confusion_matrix(y_test, y_pred)
        a_cm[a_value].append(cm)

        # Update the a_vs_accuracy dictionary with the mean validation accuracies 
        # for each value of "a"
        for i, a_value in enumerate(current_param_grid['a']):
            a_vs_accuracy[a_value].append(grid_search.cv_results_[
                'mean_test_score'][i::len(current_param_grid['a'])].mean())

        all_a_vs_accuracies += all_a_vs_accuracies + [a_vs_accuracy]
        # Store mean test scores in the list
        mean_test_scores.append(grid_search.cv_results_['mean_test_score'])

        def macro_accuracy(y_true, y_pred):
            # Compute the confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            # Calculate accuracy for each class
            class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
            
            # Compute the macro-averaged accuracy
            macro_avg_accuracy = np.nanmean(class_accuracies)
        
            return macro_avg_accuracy

        # Calculate the accuracy for the current run
        val_accuracy = macro_accuracy(y_test, y_pred)
        print(_, ":", "Validation Accuracy is ", val_accuracy)
        mean_accuracies.append(val_accuracy)

        # Update the best model if the current model has a higher validation accuracy
        if val_accuracy > best_val_accuracy:
            best_model = current_model
            best_val_accuracy = val_accuracy

        # Update the feature counter with the top 50 important features of the current model
        top_50_indices = current_model.feature_importances_.argsort()[::-1][:50]
        top_50_features = X.columns[top_50_indices]
        feature_counter.update(top_50_indices)

# Calculate mean accuracy across the 20 runs
mean_accuracy = sum(mean_accuracies) / len(mean_accuracies)

# Print accuracies for all cross-validations
print("Accuracies for all cross-validations:")
for i, accuracy in enumerate(mean_accuracies, 1):
    print(f"Cross-Validation {i}: {accuracy:.4f}")

# Print mean accuracy
print(f"Mean Accuracy: {mean_accuracy:.4f}")

# print hyperparameters of the best model 
print("Best hyperparameters for the model:", grid_search.best_params_)

# Create a confusion matrix using predictions from the best model
y_pred_best = best_model.predict(X_test_selected)
cm = confusion_matrix(y_test, y_pred_best)

# Plot the confusion matrix
labels = ['ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed']
# labels = ['MinimumTill', 'NoTill-DirectSeed']
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

plt.ylabel('True label')
plt.xlabel('Predicted label')

# Displaying the values in the cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.tight_layout()
plt.show()


# Print the features that appeared most frequently in the top 50 important features
most_common_features = feature_counter.most_common()
print("Features that appeared most frequently in the top 50 important features:")
for feature, count in most_common_features:
    print(f"{feature}: {count} times")

top_50_features = [X_train_selected.columns[feature[0]] for feature in most_common_features[:50]]
top_50_importances = [feature[1] for feature in most_common_features[:50]]

plt.figure(figsize=(10, 8))
plt.barh(top_50_features, top_50_importances)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Top 50 Most Important Features')
plt.show()

# After printing important features, plot the boxplot for validation accuracies
plt.figure(figsize=(10, 8))
plt.boxplot(mean_test_scores, vert=False)
plt.xlabel('Mean Cross-Validated Accuracy')
plt.ylabel('Hyperparameter Combination')
plt.title('Boxplot of Validation Accuracies for each Hyperparameter Combination')
plt.show()

# Plot a vs mean validation accuracy
plt.figure(figsize=(10, 6))
for a_value, accuracies in a_vs_accuracy.items():
    plt.plot(accuracies, label=f'a={a_value}')
plt.xlabel('Iteration')
plt.ylabel('Mean Validation Accuracy')
plt.title('Hyperparameter "a" vs. Mean Validation Accuracy for Each Iteration')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for a_value, accuracies in a_vs_accuracy.items():
    plt.scatter([a_value] * len(accuracies), accuracies, label=f'a={a_value}')
plt.xlabel('Hyperparameter "a"')
plt.ylabel('Mean Validation Accuracy')
plt.title('Hyperparameter "a" vs. Mean Validation Accuracy for Each Iteration')
# Moved the legend to the right
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# +
# Plot confusion matrix for each a value 
a_cm
for _ in a_cm.keys():
    a_cm[_] = a_cm[_][0]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns
labels = ['ConventionalTill', 'MinimumTill', 'NoTill-DirectSeed']

# Flatten the axes array for easy iteration
axes_flat = axes.flatten()

for ax, (title, matrix) in zip(axes_flat, a_cm.items()):
    matrix = np.array(matrix, dtype=int)
    # print(matrix)
    sns.heatmap(matrix, ax=ax, cmap="Blues", fmt='d')
        # Manually add the annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j + 0.5, i + 0.5, int(matrix[i, j]), 
                    ha='center', va='center', color='black', fontsize=15)
    
    ax.set_title(f'a = {title}')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    # Set the labels
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
# # If there are any empty subplots, turn them off
# for ax in axes_flat[len(a_cm):]:
#     ax.axis('off')

plt.tight_layout()
plt.show()


# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

a_predictions_ = a_predictions.copy()
for _ in a_predictions_.keys():
    a_predictions_[_] = a_predictions_[_][0]

fig, axes = plt.subplots(2, 4, figsize=(20, 10),
                         gridspec_kw={'width_ratios': [1, 1, 1, 1],
                                      'height_ratios': [2, 2]})  # 2 rows, 4 columns
# Flatten the axes array for easy iteration
axes_flat = axes.flatten()

# Define the mapping from string labels to integers
label_mapping = {'ConventionalTill': 1, 'MinimumTill': 2, 'NoTill-DirectSeed': 3}


for ax, (title, pred) in zip(axes_flat, a_predictions_.items()):
    pred_mapped = np.array([label_mapping[label] for label in pred])
    predictions = np.array(pred_mapped, dtype=int)







    # Set aesthetic styles for the plot
    plt.style.use('ggplot')

    y_test_mapped = np.array(y_test.replace(label_mapping))
    zero_one_loss = np.where(y_test_mapped != predictions, 1, 0)

    # Combine the zero-one loss and original features into a new DataFrame
    loss_df = pd.DataFrame({'loss': zero_one_loss, 'pointID': X_test.index})

    df.loc[:, 'pointID'] = df.index.values
    df_ = df.reset_index(drop=True)

    loss_df = pd.merge(
        loss_df, df_[['pointID', 'ResidueType']], on='pointID', how='left')

    # Define a color for each Croptype, using colorblind-friendly and harmonious colors
    croptype_colors = {'legume': '#5ec962',
                    'canola': '#fde725', 'grain': '#b5de2b'}

    # Prepare data for stacked histogram and collect sample counts
    croptypes = ['legume', 'canola', 'grain']

    # Encode "ResidueType"
    encode_dict = {
        1 :'grain',
        2: 'legume',
        3: 'canola'
    }
    loss_df['ResidueType'] = loss_df['ResidueType'].replace(encode_dict)
    data = [loss_df[loss_df['ResidueType'] == croptype]
            ['loss'].values for croptype in croptypes]
    sample_counts = [len(d) for d in data]

    # Create labels with sample counts for the legend
    labels_with_counts = [f"{croptype} (n = {count})" for croptype, count in zip(
        croptypes, sample_counts)]


    n, bins, patches = ax.hist(data, bins=[-0.5, 0.5, 1.5], stacked=True,
                            color=[croptype_colors[c] for c in croptypes],
                            edgecolor='white', linewidth=1)

    cumulative_heights = np.zeros(len(bins) - 1)  # Initialize cumulative heights


    for i, bars in enumerate(patches):
        for j, bar in enumerate(bars):
            bar_height = bar.get_height()
            bar_center_x = bar.get_x() + bar.get_width() / \
                2.0  # Center the text within the bar

            # Get the bottom of the current bar. If it's the first set of bars, bottom is 0.
            bar_bottom = 0 if i == 0 else patches[i-1][j].get_height()

            # # Only label the bar if its height is greater than 0
            # if bar_height > 0:
            #     # adjust "- 0.5" if needed for better positioning
            #     text_y_position = bar_center_x 
            #     ax.text(bar_center_x, text_y_position, int(bar_height),
            #             ha='center', va='center', color='black', rotation=90)

# Set aesthetic styles for the plot
    ax.set_title(f'a = {title}', fontsize=16)  # Set title for each subplot
    
    # Set x and y labels for each subplot
    ax.set_xlabel('Zero-One Loss', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0', '1'], fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)

    # Legend for each subplot
    ax.legend(title='Croptype', labels=labels_with_counts,
              title_fontsize='8', fontsize='8')
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

   # Table to show the counts below the plot
    cell_text = [list(map(int, [bar.get_height() for bar in bars])) for bars in patches]
    row_labels = [f'{croptype} (n={count})' for croptype, count in zip(croptypes, sample_counts)]
    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=[' ', ' '],
         loc='bottom', cellLoc='center', bbox=[0.2, -0.5, 0.6, 0.2])
    for key, cell in table.get_celld().items():
        cell.set_fontsize(10)  # Adjust font size as needed
        cell.set_height(0.1)   # Adjust height as needed
        cell.set_width(0.2)    # Adjust width as needed

    # Adjust bbox to scale the entire table if necessary
    table.scale(1.2, 1.2)

# Adjust the subplot parameters globally
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.9)  # Adjust as needed
plt.subplots_adjust(hspace=1, wspace=1)  # Adjust as needed

plt.show()

