# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# +
#%%
import pandas as pd

# # Read data
# path_to_data = ("/Users/aminnorouzi/Library/CloudStorage/"
#                 "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
#                 "Projects/Tillage_Mapping/Data/field_level_data/FINAL_DATA/")

path_to_data = ("/home/amnnrz/OneDrive - "
                "a.norouzikandelati/Ph.D/Projects/Tillage_Mapping/Data"
                "/field_level_data/FINAL_DATA/")

df = pd.read_csv(path_to_data + "metric_finalData.csv", index_col=0)
df = df.dropna(subset=["Tillage", "ResidueType", "ResidueCov"])

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


# Lets check number of each category in the "Tillage", "ResidueType",
# "ResidueCov" for both dataframes
print(df1["Tillage"].value_counts(), df2["Tillage"].value_counts())
print("\n")
print(df1["ResidueType"].value_counts(), df2["ResidueType"].value_counts())
print("\n")
print(df1["ResidueCov"].value_counts(), df2["ResidueCov"].value_counts())



#%%
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
        return self
    
    def predict(self, X, **kwargs):
        return self.rf.predict(X)
    
    def predict_proba(self, X, **kwargs):
        return self.rf.predict_proba(X)
    
    @property
    def feature_importances_(self):
        return self.rf.feature_importances_
    


#%%

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
df = df1

# # # Perform one-hot encoding for "Residue Cover" features
# df_encoded = pd.get_dummies(df, columns=['ResidueCov'])
df_encoded = df

# Encode "ResidueType"
encode_dict = {
    'grain': 1,
    'legume': 2,
    'canola': 3
}
df_encoded['ResidueType'] = df_encoded['ResidueType'].replace(encode_dict)
df_encoded

# # Place the one-hot encoded columns on the left side of the dataframe
# ordered_columns = list(df_encoded.columns.difference(df.columns)) + \
# [col for col in df.columns if col not in ['ResidueCov']]
# df_ordered = df_encoded[ordered_columns]

df_ordered = df_encoded

# Remove NA from Tillage
df_ordered = df_ordered.dropna(subset=["Tillage", "ResidueCov", "ResidueType"])

le = LabelEncoder()
df_ordered['ResidueCov'] = le.fit_transform(df_ordered['ResidueCov'])

# Split features and target variable
X = df_ordered.iloc[:, np.concatenate(
    [np.arange(3, 4), np.arange(8, df_ordered.shape[1])]
    )]
    
y = df_ordered['ResidueCov']

# Impute missing values with the median
X = X.fillna(X.median())



# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    'n_estimators': [40, 50, 100, 300, 500, 1000],
    # 'n_estimators': [30],
    'max_depth': [5, 40, 55, 70, 100],
    # 'a': list(np.arange(-10, 10, 0.5))
    'a': [0, 0.2, 0.5, 1, 2, 5, 10]
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

for _ in range(10):

    if _ == 5:  # After the first three loops
        top_50_features = [feature[0]
            for feature in feature_counter.most_common(50)]
        selected_features = top_50_features
        # Adjust training and test sets to include only these 50 features
        selected_features = ['ResidueType'] + \
        list(X_train.iloc[:, np.array(top_50_features)].columns)
        selected_features
        list_without_duplicates = list(OrderedDict.fromkeys(selected_features))
        
        X_train = X_train[list_without_duplicates]
        X_test = X_test[list_without_duplicates]

    grid_search = GridSearchCV(
        CustomWeightedRF(), param_grid, cv=3, return_train_score=False)
    grid_search.fit(X_train, y_train)

    print(grid_search.cv_results_['mean_test_score'].shape)

    # Update the a_vs_accuracy dictionary with the mean validation accuracies 
    # for each value of "a"
    for i, a_value in enumerate(param_grid['a']):
        a_vs_accuracy[a_value].append(grid_search.cv_results_[
            'mean_test_score'][i::len(param_grid['a'])].mean())


    # Store mean test scores in the list
    mean_test_scores.append(grid_search.cv_results_['mean_test_score'])

    # Get the best model and its predictions
    current_model = grid_search.best_estimator_
    y_pred = current_model.predict(X_test)  # Use the test data for prediction

    # Calculate the accuracy for the current run
    val_accuracy = accuracy_score(y_test, y_pred)
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
y_pred_best = best_model.predict(X_test)
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

# Plot the 50 most important features over all runs
top_50_features = [feature[0] for feature in most_common_features[:50]]
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set aesthetic styles for the plot
plt.style.use('ggplot')

# Compute the zero-one loss for each sample in the test set
y_pred_best = best_model.predict(X_test)
zero_one_loss = np.where(y_test != y_pred_best, 1, 0)

# Combine the zero-one loss and original features into a new DataFrame
loss_df = pd.DataFrame({'loss': zero_one_loss, 'pointID': X_test.index})
loss_df = pd.merge(
    loss_df, df[['pointID', 'PriorCropT']], on='pointID', how='left')

# Define a color for each Croptype, using colorblind-friendly and harmonious colors
croptype_colors = {'legume': '#5ec962',
                   'canola': '#fde725', 'grain': '#b5de2b'}

# Prepare data for stacked histogram and collect sample counts
croptypes = ['legume', 'canola', 'grain']
data = [loss_df[loss_df['PriorCropT'] == croptype]
        ['loss'].values for croptype in croptypes]
sample_counts = [len(d) for d in data]

# Create labels with sample counts for the legend
labels_with_counts = [f"{croptype} (n = {count})" for croptype, count in zip(
    croptypes, sample_counts)]

# Plot stacked histogram
fig, ax = plt.subplots()
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

        # Only label the bar if its height is greater than 0
        if bar_height > 0:
            # adjust "- 0.5" if needed for better positioning
            text_y_position = bar_bottom + bar_height - 0.5
            ax.text(bar_center_x, text_y_position, int(bar_height),
                    ha='center', va='center', color='black')

# Existing code for customization
plt.xlabel('Zero-One Loss', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks([0, 1], fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Croptype', labels=labels_with_counts,
           title_fontsize='13', fontsize='12')
plt.title('Histogram of Loss Across Each Croptype', fontsize=16)
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

plt.tight_layout()
plt.show()


# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

param_grid = {
    'n_estimators': [20, 30, 40, 50, 100, 300],
    'max_depth': [5, 10, 20, 30, 45],
    'a': [0, 0.2, 0.4, 0.6, 1, 3, 10]
}

# Set up the colormap
cmap = plt.get_cmap('viridis')
norm = mpl.colors.Normalize(vmin=0, vmax=len(param_grid['max_depth']) - 1)

# Create a subplot for each value of 'a' in 2-column layout
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 4 * 4))
axes = axes.ravel()  # Flatten axes to easily iterate

# Remove the last unused ax
fig.delaxes(axes[-1])

# Create custom legend handles
legend_handles = [mpl.patches.Patch(color=cmap(norm(i)), label=f"max_depth={depth}")
                  for i, depth in enumerate(param_grid['max_depth'])]

for ax, a_val in zip(axes, param_grid['a']):
    for i, depth in enumerate(param_grid['max_depth']):
        start_index = (param_grid['a'].index(a_val) * len(param_grid['max_depth']) * len(param_grid['n_estimators'])) + \
                      (i * len(param_grid['n_estimators']))
        end_index = start_index + len(param_grid['n_estimators'])
        scores_for_depth = mean_test_scores[4][start_index:end_index]

        # Get color for this depth from the colormap
        color = cmap(norm(i))

        ax.plot(param_grid['n_estimators'], scores_for_depth,
                marker='o', linestyle='-', color=color)
        ax.set_title(f'a = {a_val}')
        ax.set_xlabel('n_estimators')
        ax.set_ylabel('Validation Accuracy')

# Create a common legend for max_depth using the custom handles
fig.subplots_adjust(right=0.8, hspace=0.4, wspace=0.3)
axes[-1].legend(handles=legend_handles, loc="lower right", title="max_depth")

plt.tight_layout()
plt.show()


# +
# from collections import OrderedDict
# from sklearn.svm import SVC


# selected_features = ['ResidueType'] + \
#     list(X_train.iloc[:, np.array(top_50_features)].columns)
# selected_features
# list_without_duplicates = list(OrderedDict.fromkeys(selected_features))
# list_without_duplicates
# X_train = X_train[list_without_duplicates]
# X_test = X_test[list_without_duplicates]
# print(X_train.shape, y_train.shape)


class CustomWeightedSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', degree=1,
     coef0=0, shrinking=True, a= 1, **kwargs):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.shrinking = shrinking
        self.a = a
        self.svm = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma,
                        degree=self.degree, coef0=self.coef0,
                         shrinking=self.shrinking, probability=True, **kwargs)
    
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
        
        # Now fit the SVC with the computed weights
        self.svm.fit(X, y, sample_weight=sample_weights)
        return self
    
    def predict(self, X, **kwargs):
        return self.svm.predict(X)
    
    def predict_proba(self, X, **kwargs):
        return self.svm.predict_proba(X)

param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter. The strength of the regularization is inversely proportional to C.
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Specifies the kernel type to be used in the algorithm.
    'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    'degree': [2, 3, 4],  # Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
    'coef0': [0.0, 0.5, 1.0],  # Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
    'shrinking': [True, False],  # Whether to use the shrinking heuristic.
    'a': [0, 0.2, 0.5, 1]  # Custom weight parameter for your algorithm.
}


for _ in np.arange(10):

    grid_search = GridSearchCV(CustomWeightedSVM(),
     param_grid, cv=3, return_train_score=False)
    grid_search.fit(X_train, y_train)
    print(grid_search.cv_results_['mean_test_score'].shape)

    # Update the a_vs_accuracy dictionary with the mean validation accuracies 
    # for each value of "a"
    for i, a_value in enumerate(param_grid['a']):
        a_vs_accuracy[a_value].append(grid_search.cv_results_[
            'mean_test_score'][i::len(param_grid['a'])].mean())


    # Store mean test scores in the list
    mean_test_scores.append(grid_search.cv_results_['mean_test_score'])

    # Get the best model and its predictions
    current_model = grid_search.best_estimator_
    y_pred = current_model.predict(X_test)  # Use the test data for prediction

    # Calculate the accuracy for the current run
    val_accuracy = accuracy_score(y_test, y_pred)
    print(_, ":", "Validation Accuracy is ", val_accuracy)
    mean_accuracies.append(val_accuracy)

    # Update the best model if the current model has a higher validation accuracy
    if val_accuracy > best_val_accuracy:
        best_model = current_model
        best_val_accuracy = val_accuracy

# +
import pandas as pd
path_to_data = ("/home/amnnrz/OneDrive - "
                "a.norouzikandelati/Ph.D/Projects/Tillage_Mapping/Data"
                "/field_level_data/FINAL_DATA/")

# X_train.to_csv(path_to_data + "X_train.csv")
# X_test.to_csv(path_to_data + "X_test.csv")
# y_train.to_csv(path_to_data + "y_train.txt")
# y_test.to_csv(path_to_data + "y_test.txt")

X_train = pd.read_csv(path_to_data + "X_train.csv", index_col=0)
X_test = pd.read_csv(path_to_data + "X_test.csv", index_col=0)
y_train = pd.read_csv(path_to_data + "y_train.txt", index_col=0)["ResidueCov"]
y_test = pd.read_csv(path_to_data + "y_test.txt", index_col=0)["ResidueCov"]
# -

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)  # Assuming y_train is a class label, not one-hot encoded
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)  # Assuming y_test is a class label, not one-hot encoded


# +
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer





# Define the PyTorch neural network
class Net(nn.Module):
    def __init__(self, num_input_features, num_units=10, nonlin=nn.ReLU()):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(num_input_features, num_units)  # Adjust the input features accordingly
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, 3)  # 3 classes

    def forward(self, X):
        X = self.nonlin(self.dense1(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense2(X))
        X = self.output(X)
        return X

class CustomNN(BaseEstimator, ClassifierMixin):
    def __init__(self, num_input_features, module=Net, criterion=nn.CrossEntropyLoss, optimizer=optim.Adam, lr=0.01, max_epochs=10, batch_size=32, a=1):
        self.num_input_features = num_input_features
        self.module = module
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.a = a
        self.clf = None

    def fit(self, X, y):
        self.clf = NeuralNetClassifier(
            module=self.module(num_input_features=self.num_input_features),
            criterion=self.criterion,
            optimizer=self.optimizer,
            lr=self.lr,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            optimizer__weight_decay=self.a,  # Example of using the custom parameter 'a' as weight decay
            # Add other relevant parameters and callbacks
        )
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# Define the search space for Bayesian Optimization
search_space = {
    'lr': Real(1e-6, 1e-1, prior='log-uniform'),
    'max_epochs': Integer(5, 50),
    'batch_size': Categorical([16, 32, 64, 128]),
    'module__num_units': Integer(10, 100),
    'a': Real(1e-6, 1e-1, prior='log-uniform')  # Custom weight parameter for your algorithm
}



# Initialize Bayesian optimization
bayes_search = BayesSearchCV(
    estimator=CustomNN(num_input_features=X_train_tensor.shape[1], module=Net),
    search_spaces=search_space,
    n_iter=30,  # Number of iterations
    cv=3,  # 3-fold cross-validation
    n_jobs=1,  # Use all available cores
    return_train_score=False,
    refit=True,
    random_state=42
)

# Perform the search
bayes_search.fit(X_train_tensor, y_train_tensor)

# Best model found
best_model = bayes_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test_tensor)

# Calculate the accuracy
val_accuracy = accuracy_score(y_test, y_pred)
print("Validation Accuracy is ", val_accuracy)

