# ---
# jupyter:
#   jupytext:
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
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# +
# # # Read data
# path_to_data = ("/Users/aminnorouzi/Library/CloudStorage/"
#                 "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
#                 "Projects/Tillage_Mapping/Data/field_level_data/FINAL_DATA/")

path_to_data = ("/home/amnnrz/OneDrive - "
                "a.norouzikandelati/Ph.D/Projects/Tillage_Mapping/Data"
                "/field_level_data/FINAL_DATA/")

df = pd.read_csv(path_to_data + "metric_finalData.csv", index_col=0)
df = df.dropna(subset=["Tillage", "ResidueType", "ResidueCov"])
print(df['ResidueType'].value_counts(), df['ResidueCov'].value_counts())
le_resCov = LabelEncoder()
le_resType = LabelEncoder()
le_tillage = LabelEncoder()
df['ResidueCov'] = le_resCov.fit_transform(df['ResidueCov'])
df['ResidueType'] = le_resType.fit_transform(df['ResidueType'])
df['Tillage'] = le_tillage.fit_transform(df['Tillage'])
print(df['ResidueType'].value_counts(), df['ResidueCov'].value_counts())
df = df.set_index('pointID')
df = df.iloc[:, [2, 4, 5] + list(range(7, df.shape[1]))]
# Impute missing values with the median
df = df.fillna(df.median())
doyCols = df.filter(like="doy").columns
df = df.drop(columns=doyCols)
df


# +
mainBands_df = df.iloc[:, 0:303]

mainBands_df_0 = mainBands_df.loc[mainBands_df['Tillage'] == 0]
mainBands_df_1 = mainBands_df.loc[mainBands_df['Tillage'] == 1]
mainBands_df_2 = mainBands_df.loc[mainBands_df['Tillage'] == 2]

data = mainBands_df_0.iloc[:, 3:]

# Assume 'data' is your dataset
median = np.median(data)
abs_deviation = np.abs(data - median)
mad = np.median(abs_deviation)

# To make MAD comparable to the standard deviation for a normal distribution
# mad_scaled = mad * 1.4826

# Set a threshold (commonly 2.5 or 3) for identifying outliers
threshold = 3

# Identify outliers
outliers = abs_deviation > threshold
# Filter outliers
data_filtered = data[~outliers]
data_filtered.shape, data.shape

# +
from sklearn.preprocessing import StandardScaler

X = df.iloc[:, [0] + list(range(3, df.shape[1]))]
Xinfo = X.describe()
print(Xinfo.loc['min'].min())
print(Xinfo.loc['max'].max())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.iloc[:,2:])
X_scaled = pd.DataFrame(X_scaled, columns=X.iloc[:,2:].columns,
                         index=list(X.index))
X = pd.concat([X.iloc[:, 0:2], X_scaled], axis=1)

Xinfo = X.describe()
print(Xinfo.loc['min'].min())
print(Xinfo.loc['max'].max())

y = df['ResidueCov']


# -

X

# +
from sklearn.decomposition import PCA

# Create principal components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

X_pca.head()
# -

loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=X.columns,  # and the rows are the original features
)
loadings


# +
def plot_variance(pca):
    # The amount of variance that each principal component explains
    var = pca.explained_variance_ratio_[:10]
    
    # Cumulative variance explained
    cum_var = np.cumsum(var)
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(var) + 1), var, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(1, len(cum_var) + 1), cum_var, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

# Look at explained variance
plot_variance(pca)
# -

mi_scores = mutual_info_classif(X_pca, y)
mi_scores

# +
# Calculate mutual information between each feature and the target
mi_scores = mutual_info_classif(X, y, discrete_features='auto')
mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores.head(20))



# +
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Generate synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'Numerical Feature': np.random.randn(300),
    'Target': np.random.choice(['Class 1', 'Class 2', 'Class 3'], 300)
})

# Box Plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Target', y='Numerical Feature', data=data)
plt.title('Box Plot of Numerical Feature by Target Category')
plt.show()

# Violin Plot
plt.figure(figsize=(8, 6))
sns.violinplot(x='Target', y='Numerical Feature', data=data)
plt.title('Violin Plot of Numerical Feature by Target Category')
plt.show()

# Swarm Plot
plt.figure(figsize=(8, 6))
sns.swarmplot(x='Target', y='Numerical Feature', data=data)
plt.title('Swarm Plot of Numerical Feature by Target Category')
plt.show()

# Facet Grid with Scatter Plot (using another numerical feature)
data['Numerical Feature 2'] = np.random.randn(300) + data['Target'].map({'Class 1': 0, 'Class 2': 2, 'Class 3': 4})
grid = sns.FacetGrid(data, hue="Target", height=6)
grid.map(plt.scatter, 'Numerical Feature', 'Numerical Feature 2').add_legend()
plt.title('Scatter Plot of Two Numerical Features by Target Category')
plt.show()

