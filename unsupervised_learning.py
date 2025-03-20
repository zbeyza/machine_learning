###############################################################
# UNSUPERVISED LEARNING
###############################################################

# Imports and Display Settings

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)



#######################################################
# K-Means Clustering
#######################################################

"""
USArrests.csv

Clustering U.S. states based on crime statistics.

The goal is to group U.S. states into 4-5 clusters based on certain values.
"""

df = pd.read_csv("datasets/USArrests.csv", index_col=0)

df.head()
df.isnull().sum()
df.info()

df.describe().T

# Standardizing values
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

# K-Means model
kmeans = KMeans(n_clusters=4, random_state=17).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

# SSE (Sum of Squared Errors)
kmeans.inertia_

################################
# Determining the Optimal Number of Clusters
################################

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Different K Values vs. SSE")
plt.title("Elbow Method for Optimal Clusters")
plt.show()

# Using Yellowbrick for an optimized approach
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()
elbow.elbow_value_

#######################################
# Creating Final Clusters
#######################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

clusters_kmeans = kmeans.labels_

df = pd.read_csv("datasets/USArrests.csv", index_col=0)
df["cluster"] = clusters_kmeans
df["cluster"] = df["cluster"] + 1

df.groupby("cluster").agg(["count", "mean", "median"])

df.to_csv("clusters.csv")



#######################################################
# Hierarchical Clustering
#######################################################

df = pd.read_csv("datasets/USArrests.csv", index_col=0)

# Standardizing the dataset
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

hc_average = linkage(df, "average")

plt.figure(figsize=(10, 5))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Observations")
plt.ylabel("Distances")
dendrogram(hc_average, leaf_font_size=10)
plt.show()

# Truncated dendrogram
plt.figure(figsize=(10, 5))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Observations")
plt.ylabel("Distances")
dendrogram(hc_average, truncate_mode="lastp", p=10, show_contracted=True, leaf_font_size=10)
plt.show()

#########################################
# Determining the Number of Clusters
#########################################

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.6, color="b", linestyle="--")
plt.axhline(y=0.5, color="r", linestyle="--")
plt.show()

###########################################
# Creating the Final Model
###########################################

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")

clusters = cluster.fit_predict(df)

df = pd.read_csv("datasets/USArrests.csv", index_col=0)
df["hi_cluster_no"] = clusters
df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df["kmeans_cluster_no"] = clusters_kmeans
df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1



#######################################################
# Principal Component Analysis (PCA)
#######################################################

df = pd.read_csv("datasets/Hitters.csv")

"""
Hitters Dataset:
Contains baseball players' salaries and various features.
The goal is to predict salary, but here we focus on dimensionality reduction.
"""

# Removing Salary and categorical variables
num_cols = [col for col in df.columns if df[col].dtype != "O" and "Salary" not in col]
df = df[num_cols]

# Removing missing values
df.dropna(inplace=True)

# Standardization
df = StandardScaler().fit_transform(df)

# PCA Model
pca = PCA()
pca_fit = pca.fit_transform(df)

# Explained Variance
pca.explained_variance_ratio_

# Cumulative Variance
np.cumsum(pca.explained_variance_ratio_)

#################################
# Choosing Optimal Components
#################################

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Ratio")
plt.show()

# Choosing 3 components

##############################
# Creating the Final PCA Model
##############################

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)



#######################################################
# BONUS: Principal Component Regression (PCR)
#######################################################

df = pd.read_csv("datasets/Hitters.csv")

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
others = [col for col in df.columns if col not in num_cols]

final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1", "PC2", "PC3"]), df[others]], axis=1)

# Encoding categorical variables
def label_encoder(dataframe, binary_cols):
    labelencoder = LabelEncoder()
    dataframe[binary_cols] = labelencoder.fit_transform(dataframe[binary_cols])
    return dataframe

for col in ["NewLeague", "Division", "League"]:
    label_encoder(final_df, col)

final_df.dropna(inplace=True)

y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))



#######################################################
# BONUS: PCA-Based Visualization of High-Dimensional Data
#######################################################

#####################
# Breast Cancer Dataset
#####################

df = pd.read_csv("datasets/breast_cancer.csv")

y = df["diagnosis"]
X = df.drop(["diagnosis", "id", "Unnamed: 32"], axis=1)

def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df

pca_df = create_pca_df(X, y)

def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

plot_pca(pca_df, "diagnosis")

