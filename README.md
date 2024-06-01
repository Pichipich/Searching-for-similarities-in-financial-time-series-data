# Searching for Similarities in Financial Time Series Data

The objective of this project is to explore and analyze the effectiveness of clustering algorithms in identifying similarities within financial time series data. The financial datasets used include stock prices and mutual fund returns, with the goal of determining how well different clustering methods and distance metrics can group similar financial instruments together.

**Methods**

The project utilizes the following clustering algorithms and distance metrics:

+ **Clustering Algorithms:**
  - K-Medoids
  - Hierarchical Clustering

+ **Distance Metrics:**
  - Dynamic Time Warping (DTW)
  - Kendall's Tau
  - Euclidean Distance

**Data**

The data used in this project include three different length datasets for each financial instrument in order to evaluate scalability:

+ Stock prices (open, close, adjusted close, volume)
+ Mutual fund returns

The data was collected and preprocessed to ensure consistency and to handle missing values through linear interpolation and forward/backward filling.

**Implementation**

The implementation is done in Python, using libraries such as scipy, numpy, pandas, matplotlib, sklearn, and tslearn.

**Preprocessing:**
+ Load data from an SQLite database.
+ Normalize and scale the data.
+ Calculate returns for mutual funds.

**Clustering:**
1. Compute distance matrices using DTW, Kendall's Tau, and Euclidean distance.
2. Apply K-Medoids and Hierarchical clustering algorithms.
3. Determine the optimal number of clusters using evaluation metrics and the Elbow method.
4. Visualize clustering results using dendrograms and Multidimensional Scaling (MDS) plots.

**Evaluation:**
The evaluation metrics used include Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index to find the optimal number of clusters and evaluate clustering performance.

Clustering stability over time for the different methods was assessed using the Adjusted Rand Index (ARI).

**Similarity Search**

In addition to clustering based on financial performance, this project also explores the similarity search within financial time series data clusters based on sector and company affiliations.

1. **Sector-Based Clustering:** For stocks, we examine whether clustering results align with the sector each stock belongs to. This involves mapping each stock to its corresponding sector and evaluating cluster purity based on sector homogeneity.

2. **Company-Based Clustering:** For mutual funds, we examine whether funds that tend to be clustered together are managed by the same company. Similarly to stocks, we map each mutual fund to its managing company and assess cluster purity based on company affiliations.

3. **Intra-Cluster Metrics:**
  + **Correlation:** We calculate the average correlation of returns within each cluster to understand intra-cluster similarity.
  + **Volatility:** We measure the average volatility within clusters to identify the risk profiles of clustered assets.
  + **Trends:** We analyze the average trends within clusters using linear regression to determine common movement patterns.

**Visualization:**
- Dendrograms: Used to visualize the hierarchical clustering process and help determine the optimal number of clusters.
- Elbow Method: Used to identify the point where the rate of decrease in the distortion slows down, indicating the optimal number of clusters.
- MDS Plots: Used to visualize the distance matrix in a lower-dimensional space to better understand the clustering results.

This README file provides a comprehensive overview of the project, its methods, implementation, and usage instructions. It serves as a guide for anyone interested in replicating or extending the analysis.
