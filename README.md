# Searching for similarities in financial time series data

The objective of this project is to explore and analyze the effectiveness 
of clustering algorithms in identifying similarities within financial time series data. 
The financial datasets used include stock prices and mutual fund returns, 
with the goal of determining how well different clustering methods and distance metrics can group similar financial instruments together.

Methods

The project utilizes the following clustering algorithms and distance metrics:

Clustering Algorithms:

K-Medoids,
Hierarchical Clustering

Distance Metrics:

Dynamic Time Warping (DTW)
,Kendall's Tau
,Euclidean Distance

Data
The datasets used in this project include:

Stock prices (open, close, adjusted close, volume)

Mutual fund returns

The data was collected and preprocessed to ensure consistency and to handle missing values through linear interpolation and forward/backward filling.

Implementation
The implementation is done in Python, using libraries such as scipy, numpy, pandas, matplotlib, sklearn, and tslearn.

Preprocessing:

Load data from an SQLite database.
Normalize and scale the data.
Calculate returns for mutual funds.


Clustering:

Compute distance matrices using DTW, Kendall's Tau, and Euclidean distance.

Apply K-Medoids and Hierarchical clustering algorithms.

Determine the optimal number of clusters using evaluation metrics.

Evaluation:

The evaluation metrics used was Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index in order to find the optimal number of clusters and evaluate clustering performance .
Clustering stability over time for the different methods was implemented using the Adjusted Rand Index (ARI).
Visualize results using dendrograms and Multidimensional Scaling (MDS) plots.
