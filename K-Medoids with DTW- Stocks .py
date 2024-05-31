
from tslearn.metrics import dtw as ts_dtw
from kneed import KneeLocator
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import pandas as pd
import time
from sklearn.manifold import MDS
from scipy.stats import linregress, zscore
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
import seaborn as sns

symbols = ['AMAT', 'JPM', 'NVDA', 'GE', 'HD', 'INTC', 'QCOM', 'HON', 'WFC',
           'AAPL', 'MSFT', 'TXN', 'GOOG', 'USB', 'GOOGL', 'GILD', 'AMZN', 'PEP',
           'LLY', 'MNST', 'CDNS', 'META', 'MU', 'TSLA', 'HSBC',
           'T', 'CVS', 'BA', 'UNH', 'TSM', 'V', 'NVO', 'SHEL', 'BBY', 'WMT', 'XOM',
           'MA', 'JNJ', 'AVGO', 'PG', 'ORCL', 'BHP', 'AZN', 'ADBE', 'ASML', 'CVX',
           'MRK', 'COST', 'TM', 'ABBV', 'KO', 'NGG', 'CRM', 'BAC', 'LEN', 'ACN', 'MCD',
           'NVS', 'BIIB', 'NFLX', 'LIN', 'SAP', 'CSCO', 'AMD', 'TMO', 'PDD', 'BABA',
           'ABT', 'TMUS', 'NKE', 'TTE', 'TBC', 'CMCSA', 'DIS', 'PFE', 'DHR', 'VZ', 'TBB',
           'INTU', 'PHM', 'LYG', 'IBM', 'AMGN', 'PM', 'UNP', 'NOW', 'RYAAY', 'COP',
           'SPGI', 'TFC', 'MS', 'UPS', 'CAT', 'RY', 'AXP', 'UL', 'NEE', 'RTX',
           'LOW', 'SNY']


conn = sqlite3.connect('small_stock_dataset.db')
dfs = {}  # Filtered data
unscaled_dfs = {}  # Raw data
outliers = []
zscore_threshold = 10

for symbol in symbols:
    query = f"SELECT Date, Open, Close, `Adj Close`, Volume FROM `{symbol}` ORDER BY Date"
    df = pd.read_sql(query, conn)
    df.set_index('Date', inplace=True)
    df.interpolate(method='linear', inplace=True)
    normalized_df = df.apply(zscore)

    if (normalized_df.abs() > zscore_threshold).any().any():
        outliers.append(symbol)
    else:
        dfs[symbol] = normalized_df
        unscaled_dfs[symbol] = df
    # Remove outliers from the symbols list
    symbols = [symbol for symbol in symbols if symbol not in outliers]
conn.close()


# Prepare for DTW distance calculation
processed_symbols = list(dfs.keys())  # Use only symbols without outliers
num_processed_symbols = len(processed_symbols)
dtw_distance_matrix = np.zeros((num_processed_symbols, num_processed_symbols))

# Calculate DTW distances
for i in range(num_processed_symbols):
    for j in range(i + 1, num_processed_symbols):
        symbol_i = processed_symbols[i]
        symbol_j = processed_symbols[j]
        time_series_data_i = dfs[symbol_i][['Open', 'Close', 'Adj Close', 'Volume']].values
        time_series_data_j = dfs[symbol_j][['Open', 'Close', 'Adj Close', 'Volume']].values
        distance = ts_dtw(time_series_data_i, time_series_data_j)
        dtw_distance_matrix[i, j] = dtw_distance_matrix[j, i] = distance

# Perform MDS
mds = MDS(n_components=4, random_state=42, dissimilarity='precomputed')
features = mds.fit_transform(dtw_distance_matrix)

# Cluster analysis
inertia_values = []
possible_clusters = range(1, 10)
for k in possible_clusters:
    kmedoids = KMedoids(n_clusters=k, metric="precomputed", random_state=42)
    kmedoids.fit(dtw_distance_matrix)
    inertia_values.append(kmedoids.inertia_)

# Find the optimal number of clusters
kneedle = KneeLocator(possible_clusters, inertia_values, curve='convex', direction='decreasing')
optimal_clusters = kneedle.knee
print(f"Optimal Number of Clusters (Knee Point): {optimal_clusters}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(possible_clusters, inertia_values, linestyle='-', color='b')
plt.title('Elbow Method', fontsize=20)
plt.xlabel('Number of Clusters (k)', fontsize=18)
plt.ylabel('Inertia', fontsize=18)
plt.show()


# perform MDS on the distance matrix
mds = MDS(n_components=4, random_state=42, dissimilarity='precomputed')
reduced_data_dms = mds.fit_transform(dtw_distance_matrix)

reduced_data_with_clusters_dms = pd.DataFrame(reduced_data_dms, columns=['MDS1', 'MDS2', 'MDS3', 'MDS4'])

#scores before outlier detection
silhouette_scores_original = []
calinski_harabasz_scores_original = []
davies_bouldin_scores_original = []

for k in range(2, 10):
    kmedoids_dtw = KMedoids(n_clusters=k, random_state=42, metric='precomputed')
    labels_original = kmedoids_dtw.fit_predict(dtw_distance_matrix)

    silhouette_scores_original.append(silhouette_score(dtw_distance_matrix, labels_original, metric='precomputed'))
    calinski_harabasz_scores_original.append(calinski_harabasz_score(reduced_data_dms, labels_original))
    davies_bouldin_scores_original.append(davies_bouldin_score(reduced_data_dms, labels_original))

silhouette_scores_normalized_original = [score / max(silhouette_scores_original) for score in silhouette_scores_original]
calinski_harabasz_scores_normalized_original = [score / max(calinski_harabasz_scores_original) for score in calinski_harabasz_scores_original]
davies_bouldin_scores_normalized_original = [1 / (score + 1e-8) for score in davies_bouldin_scores_original]
davies_bouldin_scores_normalized_original = [score / max(davies_bouldin_scores_normalized_original) for score in davies_bouldin_scores_normalized_original]

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.plot(range(2, 10), silhouette_scores_normalized_original, label='Silhouette Score - Original', color='red')
plt.plot(range(2, 10), calinski_harabasz_scores_normalized_original, label='Calinski-Harabasz Index - Original', color='blue')
plt.plot(range(2, 10), davies_bouldin_scores_normalized_original, label='Inverted Davies-Bouldin Index - Original', color='green')

plt.legend()
plt.title('Clustering Evaluation Scores', fontsize=22)
plt.xlabel('Number of Clusters (k)', fontsize=18)
plt.ylabel('Normalized Scores', fontsize=18)
plt.show()


#EVALUATION SCORES
# Chosen number of clusters after evaluation
chosen_k =3

# Assume euclidean_distance_matrix is your precomputed distance matrix
kmedoids = KMedoids(n_clusters=chosen_k, metric='precomputed', random_state=42)
kmedoids.fit(dtw_distance_matrix)
clusters = kmedoids.labels_  # This will give you the cluster labels

silhouette_avg = silhouette_score(dtw_distance_matrix, clusters, metric='precomputed')
calinski_harabasz_avg = calinski_harabasz_score(reduced_data_dms, clusters)  # Ensure reduced_data_dms is correctly defined
davies_bouldin_avg = davies_bouldin_score(reduced_data_dms, clusters)

# Print out the evaluation scores for the chosen k
print(f"Silhouette Score: {silhouette_avg}")
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

# Assume silhouette_scores_original, calinski_harabasz_scores_original, and davies_bouldin_scores_original contain the scores from 2 to 40 clusters as calculated previously

# Normalize scores for the chosen k
def normalize_score(score, scores_list):
    return score / max(scores_list) if max(scores_list) > 0 else 0

chosen_k_index = chosen_k - 2  # Adjusting index since range starts from 2
normalized_silhouette = normalize_score(silhouette_avg, silhouette_scores_original)
normalized_calinski = normalize_score(calinski_harabasz_avg, calinski_harabasz_scores_original)
normalized_davies_bouldin = 1 / (davies_bouldin_avg + 1e-8)  # Inverting Davies-Bouldin
normalized_davies_bouldin = normalize_score(normalized_davies_bouldin, [1 / (x + 1e-8) for x in davies_bouldin_scores_original])

print(f"Normalized Silhouette Score: {normalized_silhouette}")
print(f"Normalized Calinski-Harabasz Score: {normalized_calinski}")
print(f"Normalized (Inverted) Davies-Bouldin Score: {normalized_davies_bouldin}")

# Visualization for evaluation
plt.figure(figsize=(10, 6))
plt.bar(['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin'],
        [normalized_silhouette, normalized_calinski, normalized_davies_bouldin],
        color=['red', 'blue', 'green'])
plt.ylabel('Normalized Metric Score')
plt.title('Normalized Clustering Evaluation Metrics')
plt.show()


clusters_df = pd.DataFrame({'Symbol': symbols, 'Cluster': clusters})
reduced_data_with_clusters_dms['Cluster'] = clusters


reduced_data_with_clusters_dms['Cluster'] = clusters
from sklearn.manifold import MDS

mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
mds_result = mds.fit_transform(dtw_distance_matrix)

plt.scatter(mds_result[:, 0], mds_result[:, 1], c=clusters, cmap='viridis')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.title('MDS Clustering Results [K-Medoids with Euclidean Distances]')
plt.show()



#plot MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_result = mds.fit_transform(dtw_distance_matrix)

colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
    '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'
]

# Create a colormap from the list of colors
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(colors[:chosen_k])

plt.figure(figsize=(8, 6))  # Use equal width and height for a square plot
scatter = plt.scatter(mds_result[:, 0], mds_result[:, 1], c=clusters, cmap=custom_cmap, edgecolor='none', s=50)
plt.xlabel('MDS Dimension 1', fontsize=14)
plt.ylabel('MDS Dimension 2', fontsize=14)
plt.title('MDS Clustering Results [K-Medoids with DTW]', fontsize=16)
plt.grid(True)
plt.axis('equal')  # Set equal scaling by changing axis limits to equal ranges
plt.show()





import seaborn as sns

# Distance matrix and clustering
processed_symbols = list(dfs.keys())
stock_data = np.array([dfs[symbol].values.flatten() for symbol in processed_symbols])
num_clusters = 4  # Set based on earlier analysis
window_size = 104  # half-yearly data
step_size = 52   # overlap of half a year

def compute_distance_matrix(data):
    num_samples = data.shape[0]
    distance_matrix = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            dist = ts_dtw(data[i], data[j])
            distance_matrix[i, j] = distance_matrix[j, i] = dist
    return distance_matrix

def temporal_cluster_validation(stock_data, window_size, step_size, num_clusters):
    num_windows = (stock_data.shape[1] - window_size) // step_size + 1
    cluster_labels_over_time = []

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window_data = stock_data[:, start_idx:end_idx]

        flattened_data = window_data.reshape(window_data.shape[0], -1)
        window_distance_matrix = compute_distance_matrix(flattened_data)

        kmedoids = KMedoids(n_clusters=num_clusters, metric='precomputed', random_state=42)
        cluster_labels = kmedoids.fit_predict(window_distance_matrix)
        cluster_labels_over_time.append(cluster_labels)

    return np.array(cluster_labels_over_time).T  # Transpose to make rows correspond to stocks

# Visualize the temporal stability of clusters
cluster_labels_matrix = temporal_cluster_validation(stock_data, window_size, step_size, num_clusters)
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_labels_matrix, annot=False, cmap='viridis', cbar_kws={'label': 'Cluster Label'})
plt.title('Cluster Assignments Over Time Windows')
plt.xlabel('Time Window Index')
plt.ylabel('Stock Index')
plt.show()

from sklearn.metrics import adjusted_rand_score

def compute_temporal_ari(cluster_labels_matrix):
    num_windows = cluster_labels_matrix.shape[1]
    ari_scores = []

    for i in range(num_windows - 1):
        current_labels = cluster_labels_matrix[:, i]
        next_labels = cluster_labels_matrix[:, i + 1]
        ari = adjusted_rand_score(current_labels, next_labels)
        ari_scores.append(ari)

    return ari_scores

# Compute Adjusted Rand Index over time
ari_scores = compute_temporal_ari(cluster_labels_matrix)

# Plot Adjusted Rand Index over time
plt.figure(figsize=(10, 6))
plt.plot(range(len(ari_scores)), ari_scores, linestyle='-')
plt.title('Adjusted Rand Index Over Time',fontsize=22)
plt.xlabel('Time Window Index')
plt.ylabel('Adjusted Rand Index')
plt.grid(True)
plt.show()




from sklearn.metrics import jaccard_score

def compute_temporal_jaccard(cluster_labels_matrix):
    num_windows = cluster_labels_matrix.shape[1]
    jaccard_scores = []

    for i in range(num_windows - 1):
        current_labels = cluster_labels_matrix[:, i]
        next_labels = cluster_labels_matrix[:, i + 1]
        jaccard = jaccard_score(current_labels, next_labels, average='macro')  # or 'weighted', 'micro'
        jaccard_scores.append(jaccard)

    return jaccard_scores

# Compute Jaccard Index over time
jaccard_scores = compute_temporal_jaccard(cluster_labels_matrix)

# Plot Jaccard Index over time
plt.figure(figsize=(10, 6))
plt.plot(range(len(jaccard_scores)), jaccard_scores, linestyle='-')
plt.title('Jaccard Index Over Time',fontsize=22)
plt.xlabel('Time Window Index')
plt.ylabel('Jaccard Index')
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming 'cluster_labels_matrix' holds the cluster labels for each stock over time.
num_windows = cluster_labels_matrix.shape[1]
window_labels = [f'Time Window {i+1}' for i in range(num_windows)]

# Set a manageable figure size
plt.figure(figsize=(12, 8))  # Standard size, modify if needed
ax = sns.heatmap(cluster_labels_matrix, annot=False, cmap='Blues', cbar_kws={'label': 'Cluster Number'})
ax.set_title('Cluster Evolution Over Time [K-Medoids - DTW]', fontsize=20)
ax.set_xlabel('Time Interval', fontsize=18)
ax.set_ylabel('Stock Index', fontsize=18)
ax.set_xticklabels(window_labels, rotation=45, ha="right")  # Rotate for better visibility
plt.tight_layout()  # Adjust layout to make room for labels if necessary

plt.show()



def temporal_cluster_validation(stock_data, window_size, step_size, num_clusters):
    num_windows = (stock_data.shape[1] - window_size) // step_size + 1
    cluster_labels_over_time = []
    average_ari_scores = []

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window_data = stock_data[:, start_idx:end_idx]

        flattened_data = window_data.reshape(window_data.shape[0], -1)
        window_distance_matrix = compute_distance_matrix(flattened_data)

        kmedoids = KMedoids(n_clusters=num_clusters, metric='precomputed', random_state=42)
        cluster_labels = kmedoids.fit_predict(window_distance_matrix)
        cluster_labels_over_time.append(cluster_labels)

        # Calculate ARI score for this time window
        if i > 0:  # Skip the first window as there is no previous window to compare with
            current_labels = cluster_labels_over_time[-1]
            prev_labels = cluster_labels_over_time[-2]
            ari = adjusted_rand_score(prev_labels, current_labels)
            average_ari_scores.append(ari)

    return np.array(cluster_labels_over_time).T, average_ari_scores


# Usage
cluster_labels_matrix, average_ari_scores = temporal_cluster_validation(stock_data, window_size, step_size,
                                                                        num_clusters)

# Calculate the mean of the ARI scores
mean_ari = np.mean(average_ari_scores) if average_ari_scores else float('nan')  # Safeguard in case the list is empty

# Print or visualize the average ARI score
print(f"Average ARI across all time windows: {mean_ari}")





import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'cluster_labels_matrix' holds the cluster labels for each stock over time.

plt.figure(figsize=(12, 8))
sns.heatmap(cluster_labels_matrix, annot=False, cmap='Blues', cbar_kws={'label': 'Cluster Label'})
plt.title('Cluster Assignments Over Time Windows', fontsize=20, y=1.03)  # Adjust title size and position
plt.xlabel('Time Window Index', fontsize=16)  # Adjust x-axis label size
plt.ylabel('Cluster Number', fontsize=16)  # Adjust y-axis label size
plt.xticks(fontsize=12)  # Increase font size of x-axis tick labels
plt.yticks(fontsize=12)  # Increase font size of y-axis tick labels
plt.tight_layout()  # Adjust layout to make room for labels if necessary
plt.show()




import matplotlib.pyplot as plt
import seaborn as sns

num_windows = cluster_labels_matrix.shape[1]  # Number of time windows

fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Adjust the figsize to fit the aspect ratio and size you desire

sns.heatmap(cluster_labels_matrix, annot=False, cmap='Blues', cbar_kws={'label': 'Cluster Number'}, ax=ax[0])
ax[0].set_title('Cluster Assignments Over Time Windows')
ax[0].set_xlabel('Time Window Index')
ax[0].set_ylabel('Stock Index')

ax[1].plot(range(1, len(ari_scores) + 1), ari_scores, linestyle='-', marker='o')
ax[1].set_title('Adjusted Rand Index Over Time')
ax[1].set_xlabel('Time Window Index')
ax[1].set_ylabel('Adjusted Rand Index')
ax[1].grid(True)

plt.tight_layout()
plt.savefig("plots.png", dpi=300)  # High-resolution save for better quality in documents
plt.show()






import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import fcluster


mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_result = mds.fit_transform(dtw_distance_matrix)

colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
    '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'
]

# Create a colormap from the list of colors
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(colors[:chosen_k])

plt.figure(figsize=(8, 6))  # Use equal width and height for a square plot
scatter = plt.scatter(mds_result[:, 0], mds_result[:, 1], c=clusters, cmap=custom_cmap, edgecolor='none', s=50)
plt.xlabel('MDS Dimension 1', fontsize=14)
plt.ylabel('MDS Dimension 2', fontsize=14)
plt.title('MDS Clustering Results [K-Medoids with DTW]', fontsize=16)
plt.grid(True)
plt.axis('equal')  # Set equal scaling by changing axis limits to equal ranges
plt.show()



for cluster_num in sorted(clusters_df['Cluster'].unique()):
    cluster_stocks = clusters_df[clusters_df['Cluster'] == cluster_num]['Symbol'].tolist()
    print(f"Cluster {cluster_num}: {', '.join(cluster_stocks)}")


#OUTLIERS

min_cluster_size = 2

clusters_df = pd.DataFrame({'Cluster': clusters})
cluster_sizes = clusters_df['Cluster'].value_counts()
outlier_clusters = cluster_sizes[cluster_sizes < min_cluster_size].index


outlier_stocks = []
for cluster in outlier_clusters:
    stocks_in_cluster = clusters_df[clusters_df['Cluster'] == cluster]['Symbol'].tolist()
    outlier_stocks.extend(stocks_in_cluster)

outlier_clusters = cluster_sizes[cluster_sizes < min_cluster_size].index.to_list()
outlier_indices = clusters_df[clusters_df['Cluster'].isin(outlier_clusters)].index
filtered_features = np.delete(features, outlier_indices, axis=0)

clusters_df = pd.DataFrame({'Symbol': symbols, 'Cluster': clusters})
cluster_sizes = clusters_df['Cluster'].value_counts()
outlier_clusters = cluster_sizes[cluster_sizes < min_cluster_size].index.to_list()

outlier_symbols = clusters_df[clusters_df['Cluster'].isin(outlier_clusters)]['Symbol'].tolist()
outlier_indices = [symbols.index(symbol) for symbol in outlier_symbols if symbol in symbols]
filtered_distance_matrix = np.delete(np.delete(dtw_distance_matrix, outlier_indices, axis=0), outlier_indices, axis=1)

filtered_kmedoids = KMedoids(n_clusters=chosen_k, random_state=42, metric="precomputed")
filtered_clusters = filtered_kmedoids.fit_predict(filtered_distance_matrix)




sector_mapping = {
    "AMAT": "Technology", "JPM": "Finance", "NVDA": "Technology", "HD": "Consumer Discretionary",
    "INTC": "Technology", "QCOM": "Technology", "HON": "Industrials", "WFC": "Finance",
    "AAPL": "Technology", "MSFT": "Technology", "TXN": "Technology", "GOOG": "Technology",
    "USB": "Finance", "GOOGL": "Technology", "GILD": "Health Care", "AMZN": "Consumer Discretionary",
    "PEP": "Consumer Staples", "LLY": "Health Care", "MNST": "Consumer Staples", "CDNS": "Technology",
    "META": "Technology", "MU": "Technology", "TSLA": "Consumer Discretionary", "HSBC": "Finance",
    "T": "Telecommunications", "CVS": "Consumer Staples", "BA": "Industrials", "UNH": "Health Care",
    "TSM": "Technology", "V": "Consumer Discretionary", "NVO": "Health Care", "SHEL": "Energy",
    "BBY": "Consumer Discretionary", "WMT": "Consumer Discretionary", "XOM": "Energy",
    "MA": "Consumer Discretionary", "JNJ": "Health Care", "AVGO": "Technology", "PG": "Consumer Discretionary",
    "ORCL": "Technology", "BHP": "Basic Materials", "AZN": "Health Care", "ADBE": "Technology",
    "ASML": "Technology", "CVX": "Energy", "MRK": "Health Care", "COST": "Consumer Discretionary",
    "TM": "Consumer Discretionary", "ABBV": "Health Care", "KO": "Consumer Staples",
    "NGG": "Utilities", "CRM": "Technology", "BAC": "Finance", "LEN": "Consumer Discretionary",
    "ACN": "Technology", "MCD": "Consumer Discretionary", "NVS": "Health Care", "BIIB": "Health Care",
    "NFLX": "Consumer Discretionary", "LIN": "Industrials", "SAP": "Technology", "CSCO": "Technology",
    "AMD": "Technology", "TMO": "Health Care", "PDD": "Consumer Discretionary", "BABA": "Consumer Discretionary",
    "ABT": "Health Care", "TMUS": "Telecommunications", "NKE": "Consumer Discretionary",
    "TTE": "Energy", "TBC": "Telecommunications", "CMCSA": "Consumer Discretionary",
    "DIS": "Consumer Discretionary", "PFE": "Health Care", "DHR": "Health Care", "VZ": "Telecommunications",
    "TBB": "Telecommunications", "INTU": "Technology", "PHM": "Consumer Discretionary",
    "LYG": "Finance", "IBM": "Technology", "AMGN": "Health Care", "PM": "Consumer Staples",
    "UNP": "Industrials", "NOW": "Technology", "RYAAY": "Consumer Discretionary", "COP": "Energy",
    "SPGI": "Finance", "TFC": "Finance", "MS": "Finance", "UPS": "Industrials", "CAT": "Industrials",
    "RY": "Finance", "AXP": "Finance", "UL": "Consumer Staples", "NEE": "Utilities", "UBER": "Consumer Discretionary",
    "RTX": "Industrials", "LOW": "Consumer Discretionary", "SNY": "Health Care"
}


filtered_sector_mapping = {
    symbol: sector for symbol, sector in sector_mapping.items() if symbol in symbols
}

sector_df = pd.DataFrame(list(sector_mapping.items()), columns=['Symbol', 'Sector'])
merged_df = pd.merge(clusters_df, sector_df, on='Symbol')
sector_composition = merged_df.groupby(['Cluster', 'Sector']).size().unstack(fill_value=0)


sector_totals = sector_df['Sector'].value_counts()
cluster_sector_counts = merged_df.groupby(['Cluster', 'Sector']).size().unstack(fill_value=0)
cluster_sector_percentages = cluster_sector_counts.div(cluster_sector_counts.sum(axis=1), axis=0) * 100
print(cluster_sector_percentages)


#sector mapping
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(cluster_sector_percentages, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('DTW with K-Medoids',fontsize=20)
plt.xlabel('Sector',fontsize=18)
plt.ylabel('Cluster',fontsize=18)
plt.xticks(rotation=55, fontsize=14)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

sector_composition.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
plt.title('Sector Composition within Clusters')
plt.xlabel('Cluster')
plt.ylabel('Number of Stocks')
plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


filtered_symbols = [symbol for symbol in symbols if symbol not in outlier_stocks]
filtered_clusters_df = clusters_df[clusters_df['Symbol'].isin(filtered_symbols)].copy()

from scipy.stats import linregress
for symbol, df in dfs.items():
    df['Return'] = df['Adj Close'].pct_change()

intra_cluster_corr_filtered = {}
for cluster_num in filtered_clusters_df['Cluster'].unique():
    cluster_symbols = filtered_clusters_df[filtered_clusters_df['Cluster'] == cluster_num]['Symbol'].tolist()
    returns_df = pd.concat([dfs[symbol]['Return'] for symbol in cluster_symbols if symbol in dfs], axis=1)
    corr_matrix = returns_df.corr(method='pearson')
    mean_corr = corr_matrix.mean().mean()
    intra_cluster_corr_filtered[cluster_num] = mean_corr


#cluster trends and volality through unscaled data


cluster_volatility_filtered = {}
for cluster_num, symbols in filtered_clusters_df.groupby('Cluster')['Symbol']:
    cluster_returns = pd.concat([unscaled_dfs[symbol]['Adj Close'].pct_change() for symbol in symbols], axis=1)
    cluster_volatility_filtered[cluster_num] = cluster_returns.std().mean()

cluster_trends_filtered = {}
for cluster_num, symbols in filtered_clusters_df.groupby('Cluster')['Symbol']:
    trend_slopes = []
    for symbol in symbols:
        df = unscaled_dfs[symbol]
        df['Time'] = np.arange(len(df))
        slope, _, _, _, _ = linregress(df['Time'], df['Adj Close'])
        trend_slopes.append(slope)
    cluster_trends_filtered[cluster_num] = np.mean(trend_slopes)


# Print or analyze the recalculated metrics
print('Intra-cluster Correlation (Filtered):', intra_cluster_corr_filtered)
print('Cluster Volatility (Filtered):', cluster_volatility_filtered)
print('Cluster Trends (Filtered):', cluster_trends_filtered)


average_volume_per_stock = {symbol: df['Volume'].mean() for symbol, df in unscaled_dfs.items()}

cluster_volumes_filtered = filtered_clusters_df.copy()
cluster_volumes_filtered['Volume'] = cluster_volumes_filtered['Symbol'].map(average_volume_per_stock)
average_volume_by_cluster_filtered = cluster_volumes_filtered.groupby('Cluster')['Volume'].mean()

print("Average Trading Volume by Cluster (Filtered):")
print(average_volume_by_cluster_filtered)


from collections import defaultdict, Counter

# Extract the filtered symbols and their corresponding cluster labels.
filtered_symbols = filtered_clusters_df['Symbol'].tolist()
filtered_clusters = filtered_clusters_df['Cluster'].tolist()


# Adjusted cluster_purity function to directly use filtered data from DataFrame.
def cluster_purity(filtered_clusters, filtered_symbols, sector_mapping):
    symbol_to_cluster = {symbol: cluster for symbol, cluster in zip(filtered_symbols, filtered_clusters)}
    cluster_sectors = defaultdict(Counter)

    for symbol in filtered_symbols:
        cluster_label = symbol_to_cluster[symbol]
        sector = sector_mapping.get(symbol, "Unknown")  # Handle missing symbols safely
        cluster_sectors[cluster_label][sector] += 1

    total_correct = 0
    for sectors in cluster_sectors.values():
        most_common_sector_count = sectors.most_common(1)[0][1]
        total_correct += most_common_sector_count

    purity = total_correct / len(filtered_symbols)
    return purity


# Calculate purity using filtered data.
purity = cluster_purity(filtered_clusters, filtered_symbols, sector_mapping)
print(f"Cluster Purity: {purity}")
