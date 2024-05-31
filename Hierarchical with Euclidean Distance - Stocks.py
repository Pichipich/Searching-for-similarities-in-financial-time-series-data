

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import sqlite3
import pandas as pd
import time
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import linregress, zscore

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
dfs = {}  # This will store filtered data
outliers = []
zscore_threshold = 10  # Set Z-score threshold
unscaled_dfs = {}  # Raw data

for symbol in symbols:
    query = f"SELECT Date, Open, Close, `Adj Close`, Volume FROM `{symbol}` ORDER BY Date"
    df = pd.read_sql(query, conn)
    df.set_index('Date', inplace=True)
    df.interpolate(method='linear', inplace=True)
    df.ffill().bfill()  # Fill missing values

    # Normalize data using Z-score
    normalized_df = df.apply(zscore)

    # Detect and filter outliers based on the normalized data
    if (normalized_df.abs() > zscore_threshold).any().any():
        outliers.append(symbol)
    else:
        dfs[symbol] = normalized_df
        unscaled_dfs[symbol] = df
    # Remove outliers from the symbols list
    symbols = [symbol for symbol in symbols if symbol not in outliers]

conn.close()

print(outliers)

columns_of_interest = ['Open', 'Close', 'Adj Close', 'Volume']
num_symbols = len(symbols)

start_time = time.time()
euclidean_distance_matrix = np.zeros((num_symbols, num_symbols))

from scipy.spatial.distance import euclidean

for i in range(len(symbols)):
    for j in range(i + 1, len(symbols)):
        symbol_i = symbols[i]
        symbol_j = symbols[j]
        if symbol_i in dfs and symbol_j in dfs:
            time_series_data_i = dfs[symbol_i][columns_of_interest].values.flatten()
            time_series_data_j = dfs[symbol_j][columns_of_interest].values.flatten()
            distance = euclidean(time_series_data_i, time_series_data_j)
            euclidean_distance_matrix[i, j] = distance
            euclidean_distance_matrix[j, i] = distance

linkage_matrix = linkage(squareform(euclidean_distance_matrix), metric='precomputed', method='average')

plt.figure(figsize=(16, 8))
dendrogram(linkage_matrix, labels=symbols, leaf_rotation=90, leaf_font_size=8)
plt.title('Dendrogram',fontsize=25)
plt.show()

#determine the clusters
from scipy.cluster.hierarchy import fcluster

max_merge_distance = max(linkage_matrix[:, 2])
threshold = 0.4 * max_merge_distance
clusters = fcluster(linkage_matrix, threshold, criterion='maxclust')

num_clusters = len(np.unique(clusters))
print(f"Number of clusters based on the threshold: {num_clusters}")

# perform MDS on the distance matrix
mds = MDS(n_components=4, random_state=42, dissimilarity='precomputed')
reduced_data_dms = mds.fit_transform(euclidean_distance_matrix)
reduced_data_with_clusters_dms = pd.DataFrame(reduced_data_dms, columns=['MDS1', 'MDS2', 'MDS3', 'MDS4'])


#calculate evaluation metrics for different number of clusters

silhouette_scores_original = []
calinski_harabasz_scores_original = []
davies_bouldin_scores_original = []

for k in range(2, 40):
    clusters = fcluster(linkage_matrix, k, criterion='maxclust')

    silhouette_scores_original.append(silhouette_score(euclidean_distance_matrix, clusters, metric='precomputed'))
    calinski_harabasz_scores_original.append(calinski_harabasz_score(reduced_data_dms, clusters))
    davies_bouldin_scores_original.append(davies_bouldin_score(reduced_data_dms, clusters))

silhouette_scores_normalized_original = [score / max(silhouette_scores_original) for score in silhouette_scores_original]
calinski_harabasz_scores_normalized_original = [score / max(calinski_harabasz_scores_original) for score in calinski_harabasz_scores_original]
davies_bouldin_scores_normalized_original = [1 / (score + 1e-8) for score in davies_bouldin_scores_original]
davies_bouldin_scores_normalized_original = [score / max(davies_bouldin_scores_normalized_original) for score in davies_bouldin_scores_normalized_original]

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.plot(range(2, 40), silhouette_scores_normalized_original, label='Silhouette Score - Original', color='red')
plt.plot(range(2, 40), calinski_harabasz_scores_normalized_original, label='Calinski-Harabasz Index - Original', color='blue')
plt.plot(range(2, 40), davies_bouldin_scores_normalized_original, label='Inverted Davies-Bouldin Index - Original', color='green')

plt.legend()
plt.title('Clustering Quality Scores by Number Of Clusters', fontsize=22)
plt.xlabel('Number of Clusters (k)', fontsize=18)
plt.ylabel('Normalized Scores', fontsize=18)
plt.show()

optimal_clusters_hierarchical = np.argmax(silhouette_scores_original) + 2  # Add 2 to start from k=2
print(f"Optimal Number of Clusters (Silhouette Score): {optimal_clusters_hierarchical}")

optimal_clusters_hierarchical_calinski = np.argmax(calinski_harabasz_scores_original) + 2
print(f"Optimal Number of Clusters (Calinski-Harabasz): {optimal_clusters_hierarchical_calinski}")

optimal_clusters_hierarchical_davies_bouldin = np.argmin(davies_bouldin_scores_original) + 2
print(f"Optimal Number of Clusters (Davies-Bouldin): {optimal_clusters_hierarchical_davies_bouldin}")



#EVALUATION SCORES
# Chosen number of clusters after evaluation
chosen_k =11

# Generating the clusters based on the chosen k
clusters = fcluster(linkage_matrix, chosen_k, criterion='maxclust')

# Calculating the evaluation scores for the chosen k
silhouette_avg = silhouette_score(euclidean_distance_matrix, clusters, metric='precomputed')
calinski_harabasz_avg = calinski_harabasz_score(reduced_data_dms, clusters)
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


clusters = fcluster(linkage_matrix, chosen_k, criterion='maxclust')
clusters_df = pd.DataFrame({'Symbol': symbols, 'Cluster': clusters})

mds = MDS(n_components=4, random_state=42, dissimilarity='precomputed')
mds_result = mds.fit_transform(euclidean_distance_matrix)

plt.scatter(mds_result[:, 0], mds_result[:, 1], c=clusters, cmap='viridis')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.title('MDS Clustering Results [Hierarchical with Euclidean')
plt.show()




import seaborn as sns

# Distance matrix and clustering
processed_symbols = list(dfs.keys())
stock_data = np.array([dfs[symbol].values.flatten() for symbol in processed_symbols])
num_clusters = 7  # Set based on earlier analysis
window_size = 104  # half-yearly data
step_size = 52   # overlap of half a year

def compute_distance_matrix(data):
    num_samples = data.shape[0]
    distance_matrix = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            dist = np.linalg.norm(data[i] - data[j])
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

        linkage_matrix = linkage(squareform(window_distance_matrix), method='average')
        clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

        cluster_labels_over_time.append(clusters)
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
plt.title('Adjusted Rand Index Over Time', fontsize=22)
plt.xlabel('Time Window Index')
plt.ylabel('Adjusted Rand Index')
plt.grid(True)
plt.show()

# Create a heatmap for number of clusters vs. time windows
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_labels_matrix, annot=False, cmap='viridis', cbar_kws={'label': 'Cluster Label'})
plt.title('Cluster Assignments Over Time Windows', fontsize=22)
plt.xlabel('Time Window Index')
plt.ylabel('Cluster Number')
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
ax.set_title('Cluster Evolution Over Time [Hierarchical - Euclidean]', fontsize=20)
ax.set_xlabel('Time Interval', fontsize=18)
ax.set_ylabel('Stock Index', fontsize=18)
ax.set_xticklabels(window_labels, rotation=45, ha="right")  # Rotate for better visibility
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


# Calculation of average ARI across time windows
if ari_scores:
    # Calculate the mean of the ARI scores
    mean_ari = np.mean(ari_scores)
    print(f"Average ARI across all time windows: {mean_ari}")
else:
    print("No ARI scores to average.")


import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import fcluster


mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_result = mds.fit_transform(euclidean_distance_matrix)

colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
    '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'
]

# Create a colormap from the list of colors
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(colors[:chosen_k])

plt.figure(figsize=(8, 6))  
scatter = plt.scatter(mds_result[:, 0], mds_result[:, 1], c=clusters, cmap=custom_cmap, edgecolor='none', s=50)
plt.xlabel('MDS Dimension 1', fontsize=14)
plt.ylabel('MDS Dimension 2', fontsize=14)
plt.title('MDS Clustering Results [Hierarchical with Euclidean]', fontsize=16)
plt.grid(True)
plt.axis('equal')  
plt.show()


clusters_df['Cluster'] = clusters

# Print the names of the stocks in each cluster
for cluster_num in sorted(clusters_df['Cluster'].unique()):
    cluster_stocks = clusters_df[clusters_df['Cluster'] == cluster_num]['Symbol'].tolist()
    print(f"Cluster {cluster_num}: {', '.join(cluster_stocks)}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")





#OUTLIERS

min_cluster_size = 2
cluster_sizes = clusters_df['Cluster'].value_counts()
outlier_clusters = cluster_sizes[cluster_sizes < min_cluster_size].index

outlier_stocks = []
for cluster in outlier_clusters:
    stocks_in_cluster = clusters_df[clusters_df['Cluster'] == cluster]['Symbol'].tolist()
    outlier_stocks.extend(stocks_in_cluster)

outlier_indices = np.array([symbols.index(stock) for stock in outlier_stocks]).astype(int)


filtered_distance_matrix = np.delete(euclidean_distance_matrix, outlier_indices, axis=0)
filtered_distance_matrix = np.delete(filtered_distance_matrix, outlier_indices, axis=1)

filtered_clusters = np.delete(clusters, outlier_indices)
filtered_clusters, _ = pd.factorize(filtered_clusters)
filtered_linkage_matrix = linkage(squareform(filtered_distance_matrix), method='average')

print("Outlier Stocks:")
for stock in outlier_stocks:
    print(stock)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")
end_time = time.time()





sector_mapping = {
    "HDB": "Finance", "SCHW": "Finance", "ELV": "Health Care", "BLK": "Finance",
    "CAH": "Health Care", "GS": "Finance", "RIO": "Basic Materials", "LMT": "Industrials",
    "SYK": "Health Care", "BKNG": "Consumer Discretionary", "BUD": "Consumer Staples",
    "PLD": "Real Estate", "BCS": "Finance", "ISRG": "Health Care", "SONY": "Consumer Staples",
    "SBUX": "Consumer Discretionary", "TD": "Finance", "MUFG": "Finance", "MDT": "Health Care",
    "DE": "Industrials", "BMY": "Health Care", "BP": "Energy", "TJX": "Consumer Discretionary",
    "AMT": "Real Estate", "MMC": "Finance", "MDLZ": "Consumer Staples", "SHOP": "Technology",
    "UBS": "Finance", "PGR": "Finance", "PBR": "Energy", "ADP": "Technology", "EQNR": "Energy",
    "CB": "Finance", "PANW": "Technology", "LRCX": "Technology", "VRTX": "Technology",
    "ETN": "Technology", "ADI": "Technology", "REGN": "Health Care", "C": "Finance",
    "ABNB": "Consumer Discretionary", "IBN": "Finance", "ZTS": "Health Care", "BX": "Finance",
    "SNPS": "Technology", "BSX": "Health Care", "MELI": "Consumer Discretionary",
    "DEO": "Consumer Staples", "CME": "Finance", "FI": "Consumer Discretionary", "SO": "Utilities",
    "EQIX": "Real Estate", "CNI": "Industrials", "CI": "Health Care", "MO": "Consumer Staples",
    "ENB": "Energy", "GSK": "Health Care", "ITW": "Industrials", "INFY": "Technology",
    "RELX": "Consumer Discretionary", "KLAC": "Technology", "SHW": "Consumer Discretionary",
    "CNQ": "Energy", "SLB": "Energy", "NOC": "Industrials", "DUK": "Utilities",
    "EOG": "Energy", "BALL": "Industrials", "WDAY": "Technology", "VALE": "Basic Materials",
    "RACE": "Consumer Discretionary", "WM": "Utilities", "STLA": "Consumer Discretionary",
    "MCO": "Finance", "GD": "Industrials", "CP": "Industrials", "BDX": "Health Care",
    "RTO": "Consumer Discretionary", "SAN": "Finance", "HCA": "Health Care", "TRI": "Consumer Discretionary",
    "ANET": "Technology", "FDX": "Consumer Discretionary", "KKR": "Finance", "NTES": "Technology",
    "SMFG": "Finance", "CSX": "Industrials", "ICE": "Finance", "AON": "Finance",
    "CL": "Consumer Discretionary", "BTI": "Consumer Staples", "ITUB": "Finance",
    "PYPL": "Consumer Discretionary", "HUM": "Health Care", "TGT": "Consumer Discretionary",
    "MCK": "Health Care", "SNOW": "Technology", "CMG": "Consumer Discretionary", "BMO": "Finance",
    "MAR": "Consumer Discretionary", "APD": "Industrials", "AMX": "Telecommunications",
    "EPD": "Utilities", "ORLY": "Consumer Discretionary", "E": "Energy", "CRWD": "Technology",
    "ROP": "Industrials", "MPC": "Energy", "PSX": "Energy", "MMM": "Health Care",
    "CTAS": "Consumer Discretionary", "PH": "Industrials", "BBVA": "Finance",
    "LULU": "Consumer Discretionary", "BN": "Finance", "SCCO": "Basic Materials",
    "HMC": "Consumer Discretionary", "PNC": "Finance", "APH": "Technology",
    "ECL": "Consumer Discretionary", "CHTR": "Telecommunications", "MSI": "Technology",
    "BNS": "Finance", "NXPI": "Technology", "TDG": "Industrials", "AJG": "Finance",
    "PFH": "Finance", "PXD": "Energy", "ING": "Finance", "FCX": "Basic Materials",
    "TT": "Industrials", "APO": "Finance", "CCI": "Real Estate", "RSG": "Utilities",
    "NSC": "Industrials", "OXY": "Energy", "EMR": "Technology", "DELL": "Technology",
    "TEAM": "Technology", "PCAR": "Consumer Discretionary", "PCG": "Utilities",
    "WPP": "Consumer Discretionary", "AFL": "Finance", "WELL": "Real Estate", "MET": "Finance",
    "AESC": "Utilities", "EL": "Consumer Discretionary", "PSA": "Real Estate",
    "AZO": "Consumer Discretionary", "ADSK": "Technology", "CPRT": "Consumer Discretionary",
    "BSBR": "Finance", "AIG": "Finance", "DXCM": "Health Care", "MCHP": "Technology",
    "ABEV": "Consumer Staples", "KDP": "Consumer Staples", "ROST": "Consumer Discretionary",
    "GM": "Consumer Discretionary", "CRH": "Industrials", "SRE": "Utilities",
    "PAYX": "Consumer Discretionary", "WMB": "Utilities", "CARR": "Industrials",
    "KHC": "Consumer Staples", "COF": "Finance", "MRVL": "Technology", "DHI": "Consumer Discretionary",
    "STZ": "Consumer Staples", "TAK": "Health Care", "ET": "Utilities", "IDXX": "Health Care",
    "ODFL": "Industrials", "HLT": "Consumer Discretionary", "STM": "Technology",
    "VLO": "Energy", "SPG": "Real Estate", "HES": "Energy", "F": "Consumer Discretionary",
    "MFG": "Finance", "DLR": "Real Estate", "TRV": "Finance", "EW": "Health Care",
    "AEP": "Utilities", "SU": "Energy", "MSCI": "Consumer Discretionary", "JD": "Consumer Discretionary",
    "KMB": "Consumer Staples", "COR": "Real Estate", "NUE": "Industrials", "SGEN": "Health Care",
    "LNG": "Utilities", "OKE": "Utilities", "FTNT": "Technology", "TEL": "Technology",
    "CNC": "Health Care", "SQ": "Technology", "PLTR": "Technology", "O": "Real Estate",
    "BIDU": "Technology", "GWW": "Industrials", "NEM": "Basic Materials", "ADM": "Consumer Staples",
    "CM": "Finance", "TRP": "Utilities", "IQV": "Health Care", "KMI": "Utilities",
    "DDOG": "Technology", "D": "Utilities", "KVUE": "Consumer Discretionary", "SPOT": "Consumer Discretionary",
    "HSY": "Consumer Staples", "EXC": "Utilities", "DASH": "Consumer Discretionary", "HLN": "Consumer Discretionary",
    "CEG": "Utilities", "LHX": "Industrials", "GIS": "Consumer Staples", "A": "Industrials",
    "BK": "Finance", "JCI": "Industrials", "EA": "Technology", "SYY": "Consumer Discretionary",
    "BCE": "Telecommunications", "WDS": "Energy", "LI": "Consumer Discretionary", "MPLX": "Energy",
    "ALL": "Finance", "WCN": "Utilities", "ALC": "Health Care", "MFC": "Consumer Discretionary",
    "AME": "Industrials", "DOW": "Industrials", "AMP": "Finance", "FERG": "Miscellaneous",
    "BBD": "Finance", "PRU": "Finance", "FIS": "Consumer Discretionary", "CTSH": "Technology",
    "OTIS": "Technology", "YUM": "Consumer Discretionary", "FAST": "Consumer Discretionary",
    "VRSK": "Technology", "CSGP": "Consumer Discretionary", "LVS": "Consumer Discretionary",
    "IT": "Consumer Discretionary", "XEL": "Utilities", "ARES": "Finance", "PPG": "Consumer Discretionary",
    "COIN": "Finance", "TTD": "Technology", "IMO": "Energy", "BKR": "Industrials",
    "HAL": "Energy", "CMI": "Industrials", "URI": "Industrials", "NDAQ": "Finance",
    "KR": "Consumer Staples", "ORAN": "Telecommunications", "ROK": "Industrials", "CVE": "Energy",
    "ED": "Utilities", "SATX": "Technology", "DKNG": "Consumer Discretionary", "VICI": "Real Estate",
    "BBDO": "Finance", "PEG": "Utilities", "ON": "Technology", "MDB": "Technology",
    "CTVA": "Industrials", "GEHC": "Technology", "GPN": "Consumer Discretionary", "GOLD": "Basic Materials",
    "ACGL": "Finance", "DD": "Industrials", "LYB": "Industrials", "SYM": "Industrials", "HBANM": "Finance", "SLF": "Finance", "CHT": "Telecommunications",
    "MRNA": "Health Care", "NU": "Finance", "PUK": "Finance", "CQP": "Utilities",
    "RCL": "Consumer Discretionary", "DG": "Consumer Discretionary", "ZS": "Technology",
    "IR": "Industrials", "EXR": "Real Estate", "VEEV": "Technology", "CCEP": "Consumer Staples",
    "HPQ": "Technology", "MLM": "Industrials", "GFS": "Technology", "CDW": "Consumer Discretionary",
    "VMC": "Industrials", "DVN": "Energy", "FICO": "Consumer Discretionary", "DLTR": "Consumer Discretionary",
    "EFX": "Finance", "CPNG": "Consumer Discretionary", "PWR": "Industrials", "FMX": "Consumer Staples",
    "TU": "Telecommunications", "SBAC": "Real Estate", "PKX": "Industrials", "FANG": "Energy",
    "TTWO": "Technology", "MPWR": "Technology", "WBD": "Telecommunications", "WEC": "Utilities",
    "NTR": "Industrials", "WIT": "Technology", "AEM": "Basic Materials", "HBANP": "Finance",
    "NET": "Technology", "VOD": "Telecommunications", "ELP": "Utilities", "EC": "Energy",
    "EIX": "Utilities", "AWK": "Utilities", "SPLK": "Technology", "XYL": "Industrials",
    "ARGX": "Health Care", "BNH": "Real Estate", "DB": "Finance", "WST": "Health Care",
    "RBLX": "Technology", "HUBS": "Technology", "WTW": "Finance", "AVB": "Real Estate",
    "TEF": "Telecommunications", "DFS": "Finance", "CBRE": "Finance", "TLK": "Telecommunications",
    "KEYS": "Industrials", "BNJ": "Real Estate", "NWG": "Finance", "GLW": "Technology",
    "GIB": "Consumer Discretionary", "ANSS": "Technology", "ZBH": "Health Care", "DAL": "Consumer Discretionary",
    "HEI": "Industrials", "SNAP": "Technology", "FTV": "Industrials", "BNTX": "Health Care",
    "GRMN": "Industrials", "HIG": "Finance", "RMD": "Health Care", "RCI": "Telecommunications",
    "MTD": "Industrials", "ULTA": "Consumer Discretionary", "CHD": "Consumer Discretionary",
    "IX": "Finance", "PINS": "Technology", "APTV": "Consumer Discretionary", "BR": "Consumer Discretionary",
    "WY": "Real Estate", "QSR": "Consumer Discretionary", "STT": "Finance", "TROW": "Finance",
    "TSCO": "Consumer Discretionary", "TW": "Finance", "VRSN": "Technology", "EQR": "Real Estate",
    "ICLR": "Health Care", "DTE": "Utilities", "RJF": "Finance", "MTB": "Finance",
    "WPM": "Basic Materials", "CCL": "Consumer Discretionary", "EBAY": "Consumer Discretionary",
    "HWM": "Industrials", "SE": "Consumer Discretionary", "MOH": "Health Care", "ALNY": "Health Care",
    "WAB": "Industrials", "TCOM": "Consumer Discretionary", "FE": "Utilities", "ETR": "Utilities",
    "FCNCA": "Finance", "BRO": "Finance", "ES": "Utilities", "ZM": "Technology",
    "ARE": "Real Estate", "FNV": "Basic Materials", "HPE": "Telecommunications", "FITB": "Finance",
    "AEE": "Utilities", "INVH": "Finance", "CBOE": "Finance", "MT": "Industrials",
    "NVR": "Consumer Discretionary", "TS": "Industrials", "ROL": "Finance", "CCJ": "Basic Materials",
    "DOV": "Industrials", "FTS": "Utilities", "STE": "Health Care", "TRGP": "Utilities",
    "JBHT": "Industrials", "UMC": "Technology", "RKT": "Finance", "BEKE": "Finance",
    "EBR": "Utilities", "IRM": "Real Estate", "BGNE": "Health Care", "DRI": "Consumer Discretionary",
    "IFF": "Industrials", "EXPE": "Consumer Discretionary", "PPL": "Utilities", "PTC": "Technology",
    "CTRA": "Energy", "TECK": "Basic Materials", "TDY": "Industrials", "VTR": "Real Estate",
    "WRB": "Finance", "STLD": "Industrials", "GPC": "Consumer Discretionary", "ASX": "Technology",
    "LYV": "Consumer Discretionary", "OWL": "Finance", "DUKB": "Utilities", "NTAP": "Technology",
    "VLTO": "Industrials", "IOT": "Technology", "MKL": "Finance", "PBA": "Energy",
    "LH": "Health Care", "KOF": "Consumer Staples", "K": "Consumer Staples", "ERIC": "Technology",
    "BAX": "Health Care", "FLT": "Consumer Discretionary", "CNP": "Utilities", "MKC": "Consumer Staples",
    "VIV": "Telecommunications", "PFG": "Finance", "DECK": "Consumer Discretionary", "ILMN": "Health Care",
    "WMG": "Consumer Discretionary", "TSN": "Consumer Staples", "WBA": "Consumer Staples", "BLDR": "Consumer Discretionary",
    "BMRN": "Health Care", "CHKP": "Technology", "EXPD": "Consumer Discretionary", "PHG": "Health Care",
    "CLX": "Consumer Discretionary", "AKAM": "Consumer Discretionary", "ZTO": "Industrials", "FITBI": "Finance",
    "AXON": "Industrials", "SIRI": "Consumer Discretionary", "TYL": "Technology", "TVE": "Utilities",
    "AQNB": "Utilities", "EG": "Finance", "VRT": "Technology", "HRL": "Consumer Staples",
    "RYAN": "Finance", "RPRX": "Health Care", "FDS": "Technology", "AGNCN": "Real Estate",
    "ATO": "Utilities", "YUMC": "Consumer Discretionary", "NOK": "Technology", "EDR": "Consumer Discretionary",
    "HBAN": "Finance", "WAT": "Industrials", "AER": "Consumer Discretionary", "LPLA": "Finance",
    "CMS": "Utilities", "NTRS": "Finance", "BAH": "Consumer Discretionary", "RIVN": "Consumer Discretionary",
    "WLK": "Industrials", "HOLX": "Health Care", "COO": "Health Care", "FSLR": "Technology",
    "ALGN": "Health Care", "ASBA": "Finance", "FITBP": "Finance", "SUI": "Real Estate",
    "LUV": "Consumer Discretionary", "CINF": "Finance", "OMC": "Consumer Discretionary", "STX": "Technology",
    "J": "Industrials", "HUBB": "Technology", "DT": "Technology", "AGNCM": "Real Estate",
    "SWKS": "Technology", "RF": "Finance", "VFS": "Finance", "EQT": "Energy",
    "ENTG": "Technology", "L": "Finance", "AGNCO": "Real Estate", "MGA": "Consumer Discretionary",
    "FITBO": "Finance", "WSO": "Consumer Discretionary", "AVY": "Consumer Discretionary", "SLMBP": "Finance",
    "RS": "Industrials", "BSY": "Technology", "BG": "Consumer Staples", "IEX": "Industrials",
    "CE": "Industrials", "KB": "Finance", "DGX": "Health Care", "WDC": "Technology",
    "SREA": "Consumer Discretionary", "LDOS": "Technology", "SOJE": "Basic Materials", "ENPH": "Technology",
    "ROKU": "Telecommunications", "TXT": "Industrials", "AQNU": "Utilities", "PKG": "Consumer Discretionary",
    "MAA": "Real Estate", "EPAM": "Technology", "SNA": "Consumer Discretionary", "LII": "Industrials",
    "AEG": "Finance", "GDDY": "Technology", "JBL": "Technology", "FWONK": "Consumer Discretionary",
    "AGNCP": "Real Estate", "LW": "Consumer Staples", "CNHI": "Industrials", "AGNCL": "Real Estate",
    "JHX": "Industrials", "MRO": "Energy", "GEN": "Technology", "FOXA": "Industrials",
    "SHG": "Finance", "ESS": "Real Estate", "WPC": "Real Estate", "ERIE": "Finance",
    "SYF": "Finance", "CSL": "Industrials", "SMCI": "Technology", "AVTR": "Industrials",
    "SWK": "Consumer Discretionary", "SQM": "Industrials", "CF": "Industrials", "MANH": "Technology",
    "MAS": "Consumer Discretionary", "PATH": "Technology", "SSNC": "Technology", "BKDT": "Health Care",
    "TER": "Industrials", "TME": "Consumer Discretionary", "PARAP": "Industrials", "BAM": "Real Estate",
    "GGG": "Industrials", "CAG": "Consumer Staples", "DPZ": "Consumer Discretionary", "LOGI": "Technology",
    "POOL": "Consumer Discretionary", "NDSN": "Industrials", "CFG": "Finance", "AMCR": "Consumer Discretionary",
    "IHG": "Consumer Discretionary", "RPM": "Consumer Discretionary"}


filtered_sector_mapping = {
    symbol: sector for symbol, sector in sector_mapping.items() if symbol in symbols
}


filtered_clusters_df = clusters_df[~clusters_df['Symbol'].isin(outlier_stocks)]

sector_df = pd.DataFrame(list(filtered_sector_mapping.items()), columns=['Symbol', 'Sector'])
merged_df_filtered = pd.merge(filtered_clusters_df, sector_df, on='Symbol')

sector_composition_filtered = merged_df_filtered.groupby(['Cluster', 'Sector']).size().unstack(fill_value=0)

sector_composition_filtered.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
plt.title('Sector Composition within Clusters (Excluding Outliers)')
plt.xlabel('Cluster')
plt.ylabel('Number of Stocks')
plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

sector_totals = sector_df['Sector'].value_counts()
cluster_sector_counts = merged_df_filtered.groupby(['Cluster', 'Sector']).size().unstack(fill_value=0)
cluster_sector_percentages = cluster_sector_counts.div(cluster_sector_counts.sum(axis=1), axis=0) * 100
print(cluster_sector_percentages)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(cluster_sector_percentages, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Hierarchical With Euclidean Distance',fontsize=20)
plt.xlabel('Sector',fontsize=18)
plt.ylabel('Cluster',fontsize=18)
plt.xticks(rotation=55, fontsize=14)
plt.yticks(rotation=0)
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
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(cluster_sector_percentages, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5)
plt.title('Hierarchical Clustering with Euclidean', fontsize=22, pad=20)
plt.xlabel('Sector', fontsize=18)
plt.ylabel('Cluster', fontsize=18)
plt.xticks(rotation=45, fontsize=14, ha='right')
plt.yticks(rotation=0, fontsize=14)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Increase the figure size to better accommodate larger cells
plt.figure(figsize=(20, 12))

# Create the heatmap with increased font size for annotations
heatmap = sns.heatmap(cluster_sector_percentages, annot=True, cmap='Blues', fmt=".2f", linewidths=.5, annot_kws={"fontsize": 16})

# Title and labels with appropriate font sizes
plt.title('Hierarchical Clustering with Euclidean', fontsize=28, pad=20)
plt.xlabel('Company', fontsize=22)
plt.ylabel('Cluster', fontsize=22)

# Rotate tick labels for better visibility and fit
plt.xticks(rotation=45, fontsize=16, ha='right')
plt.yticks(rotation=0, fontsize=16)

# Adjust layout to make room for all elements; padding added to prevent clipping of tick labels
plt.tight_layout(pad=1)

# Get the color data of the heatmap
color_data = heatmap.collections[0].get_facecolors()

# Adjust text color based on background color for better visibility
for text, color in zip(heatmap.texts, color_data):
    # Calculate the brightness of the background
    luminance = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]  # Standard luminance calculation
    text.set_color('white' if luminance < 0.5 else 'black')

# Display the plot
plt.show()
