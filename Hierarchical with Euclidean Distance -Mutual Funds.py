from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import sqlite3
import pandas as pd
import time
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import linregress
from scipy.spatial.distance import euclidean, squareform
from collections import defaultdict, Counter
import seaborn as sns
from scipy.cluster.hierarchy import fcluster

symbols = [
    "WWWFX", "KINCX", "KINAX", "KMKNX", "KMKCX", "KMKAX", "KMKYX", "KNPAX", "KNPYX", "KNPCX",
    "WWNPX", "KSCYX", "KSOAX", "KSCOX", "KSOCX", "LSHEX", "LSHAX", "LSHUX", "LSHCX", "ENPIX",
    "ENPSX", "FNRCX", "FSENX", "FIKAX", "FANAX", "FANIX", "FAGNX", "NEEIX", "NEEGX", "FIJGX",
    "FSTCX", "FTUTX", "FTUCX", "FTUAX", "FTUIX", "RCMFX", "FNARX", "HICGX", "HFCGX",
    "RYPMX", "RYMNX", "RYMPX", "RYZCX", "FUGAX", "FCGCX", "FIQRX", "FFGTX", "FFGAX", "FFGIX",
    "FFGCX", "FIKIX", "FUGIX", "FSUTX", "FAUFX", "FUGCX", "BIVIX",  "BIVRX", "NEAIX",
    "FSLBX", "NEAGX", "QLEIX", "QLERX", "FACVX", "FTCVX", "FIQVX", "FICVX", "FSPCX", "RMLPX",
    "FCCVX", "FCVSX", "EAFVX", "EIFVX", "DGIFX", "AUERX", "COAGX", "TAVZX", "TAVFX", "TVFVX",
    "ECFVX", "SGGDX", "EICVX", "EICIX", "MBXAX", "UBVVX", "UBVAX", "UBVFX", "MBXIX", "FEURX",
    "UBVRX", "UBVTX", "UBVUX", "UBVSX", "DHTAX", "UBVLX", "UBVCX",  "MBXCX", "DHTYX","HWSCX",
    "HWSIX", "EIPIX", "HWSAX"
]

conn = sqlite3.connect('mutual_small_data.db')

dfs = {}
columns_to_exclude = ['High', 'Low', 'Open', 'Close']
unscaled_dfs = {}

for symbol in symbols:
    query = f"SELECT * FROM `{symbol}` ORDER BY Date"
    dfs[symbol] = pd.read_sql(query, conn)
    unscaled_dfs[symbol] = dfs[symbol].copy()

    columns_to_infer = ['Adj Close']
    dfs[symbol][columns_to_infer] = dfs[symbol][columns_to_infer].infer_objects()

    dfs[symbol] = dfs[symbol].drop(columns=columns_to_exclude)
    dfs[symbol] = dfs[symbol].interpolate(method='linear', axis=0).ffill().bfill()

    for column in columns_to_infer:
        dfs[symbol][column] = (dfs[symbol][column] - dfs[symbol][column].mean()) / dfs[symbol][column].std()

conn.close()

start_time = time.time()
columns_of_interest = ['Adj Close']

for symbol, df in dfs.items():
    df['Return'] = df['Adj Close'].pct_change()

for symbol, df in dfs.items():
    df.dropna(inplace=True)

num_symbols = len(symbols)

start_time = time.time()
returns_data = np.array([dfs[symbol]['Return'].values for symbol in symbols if 'Return' in dfs[symbol].columns])

max_length = min([len(dfs[symbol]['Return']) for symbol in symbols if 'Return' in dfs[symbol].columns])
equal_length_data = np.array([dfs[symbol]['Return'].iloc[:max_length].values for symbol in symbols if 'Return' in dfs[symbol].columns])

distance_matrix = np.zeros((num_symbols, num_symbols))

from scipy.spatial.distance import euclidean
#calculate euclidean matrix
for i in range(len(symbols)):
    for j in range(i + 1, len(symbols)):
        symbol_i = symbols[i]
        symbol_j = symbols[j]

        time_series_data_i = dfs[symbol_i][columns_of_interest].values.flatten()
        time_series_data_j = dfs[symbol_j][columns_of_interest].values.flatten()

        distance = euclidean(time_series_data_i, time_series_data_j)

        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance


linkage_matrix = linkage(squareform(distance_matrix), method='average')

plt.figure(figsize=(16, 8))
dendrogram(linkage_matrix, labels=symbols, leaf_rotation=90, leaf_font_size=8)
plt.title('Dendrogram',fontsize=25)
plt.show()



# perform MDS on the distance matrix
mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
reduced_data_dms = mds.fit_transform(distance_matrix)

reduced_data_with_clusters_dms = pd.DataFrame(reduced_data_dms, columns=['MDS1', 'MDS2'])


#scores before outlier detection

silhouette_scores_original = []
calinski_harabasz_scores_original = []
davies_bouldin_scores_original = []

for k in range(2, 20):
    clusters = fcluster(linkage_matrix, k, criterion='maxclust')

    silhouette_scores_original.append(silhouette_score(distance_matrix, clusters, metric='precomputed'))
    calinski_harabasz_scores_original.append(calinski_harabasz_score(reduced_data_dms, clusters))
    davies_bouldin_scores_original.append(davies_bouldin_score(reduced_data_dms, clusters))

silhouette_scores_normalized_original = [score / max(silhouette_scores_original) for score in silhouette_scores_original]
calinski_harabasz_scores_normalized_original = [score / max(calinski_harabasz_scores_original) for score in calinski_harabasz_scores_original]
davies_bouldin_scores_normalized_original = [1 / (score + 1e-8) for score in davies_bouldin_scores_original]
davies_bouldin_scores_normalized_original = [score / max(davies_bouldin_scores_normalized_original) for score in davies_bouldin_scores_normalized_original]

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.plot(range(2, 20), silhouette_scores_normalized_original, label='Silhouette Score - Original', color='red')
plt.plot(range(2, 20), calinski_harabasz_scores_normalized_original, label='Calinski-Harabasz Index - Original', color='blue')
plt.plot(range(2, 20), davies_bouldin_scores_normalized_original, label='Inverted Davies-Bouldin Index - Original', color='green')

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


#Evaluation scores 

# Chosen number of clusters after evaluation
chosen_k = 9

# Generating the clusters based on the chosen k
clusters = fcluster(linkage_matrix, chosen_k, criterion='maxclust')

# Calculating the evaluation scores for the chosen k
silhouette_avg = silhouette_score(distance_matrix, clusters, metric='precomputed')
calinski_harabasz_avg = calinski_harabasz_score(reduced_data_dms, clusters)
davies_bouldin_avg = davies_bouldin_score(reduced_data_dms, clusters)

# Print out the evaluation scores for the chosen k
print(f"Silhouette Score: {silhouette_avg}")
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

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

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")


# Print the names of the funds in each cluster
for cluster_num in sorted(clusters_df['Cluster'].unique()):
    cluster_funds = clusters_df[clusters_df['Cluster'] == cluster_num]['Symbol'].tolist()
    print(f"Cluster {cluster_num}: {', '.join(cluster_funds)}")


#Plot MDS scatterplot
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_result = mds.fit_transform(distance_matrix)

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



# Distance matrix and clustering between subsequences 
processed_symbols = list(dfs.keys())
fund_data = np.array([dfs[symbol].values.flatten() for symbol in processed_symbols])
num_clusters = 9  
window_size = 26  # half yearly data
step_size = 13   # overlap of half a year

def compute_distance_matrix(data):
    print(data)  # Debugging statement to check what data looks like
    num_samples = data.shape[0]
    distance_matrix = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            if isinstance(data[i], str) or isinstance(data[j], str):
                raise TypeError("Data contains string values, which are not supported for distance calculation.")
            dist = np.linalg.norm(data[i] - data[j])
            distance_matrix[i, j] = distance_matrix[j, i] = dist
    return distance_matrix

def temporal_cluster_validation(fund_data, window_size, step_size, num_clusters):
    num_samples = fund_data.shape[0]
    num_points = fund_data.shape[1]
    num_windows = (num_points - window_size) // step_size + 1
    cluster_labels_over_time = []

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window_data = fund_data[:, start_idx:end_idx]

        # Ensure window data is correctly reshaped if needed
        flattened_data = window_data.reshape(window_data.shape[0], -1)
        window_distance_matrix = compute_distance_matrix(flattened_data)

        linkage_matrix = linkage(squareform(window_distance_matrix), method='average')
        clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

        cluster_labels_over_time.append(clusters)

    return np.array(cluster_labels_over_time).T  

# Prepare the return data for temporal clustering
processed_symbols = list(dfs.keys())
fund_returns = np.array([dfs[symbol]['Return'].dropna().values for symbol in processed_symbols if 'Return' in dfs[symbol].columns])

# Calculate cluster labels over time
cluster_labels_matrix = temporal_cluster_validation(fund_returns, window_size=26, step_size=13, num_clusters=35)


from sklearn.metrics import adjusted_rand_score
#function to compute ARI over time
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


# Assuming 'cluster_labels_matrix' holds the cluster labels for each fund over time.
num_windows = cluster_labels_matrix.shape[1]
window_labels = [f'Time Window {i+1}' for i in range(num_windows)]
plt.figure(figsize=(16, 14))

# Create the heatmap without annotations
ax = sns.heatmap(cluster_labels_matrix, cmap='Blues', cbar_kws={'label': 'Cluster Number'})
plt.suptitle('Cluster Evolution Over Time [Hierarchical - Euclidean]', fontsize=35, va='top', ha='center', x=0.5, y=0.95)
ax.set_xlabel('Time Interval', fontsize=18)
ax.set_ylabel('Fund Index', fontsize=18)
ax.set_xticklabels(window_labels, rotation=45, ha="right", fontsize=14)  
ax.tick_params(axis='y', labelsize=14)
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('Cluster Number', fontsize=18)  
cbar.ax.tick_params(labelsize=16)  
plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.show()


from sklearn.metrics import adjusted_rand_score
#compute average ARI
def compute_temporal_ari(cluster_labels_matrix):
    num_windows = cluster_labels_matrix.shape[1]
    ari_scores = []

    for i in range(num_windows - 1):
        current_labels = cluster_labels_matrix[:, i]
        next_labels = cluster_labels_matrix[:, i + 1]
        ari = adjusted_rand_score(current_labels, next_labels)
        ari_scores.append(ari)

    return ari_scores

# Calculate the ARI scores over time
ari_scores = compute_temporal_ari(cluster_labels_matrix)

# Compute the average ARI if there are any scores
if ari_scores:
    average_ari = sum(ari_scores) / len(ari_scores)
    print("Average Adjusted Rand Index (ARI):", average_ari)
else:
    print("No ARI scores available to average.")


#plot heatmap of clusters over time windows
num_windows = cluster_labels_matrix.shape[1]  # Number of time windows

fig, ax = plt.subplots(1, 2, figsize=(12, 6))  

sns.heatmap(cluster_labels_matrix, annot=False, cmap='Blues', cbar_kws={'label': 'Cluster Number'}, ax=ax[0])
ax[0].set_title('Cluster Assignments Over Time Windows')
ax[0].set_xlabel('Time Window Index')
ax[0].set_ylabel('Fund Index')

ax[1].plot(range(1, len(ari_scores) + 1), ari_scores, linestyle='-', marker='o')
ax[1].set_title('Adjusted Rand Index Over Time')
ax[1].set_xlabel('Time Window Index')
ax[1].set_ylabel('Adjusted Rand Index')
ax[1].grid(True)

plt.tight_layout()
plt.savefig("plots.png", dpi=300)  # High-resolution save for better quality in documents
plt.show()



#exclusion of outliers
min_cluster_size =2
cluster_sizes = clusters_df['Cluster'].value_counts()
outlier_clusters = cluster_sizes[cluster_sizes < min_cluster_size].index

outlier_fund = []
for cluster in outlier_clusters:
    fund_in_cluster = clusters_df[clusters_df['Cluster'] == cluster]['Symbol'].tolist()
    outlier_fund.extend(s=funds_in_cluster)

outlier_indices = np.array([symbols.index(fund) for fund in outlier_funds]).astype(int)

filtered_distance_matrix = np.delete(distance_matrix, outlier_indices, axis=0)
filtered_distance_matrix = np.delete(filtered_distance_matrix, outlier_indices, axis=1)

filtered_clusters = np.delete(clusters, outlier_indices)
filtered_clusters, _ = pd.factorize(filtered_clusters)
filtered_linkage_matrix = linkage(squareform(filtered_distance_matrix), method='average')

mds = MDS(n_components=4, dissimilarity='precomputed', random_state=42)
mds_result_filtered = mds.fit_transform(filtered_distance_matrix)


print("Outlier funds:")
for fund in outlier_funds:
    print(fund)

for cluster_num in sorted(clusters_df['Cluster'].unique()):
    cluster_funds = clusters_df[clusters_df['Cluster'] == cluster_num]['Symbol'].tolist()
    print(f"Cluster {cluster_num}: {', '.join(cluster_funds)}")




company_mapping = {
    "WWWFX": "Kinetics",
    "KINCX": "Kinetics", "KINAX": "Kinetics", "KMKNX": "Kinetics", "KMKCX": "Kinetics", "KMKAX": "Kinetics", "KMKYX": "Kinetics","KNPAX": "Kinetics","KNPYX": "Kinetics", "KNPCX": "Kinetics", "WWNPX": "Kinetics","KSCYX": "Kinetics","KSOAX": "Kinetics",
    "KSCOX": "Kinetics", "KSOCX": "Kinetics", "LSHEX": "Kinetics", "LSHAX": "Kinetics", "LSHUX": "Kinetics", "LSHCX": "Kinetics", "ENPIX": "ProFunds", "ENPSX": "ProFunds",    "FNRCX": "Fidelity",
    "FSENX": "Fidelity",  "FIKAX": "Fidelity",  "FANAX": "Fidelity",   "FANIX": "Fidelity", "FAGNX": "Fidelity",
    "NEEIX": "Needham","NEEGX": "Needham", "FIJGX": "Fidelity", "FSTCX": "Fidelity", "FTUTX": "Fidelity", "FTUCX": "Fidelity",
    "FTUAX": "Fidelity", "FTUIX": "Fidelity","RCMFX": "Schwartz", "FNARX": "Fidelity",
    "FMEIX": "Fidelity",  "HICGX": "Hennessy",  "HFCGX": "Hennessy",  "RYPMX": "Rydex",  "RYMNX": "Rydex",   "RYMPX": "Rydex",   "RYZCX": "Rydex",   "FUGAX": "Fidelity",   "FCGCX": "Fidelity",  "FIQRX": "Fidelity", "FFGTX": "Fidelity", "FFGAX": "Fidelity",
    "FFGIX": "Fidelity", "FFGCX": "Fidelity", "FIKIX": "Fidelity","FUGIX": "Fidelity", "FSUTX": "Fidelity",
    "FAUFX": "Fidelity",   "FUGCX": "Fidelity",   "BIVIX": "Invenomic",   "BIVSX": "Invenomic", "BIVRX": "Invenomic","NEAIX": "Needham",
    "FSLBX": "Fidelity","NEAGX": "Needham", "QLEIX": "AQR Long-Short Equity ", "QLERX": "AQR Long-Short Equity ",  "FACVX": "Fidelity",
    "FTCVX": "Fidelity",  "FIQVX": "Fidelity",  "FICVX": "Fidelity", "FSPCX": "Fidelity",  "RMLPX": "Two Roads Shared Trust",   "FCCVX": "Fidelity",   "FCVSX": "Fidelity",  "EAFVX": "Eaton Vance",   "EIFVX": "Eaton Vance", "DGIFX": "Disciplined Growth Investors",
    "AUERX": "Auer Growth",   "COAGX": "Caldwell & Orkin - Gator Capital L/S Fd",
    "TAVZX": "Third Avenue Value","TAVFX": "Third Avenue Value",  "TVFVX": "Third Avenue Value",  "ECFVX": "Eaton Vance",  "SGGDX": "First Eagle Gold",  "EICVX": "EIC", "EICIX": "EIC",
    "MBXAX": "Catalyst/Millburn Hedge Strategy Fund",  "UBVVX": "Undiscovered Managers",  "UBVAX": "Undiscovered Managers",  "UBVFX": "Undiscovered Managers",
    "MBXIX": "Catalyst/Millburn Hedge Strategy Fund",  "FEURX": "First Eagle Gold",
    "UBVRX": "Undiscovered Managers", "UBVTX": "Undiscovered Managers", "UBVUX": "Undiscovered Managers", "UBVSX": "Undiscovered Managers",  "DHTAX": "Diamond Hill Select Fund",
    "UBVLX": "Undiscovered Managers","UBVCX": "Undiscovered Managers", "MBXFX": "Catalyst/Millburn Hedge Strategy Fund", "MBXCX": "Catalyst/Millburn Hedge Strategy Fund", "DHTYX": "Diamond Hill Select Fund"
}


filtered_company_mapping = {
    symbol: sector for symbol, sector in company_mapping.items() if symbol in symbols
}


filtered_clusters_df = clusters_df[~clusters_df['Symbol'].isin(outlier_funds)]

company_df = pd.DataFrame(list(filtered_company_mapping.items()), columns=['Symbol', 'Company'])
merged_df_filtered = pd.merge(filtered_clusters_df, company_df, on='Symbol')

sector_composition_filtered = merged_df_filtered.groupby(['Cluster', 'Company']).size().unstack(fill_value=0)

sector_composition_filtered.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
plt.title('Company Composition within Clusters (Excluding Outliers)')
plt.xlabel('Cluster')
plt.ylabel('Number of funds')
plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

sector_totals = company_df['Company'].value_counts()
cluster_company_counts = merged_df_filtered.groupby(['Cluster', 'Company']).size().unstack(fill_value=0)
cluster_sector_percentages = cluster_company_counts.div(cluster_company_counts.sum(axis=1), axis=0) * 100
print(cluster_sector_percentages)


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming cluster_sector_percentages is defined as previously
plt.figure(figsize=(20, 10))  # Increase the width of the figure to give more room
heatmap = sns.heatmap(cluster_sector_percentages, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Cluster Composition by Company', fontsize=20)
plt.xlabel('Company', fontsize=14)
plt.ylabel('Cluster', fontsize=14)
plt.xticks(rotation=90, fontsize=10)  # Rotate labels to 90 degrees for better readability
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()  # Adjust layout to fit elements better
plt.show()


filtered_symbols = [symbol for symbol in symbols if symbol not in outlier_funds]
filtered_clusters_df = clusters_df[clusters_df['Symbol'].isin(filtered_symbols)].copy()


for symbol, df in dfs.items():
    df['Return'] = df['Adj Close'].pct_change()

intra_cluster_corr_filtered = {}
for cluster_num in filtered_clusters_df['Cluster'].unique():
    cluster_symbols = filtered_clusters_df[filtered_clusters_df['Cluster'] == cluster_num]['Symbol'].tolist()
    returns_df = pd.concat([dfs[symbol]['Return'] for symbol in cluster_symbols if symbol in dfs], axis=1)
    corr_matrix = returns_df.corr(method='pearson')
    mean_corr = corr_matrix.mean().mean()
    intra_cluster_corr_filtered[cluster_num] = mean_corr



#calculate intracluster metrics to unscaled data
from scipy.stats import linregress

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


# Extract the filtered symbols and their corresponding cluster labels.
filtered_symbols = filtered_clusters_df['Symbol'].tolist()
filtered_clusters = filtered_clusters_df['Cluster'].tolist()

# Adjusted cluster_purity function to directly use filtered data from DataFrame.
def cluster_purity(filtered_clusters, filtered_symbols, company_mapping):
    symbol_to_cluster = {symbol: cluster for symbol, cluster in zip(filtered_symbols, filtered_clusters)}
    cluster_sectors = defaultdict(Counter)

    for symbol in filtered_symbols:
        cluster_label = symbol_to_cluster[symbol]
        sector = company_mapping.get(symbol, "Unknown")  # Handle missing symbols safely
        cluster_sectors[cluster_label][sector] += 1

    total_correct = 0
    for sectors in cluster_sectors.values():
        most_common_sector_count = sectors.most_common(1)[0][1]
        total_correct += most_common_sector_count

    purity = total_correct / len(filtered_symbols)
    return purity


# Calculate purity using filtered data.
purity = cluster_purity(filtered_clusters, filtered_symbols, company_mapping)
print(f"Cluster Purity: {purity}")





