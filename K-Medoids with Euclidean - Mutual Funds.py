
import sqlite3
import pandas as pd
import time
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator
from scipy.stats import kendalltau
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
import yfinance as yf
from scipy.cluster.hierarchy import fcluster
import seaborn as sns
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, jaccard_score

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


symbol_count = len(symbols)
print("Total number of symbols:", symbol_count)

unscaled_dfs = {}
dfs = {}
columns_to_exclude = ['High', 'Low' , 'Volume','Open','Close']

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

for i in range(len(symbols)):
    for j in range(i + 1, len(symbols)):
        symbol_i = symbols[i]
        symbol_j = symbols[j]

        time_series_data_i = dfs[symbol_i][columns_of_interest].values.flatten()
        time_series_data_j = dfs[symbol_j][columns_of_interest].values.flatten()

        distance = euclidean(time_series_data_i, time_series_data_j)

        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance



inertia_values = []
possible_clusters = range(1, 11)

for k in possible_clusters:
    kmedoids_kendall = KMedoids(n_clusters=k, metric="precomputed", random_state=42)
    kmedoids_kendall.fit(distance_matrix)
    inertia_values.append(kmedoids_kendall.inertia_)


kneedle = KneeLocator(possible_clusters, inertia_values, curve='convex', direction='decreasing')
optimal_clusters_hellinger = kneedle.knee
print(f"Optimal Number of Clusters (Knee Point): {optimal_clusters_hellinger}")

plt.figure(figsize=(10, 6))
plt.plot(possible_clusters, inertia_values, linestyle='-', color='b')
plt.title('Elbow Method', fontsize=20)
plt.xlabel('Number of Clusters (k)', fontsize=18)
plt.ylabel('Inertia', fontsize=18)
plt.show()




# perform MDS on the distance matrix
mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
reduced_data_dms = mds.fit_transform(distance_matrix)

reduced_data_with_clusters_dms = pd.DataFrame(reduced_data_dms, columns=['MDS1', 'MDS2'])

#scores before outlier detection

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

silhouette_scores_original = []
calinski_harabasz_scores_original = []
davies_bouldin_scores_original = []

for k in range(2, 30):
    kmedoids_dtw = KMedoids(n_clusters=k, random_state=42, metric='precomputed')
    labels_original = kmedoids_dtw.fit_predict(distance_matrix)

    silhouette_scores_original.append(silhouette_score(distance_matrix, labels_original, metric='precomputed'))
    calinski_harabasz_scores_original.append(calinski_harabasz_score(reduced_data_dms, labels_original))
    davies_bouldin_scores_original.append(davies_bouldin_score(reduced_data_dms, labels_original))

silhouette_scores_normalized_original = [score / max(silhouette_scores_original) for score in silhouette_scores_original]
calinski_harabasz_scores_normalized_original = [score / max(calinski_harabasz_scores_original) for score in calinski_harabasz_scores_original]
davies_bouldin_scores_normalized_original = [1 / (score + 1e-8) for score in davies_bouldin_scores_original]
davies_bouldin_scores_normalized_original = [score / max(davies_bouldin_scores_normalized_original) for score in davies_bouldin_scores_normalized_original]

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.plot(range(2, 30), silhouette_scores_normalized_original, label='Silhouette Score - Original', color='red')
plt.plot(range(2, 30), calinski_harabasz_scores_normalized_original, label='Calinski-Harabasz Index - Original', color='blue')
plt.plot(range(2, 30), davies_bouldin_scores_normalized_original, label='Inverted Davies-Bouldin Index - Original', color='green')

plt.legend()
plt.title('Clustering Evaluation Scores', fontsize=22)
plt.xlabel('Number of Clusters (k)', fontsize=18)
plt.ylabel('Normalized Scores', fontsize=18)
plt.show()




#EVALUATION SCORES
# Chosen number of clusters after evaluation
chosen_k = 5

# Assume euclidean_distance_matrix is your precomputed distance matrix
kmedoids = KMedoids(n_clusters=chosen_k, metric='precomputed', random_state=42)
kmedoids.fit(distance_matrix)
clusters = kmedoids.labels_  # This will give you the cluster labels

silhouette_avg = silhouette_score(distance_matrix, clusters, metric='precomputed')
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

from sklearn.manifold import MDS

mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
mds_result = mds.fit_transform(distance_matrix)

plt.scatter(mds_result[:, 0], mds_result[:, 1], c=clusters, cmap='viridis')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.title('MDS Clustering Results [K-Medoids with Euclidean Distances]')
plt.show()



# Assume dfs is a dictionary with dataframes for each mutual fund symbol
processed_symbols = list(dfs.keys())
fund_returns = np.array([dfs[symbol]['Return'].dropna().values for symbol in processed_symbols if 'Return' in dfs[symbol].columns])

def compute_distance_matrix(data):
    """ Computes the Euclidean distance matrix for the given data. """
    return squareform(pdist(data, 'euclidean'))

def temporal_cluster_validation(fund_data, window_size, step_size, num_clusters):
    num_samples = fund_data.shape[0]
    num_points = fund_data.shape[1]
    num_windows = (num_points - window_size) // step_size + 1
    cluster_labels_over_time = []

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window_data = fund_data[:, start_idx:end_idx]

        if window_data.shape[1] < window_size:
            continue  # Skip windows that do not have enough data

        flattened_data = window_data.reshape(window_data.shape[0], -1)
        distance_matrix = compute_distance_matrix(flattened_data)
        kmedoids = KMedoids(n_clusters=num_clusters, metric='precomputed', random_state=42)
        kmedoids.fit(distance_matrix)
        clusters = kmedoids.labels_

        cluster_labels_over_time.append(clusters)

    return np.array(cluster_labels_over_time).T  # Transpose to make rows correspond to mutual funds

# Set parameters for the temporal validation
num_clusters = 5  # Adjust the number of clusters based on your specific analysis
window_size = 26  # Using half-yearly data as the window size
step_size = 13    # Overlap of half a year

# Calculate cluster labels over time
cluster_labels_matrix = temporal_cluster_validation(fund_returns, window_size, step_size, num_clusters)

# Plotting and analysis functions
def compute_temporal_ari(cluster_labels_matrix):
    num_windows = cluster_labels_matrix.shape[1]
    ari_scores = []

    for i in range(num_windows - 1):
        ari = adjusted_rand_score(cluster_labels_matrix[:, i], cluster_labels_matrix[:, i + 1])
        ari_scores.append(ari)

    return ari_scores

ari_scores = compute_temporal_ari(cluster_labels_matrix)

plt.figure(figsize=(10, 6))
plt.plot(range(len(ari_scores)), ari_scores, linestyle='-')
plt.title('Adjusted Rand Index Over Time', fontsize=22)
plt.xlabel('Time Window Index')
plt.ylabel('Adjusted Rand Index')
plt.grid(True)
plt.show()

def compute_temporal_jaccard(cluster_labels_matrix):
    num_windows = cluster_labels_matrix.shape[1]
    jaccard_scores = []

    for i in range(num_windows - 1):
        jaccard = jaccard_score(cluster_labels_matrix[:, i], cluster_labels_matrix[:, i + 1], average='macro')
        jaccard_scores.append(jaccard)

    return jaccard_scores

jaccard_scores = compute_temporal_jaccard(cluster_labels_matrix)

plt.figure(figsize=(10, 6))
plt.plot(range(len(jaccard_scores)), jaccard_scores, linestyle='-')
plt.title('Jaccard Index Over Time', fontsize=22)
plt.xlabel('Time Window Index')
plt.ylabel('Jaccard Index')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(cluster_labels_matrix, annot=False, cmap='viridis', cbar_kws={'label': 'Cluster Label'})
plt.title('Cluster Assignments Over Time Windows', fontsize=22)
plt.xlabel('Time Window Index')
plt.ylabel('Fund Index')
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

# Calculate the ARI scores over time
ari_scores = compute_temporal_ari(cluster_labels_matrix)

# Compute the average ARI if there are any scores
if ari_scores:
    average_ari = sum(ari_scores) / len(ari_scores)
    print("Average Adjusted Rand Index (ARI):", average_ari)
else:
    print("No ARI scores available to average.")





import matplotlib.pyplot as plt
import seaborn as sns

num_windows = cluster_labels_matrix.shape[1]  # Number of time windows

fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Adjust the figsize to fit the aspect ratio and size you desire

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

plt.figure(figsize=(8, 6))  # Use equal width and height for a square plot
scatter = plt.scatter(mds_result[:, 0], mds_result[:, 1], c=clusters, cmap=custom_cmap, edgecolor='none', s=70)
plt.xlabel('MDS Dimension 1', fontsize=14)
plt.ylabel('MDS Dimension 2', fontsize=14)
plt.title('MDS Clustering Results [K-Medoids with Euclidean]', fontsize=16)
plt.grid(True)
plt.axis('equal')
plt.show()



for cluster_num in sorted(clusters_df['Cluster'].unique()):
    cluster_funds = clusters_df[clusters_df['Cluster'] == cluster_num]['Symbol'].tolist()
    print(f"Cluster {cluster_num}: {', '.join(cluster_funds)}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")


#outlier exclusion

min_cluster_size = 2
cluster_sizes = clusters_df['Cluster'].value_counts()
outlier_clusters = cluster_sizes[cluster_sizes < min_cluster_size].index

outlier_funds = []
for cluster in outlier_clusters:
    funds_in_cluster = clusters_df[clusters_df['Cluster'] == cluster]['Symbol'].tolist()
    outlier_funds.extend(funds_in_cluster)

outlier_indices = np.array([symbols.index(fund) for fund in outlier_funds]).astype(int)


filtered_distance_matrix = np.delete(distance_matrix, outlier_indices, axis=0)
filtered_distance_matrix = np.delete(filtered_distance_matrix, outlier_indices, axis=1)

filtered_clusters = np.delete(clusters, outlier_indices)
filtered_clusters, _ = pd.factorize(filtered_clusters)

mds = MDS(n_components=4, dissimilarity='precomputed', random_state=42)
mds_result_filtered = mds.fit_transform(filtered_distance_matrix)


print("Outlier Funds:")
for fund in outlier_funds:
    print(fund)





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



#calculate to unscaled data

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





from collections import defaultdict, Counter

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




import matplotlib.pyplot as plt
import numpy as np

company_counts = merged_df_filtered.groupby(['Cluster', 'Company']).size().unstack(fill_value=0)
cluster_totals = company_counts.sum(axis=1)
company_percentage = company_counts.div(cluster_totals, axis=0) * 100

plt.figure(figsize=(14, 8))
ax = company_percentage.plot(kind='bar', stacked=True, colormap='viridis', figsize=(14, 8))
ax.set_title('Cluster Composition by Company - [K-Medoids - Euclidean]', fontsize=22)
ax.set_xlabel('Cluster', fontsize=18)
ax.set_ylabel('Percentage of Companies', fontsize=18)
ax.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xticks(rotation=0, fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

plt.show()




