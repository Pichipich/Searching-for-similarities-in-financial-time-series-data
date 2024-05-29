
import matplotlib.pyplot as plt
from tslearn.metrics import dtw as ts_dtw
import sqlite3
import pandas as pd
import time
from sklearn.manifold import MDS
from scipy.stats import linregress, zscore
from kneed import KneeLocator
from sklearn_extra.cluster import KMedoids
import numpy as np


symbols = [
    "RYPMX", "RYMPX", "RYMNX", "RYZCX", "ENPIX", "ENPSX", "FNARX", "JMCGX", "JMIGX",
    "FFGIX", "FFGCX", "FCGCX", "FIQRX", "FFGTX", "FFGAX", "SGGDX", "FEGIX", "KINCX",
    "WWWFX", "FEGOX", "KINAX", "FEURX", "KMKNX", "KMKCX",   "KMKYX",
    "GOFIX", "KMKAX", "GOVIX",  "FAGNX", "FIKAX", "FSENX", "FANIX", "FANAX",
    "KNPAX", "FNRCX", "KNPYX", "WWNPX", "KNPCX", "FKRCX", "FGPMX", "FGADX", "BIPIX",
    "BIPSX", "LSHEX", "KSCYX", "KSOCX", "KSCOX", "KSOAX", "LSHAX", "LSHUX", "LSHCX",
    "COBYX", "HICGX", "HFCGX", "MCMVX", "FGFLX", "FGFRX", "FGRSX", "FGFAX", "MOWNX",
    "MOWIX", "FGFCX", "RCMFX", "FSHOX", "TVFVX", "TAVZX", "TAVFX", "RMLPX", "TCMSX",
     "SVFAX", "SVFKX", "SVFDX", "SVFFX", "SMVLX", "SVFYX", "BIVRX",
    "HWAAX", "SSSIX", "HWAIX",  "BIVIX", "HWACX", "SSSFX", "UMPSX",
    "UMPIX", "FVIFX", "BGRSX",  "FTVFX", "HIMDX", "BGLSX", "CSERX", "FAVFX", "HFMDX", "SLVRX",
    "CPLSX", "CSRYX", "FVLZX", "SVLCX", "FDVLX", "FDVLX", "AUERX","FVLKX", "SLVAX", "CSVZX",
    "SLVIX", "CPCLX", "JORFX", "FSRPX", "YAFIX", "JORCX",
    "MBXIX",  "CPLIX", "MBXCX", "TGIRX", "CSVAX", "CSVFX", "THVRX", "THGCX", "CGOLX",
    "TGVRX", "JANRX", "JSLNX", "JORNX", "JORAX", "FUGAX", "JORIX", "YACKX", "YAFFX", "SNOIX",
     "BUFOX", "FUGCX",  "FIKIX", "FUGIX", "TIVRX", "TGVIX", "FSUTX",
    "SNOCX", "FAUFX", "HWLCX", "CSGRX", "TGVAX", "CADPX", "HWLAX", "FMDCX", "CLSYX", "SNOAX",
    "DSCPX", "HWLIX", "CSRCX", "JORRX", "HWCIX", "HULEX", "HULIX", "HWSAX", "FSTRX", "QRLVX",
    "FMSTX", "QCLVX", "FSTKX", "FSTLX", "HWSCX", "DHLAX", "DHLRX", "DHLYX", "HWCAX",
    "HWSIX", "HWCCX", "FMCRX", "FMCLX", "BLUEX", "TBGVX", "NECOX", "CSVRX", "NEOYX", "CSMIX",
    "TFIFX", "NOANX", "PRISX", "CVVRX", "CUURX", "CSCZX", "CSVYX", "FIJCX", "FSPCX", "FDIGX",
    "FDTGX", "FDFAX","TBWIX", "DODWX", "MRFOX", "FCVTX",  "FSLBX", "EIPFX", "FEVCX", "WBSNX", "CMIDX",
    "NEFJX", "THOIX", "FCVCX", "FCVAX", "IMIDX"

]

conn = sqlite3.connect('mutual_data.db')

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

# Calculate DTW distance matrix
dtw_distance_matrix = np.zeros((num_symbols, num_symbols))

for i in range(num_symbols):
    for j in range(i + 1, num_symbols):
        distance = ts_dtw(equal_length_data[i], equal_length_data[j])
        dtw_distance_matrix[i, j] = dtw_distance_matrix[j, i] = distance

# Perform clustering using KMedoids
inertia_values = []
possible_clusters = range(1, 11)

for k in possible_clusters:
    kmedoids = KMedoids(n_clusters=k, metric="precomputed", random_state=42)
    kmedoids.fit(dtw_distance_matrix)
    inertia_values.append(kmedoids.inertia_)

# Find the optimal number of clusters using KneeLocator
kneedle = KneeLocator(possible_clusters, inertia_values, curve='convex', direction='decreasing')
optimal_clusters = kneedle.knee
print(f"Optimal Number of Clusters (Knee Point): {optimal_clusters}")

plt.figure(figsize=(10, 6))
plt.plot(possible_clusters, inertia_values, linestyle='-', color='b')
plt.title('Elbow Method', fontsize=20)
plt.xlabel('Number of Clusters (k)', fontsize=18)
plt.ylabel('Inertia', fontsize=18)
plt.show()


# perform MDS on the distance matrix
mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
reduced_data_dms = mds.fit_transform(dtw_distance_matrix)

reduced_data_with_clusters_dms = pd.DataFrame(reduced_data_dms, columns=['MDS1', 'MDS2'])

#scores before outlier detection

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

silhouette_scores_original = []
calinski_harabasz_scores_original = []
davies_bouldin_scores_original = []

for k in range(2, 20):
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
plt.plot(range(2, 20), silhouette_scores_normalized_original, label='Silhouette Score - Original', color='red')
plt.plot(range(2, 20), calinski_harabasz_scores_normalized_original, label='Calinski-Harabasz Index - Original', color='blue')
plt.plot(range(2, 20), davies_bouldin_scores_normalized_original, label='Inverted Davies-Bouldin Index - Original', color='green')

plt.legend()
plt.title('Clustering Quality Scores by Number Of Clusters', fontsize=22)
plt.xlabel('Number of Clusters (k)', fontsize=18)
plt.ylabel('Normalized Scores', fontsize=18)
plt.show()







#EVALUATION SCORES
# Chosen number of clusters after evaluation
chosen_k = 5

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

from sklearn.manifold import MDS

mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
mds_result = mds.fit_transform(dtw_distance_matrix)

plt.scatter(mds_result[:, 0], mds_result[:, 1], c=clusters, cmap='viridis')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.title('MDS Clustering Results [K-Medoids with DTW]')
plt.show()





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.metrics import dtw

from sklearn.metrics import adjusted_rand_score, jaccard_score

# Assume dfs is a dictionary with dataframes for each stock symbol
processed_symbols = list(dfs.keys())
stock_returns = np.array([dfs[symbol]['Return'].dropna().values for symbol in processed_symbols if 'Return' in dfs[symbol].columns])

def compute_distance_matrix(data):
    num_samples = data.shape[0]
    distance_matrix = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            # Calculate DTW distance between each pair of sequences
            dist = dtw(data[i], data[j])
            distance_matrix[i, j] = distance_matrix[j, i] = dist

    return distance_matrix

def temporal_cluster_validation(stock_data, window_size, step_size, num_clusters):
    num_samples = stock_data.shape[0]
    num_points = stock_data.shape[1]
    num_windows = (num_points - window_size) // step_size + 1
    cluster_labels_over_time = []

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window_data = stock_data[:, start_idx:end_idx]

        if window_data.shape[1] < window_size:
            continue  # Skip windows that do not have enough data

        flattened_data = window_data.reshape(window_data.shape[0], -1)
        distance_matrix = compute_distance_matrix(flattened_data)
        kmedoids = KMedoids(n_clusters=num_clusters, metric='precomputed', random_state=42)
        kmedoids.fit(distance_matrix)
        clusters = kmedoids.labels_

        cluster_labels_over_time.append(clusters)

    return np.array(cluster_labels_over_time).T  # Transpose to make rows correspond to stocks

# Set parameters for the temporal validation
num_clusters = 3  # Adjust the number of clusters based on your specific analysis
window_size = 26  # Using half-yearly data as the window size
step_size = 13    # Overlap of half a year

# Calculate cluster labels over time
cluster_labels_matrix = temporal_cluster_validation(stock_returns, window_size, step_size, num_clusters)

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
plt.axis('equal')
plt.show()

for cluster_num in sorted(clusters_df['Cluster'].unique()):
    cluster_stocks = clusters_df[clusters_df['Cluster'] == cluster_num]['Symbol'].tolist()
    print(f"Cluster {cluster_num}: {', '.join(cluster_stocks)}")

























#OUTLIERS

min_cluster_size = 2
cluster_sizes = clusters_df['Cluster'].value_counts()
outlier_clusters = cluster_sizes[cluster_sizes < min_cluster_size].index

outlier_stocks = []
for cluster in outlier_clusters:
    stocks_in_cluster = clusters_df[clusters_df['Cluster'] == cluster]['Symbol'].tolist()
    outlier_stocks.extend(stocks_in_cluster)

outlier_indices = np.array([symbols.index(stock) for stock in outlier_stocks]).astype(int)


filtered_distance_matrix = np.delete(dtw_distance_matrix, outlier_indices, axis=0)
filtered_distance_matrix = np.delete(filtered_distance_matrix, outlier_indices, axis=1)

filtered_clusters = np.delete(clusters, outlier_indices)
filtered_clusters, _ = pd.factorize(filtered_clusters)

mds = MDS(n_components=4, dissimilarity='precomputed', random_state=42)
mds_result_filtered = mds.fit_transform(filtered_distance_matrix)


print("Outlier Stocks:")
for stock in outlier_stocks:
    print(stock)



company_mapping = {

    "ZGFIX": "Ninety One",   "FDVLX": "Fidelity",  "GNSRX": "abrdn",  "FSLBX": "Fidelity",  "CRIMX": "CRM",  "NWKCX": "Nationwide",   "NWHZX": "Nationwide",
    "GGEAX": "Nationwide", "RRTRX": "T. Rowe Price",
    "DODGX": "Dodge & Cox",  "FGIOX": "Fidelity",   "FERIX": "Fidelity",   "GWGVX": "AMG",  "PSCZX": "PGIM",  "FEATX": "Fidelity",  "CWCFX": "Christopher Weil & Co",
    "GWGZX": "AMG", "JSCRX": "PGIM",  "PGOAX": "PGIM",  "VSEQX": "Vanguard",  "TLVCX": "Timothy",
    "FSHCX": "Fidelity",  "PSCCX": "PGIM",  "GSXIX": "abrdn",  "CCVAX": "Calvert",  "FSEAX": "Fidelity",  "FEAAX": "Fidelity",
    "PSCJX": "PGIM", "BRWIX": "AMG",  "GSCIX": "abrdn", "LCIAX": "SEI", "GSXAX": "abrdn", "CSVIX": "Calvert", "JDMNX": "Janus Henderson",   "TLVAX": "Timothy", "JANEX": "Janus Henderson",
    "DHLSX": "Diamond Hill",  "FERCX": "Fidelity",  "FIJPX": "Fidelity", "FIQPX": "Fidelity",  "FGIZX": "Fidelity",   "TNBRX": "1290 SmartBeta",
    "JGRTX": "Janus Henderson", "DIAYX": "Diamond Hill", "JAENX": "Janus Henderson", "JDMAX": "Janus Henderson",  "SGIIX": "First Eagle",
    "PWJQX": "PGIM", "FEGRX": "First Eagle",  "FAMRX": "Fidelity",  "TNBIX": "1290 SmartBeta","JMGRX": "Janus Henderson",
    "PWJRX": "PGIM", "DIAMX": "Diamond Hill", "HWCAX": "Hotchkis & Wiley",  "LGMAX": "Loomis Sayles",  "TAMVX": "T. Rowe Price", "TMVIX": "Timothy",
    "CRMMX": "CRM", "FZAJX": "Fidelity", "CSCCX": "Calvert", "FESGX": "First Eagle", "FIFFX": "Fidelity",  "WSMNX": "William Blair",   "GSXCX": "abrdn", "JDMRX": "Janus Henderson",
    "MSFLX": "Morgan Stanley", "JGRCX": "Janus Henderson","VHCOX": "Vanguard",  "FVIFX": "Fidelity",  "FVLZX": "Fidelity",  "RRTDX": "T. Rowe Price",  "MSFBX": "Morgan Stanley",   "TRSSX": "T. Rowe Price","SGENX": "First Eagle",
    "THOFX": "Thornburg",  "LGMCX": "Loomis Sayles", "LSWWX": "Loomis Sayles",  "LGMNX": "Loomis Sayles",
    "THOGX": "Thornburg",  "GCPNX": "Gateway",  "PWJBX": "PGIM",  "RPMAX": "Reinhart",  "PWJAX": "PGIM",   "FNSDX": "Fidelity", "FDEEX": "Fidelity",  "FGRIX": "Fidelity", "CCGSX": "Baird", "FEYCX": "Fidelity",
    "PSCHX": "PGIM", "PWJCX": "PGIM", "VHCAX": "Vanguard", "MSGFX": "Morgan Stanley",  "PARDX": "T. Rowe Price",
    "OTCFX": "T. Rowe Price",  "HMXIX": "AlphaCentric",  "FTVFX": "Fidelity",  "OTIIX": "T. Rowe Price", "FCPCX": "Fidelity", "ZGFAX": "Ninety One",
    "FJPNX": "Fidelity",   "NWAMX": "Nationwide",  "LCORX": "Leuthold Core",   "PWJDX": "PGIM",   "CCGIX": "Baird",  "PASSX": "T. Rowe Price",  "FEYTX": "Fidelity",  "THORX": "Thornburg",  "AFCSX": "American Century",  "FCPAX": "Fidelity",  "FAVFX": "Fidelity",   "LCRIX": "Leuthold","TRRDX": "T. Rowe Price",   "THOVX": "Thornburg",
    "FVLKX": "Fidelity", "MSFAX": "Morgan Stanley", "MGISX": "Morgan Stanley", "PWJZX": "PGIM", "NEFOX": "Natixis",
    "AFCMX": "American Century",    "FEYAX": "Fidelity",   "HWCIX": "Hotchkis & Wiley",   "GWGIX": "AMG",
    "SIBAX": "Sit Balanced",  "FGIKX": "Fidelity",  "FIVFX": "Fidelity",  "FCVFX": "Fidelity", "THOAX": "Thornburg",  "NECOX": "Natixis Oakmark",
    "TRMIX": "T. Rowe Price",  "WSMDX": "William Blair",  "WBSIX": "William Blair",  "HWCCX": "Hotchkis & Wiley",  "FEYIX": "Fidelity",  "TRMCX": "T. Rowe Price",  "SLVAX": "Columbia",
    "THOIX": "Thornburg","HMXAX": "AlphaCentric","FIATX": "Fidelity",   "AFCHX": "American Century",   "HMXCX": "AlphaCentric", "VPCCX": "Vanguard",    "WBSNX": "William Blair",
    "CSVZX": "Columbia",  "FITGX": "Fidelity",  "RRMVX": "T. Rowe Price",  "FCPIX": "Fidelity",    "FIAGX": "Fidelity",  "FIDZX": "Fidelity", "HWLCX": "Hotchkis & Wiley",
    "FTFFX": "Fidelity",  "DFSGX": "DF Dent",  "SLVRX": "Columbia",  "FIIIX": "Fidelity",   "SPINX": "SEI",   "THOCX": "Thornburg",
    "CSERX": "Columbia",    "HWLAX": "Hotchkis & Wiley",    "NEOYX": "Natixis",    "FIGFX": "Fidelity",    "SVLCX": "Columbia",
    "CSRYX": "Columbia",  "NOANX": "Natixis",  "SSQSX": "State Street",  "HWLIX": "Hotchkis & Wiley",  "SIVIX": "State Street",  "FAFFX": "Fidelity",
    "CAMWX": "Cambiar",  "FGTNX": "Fidelity",  "CAMOX": "Cambiar",  "AFCWX": "American Century",  "PORIX": "Trillium",  "BGRSX": "Boston Partners",
    "FKGLX": "Fidelity",  "FIGCX": "Fidelity",  "AFVZX": "Applied Finance",  "PORTX": "Trillium",  "FOSFX": "Fidelity",  "AASMX": "Thrivent",  "SLVIX": "Columbia",
    "RPGIX": "T. Rowe Price", "AFCLX": "American Century", "TRGAX": "T. Rowe Price", "FPJAX": "Fidelity",  "BGLSX": "Boston Partners",  "LSOFX": "LS Opportunity",
    "TILCX": "T. Rowe Price",  "FIQLX": "Fidelity",  "AFCNX": "American Century",  "VADFX": "Invesco",  "VADAX": "Invesco",  "POAGX": "PRIMECAP",  "FOSKX": "Fidelity",  "USPCX": "Union Street Partners", "FJPIX": "Fidelity",
    "QKBGX": "Federated",  "ABLOX": "Alger",  "FCFFX": "Fidelity",  "VADRX": "Invesco",  "DFDSX": "DF Dent",  "USPFX": "Union","VADDX": "Invesco",
    "PURRX": "PGIM",  "GWEIX": "AMG",  "VADCX": "Invesco",  "GWEZX": "AMG",  "BSGSX": "Baird",   "PVFAX": "Paradigm Value",
    "AAUTX": "Thrivent", "OLVAX": "JPMorgan", "OLVRX": "JPMorgan", "PGRQX": "PGIM", "PURZX": "PGIM", "TRRJX": "T. Rowe Price", "RCMFX": "Schwartz",
    "MUNDX": "Mundoval",  "TLVIX": "Thrivent",  "BSGIX": "Baird",  "CBLRX": "Columbia",  "GWETX": "AMG","VDIGX": "Vanguard",   "ECSTX": "Eaton Vance",  "SSSIX": "SouthernSun",
    "VPMAX": "Vanguard",  "CBDYX": "Columbia",  "OLVCX": "JPMorgan",  "USPVX": "Union Street Partners",  "VGSAX": "Virtus Duff & Phelps",  "VPMCX": "Vanguard",
    "FGABX": "Fidelity", "TSCSX": "Thrivent", "FJPCX": "Fidelity", "QCBGX": "Hermes", "JEQIX": "Johnson",
    "BLUEX": "AMG", "SSSFX": "SouthernSun", "CBALX": "Columbia", "VRGEX": "Virtus", "CLREX": "Columbia",    "VGISX": "Virtus Duff & Phelps","CBLAX": "Columbia",  "PURCX": "PGIM",  "EXHAX": "Manning & Napier",   "VLSIX": "Virtus",   "MNHIX": "Manning & Napier",
    "FJPTX": "Fidelity",  "OLVTX": "JPMorgan",  "CBDRX": "Columbia",  "QABGX": "Hermes",
    "HLQVX": "JPMorgan", "RRTPX": "T. Rowe Price",   "PACLX": "T. Rowe Price", "QIBGX": "Hermes",   "NRGSX": "Neuberger Berman",  "NBGIX": "Neuberger Berman", "NBGAX": "Neuberger Berman",
    "JLVMX": "JPMorgan", "COAGX": "Caldwell & Orkin", "VGSCX": "Virtus", "JLVZX": "JPMorgan", "ERSTX": "Eaton Vance",   "JLVRX": "JPMorgan",
    "NEAGX": "Needham","CBLCX": "Columbia","EHSTX": "Eaton Vance","PRWCX": "T. Rowe Price","TRAIX": "T. Rowe Price","PARKX": "T. Rowe Price",   "SEVSX": "Guggenheim",   "EILVX": "Eaton Vance",    "ERLVX": "Eaton Vance","NBGEX": "Neuberger Berman",
    "PURAX": "PGIM",  "DREGX": "Driehaus",  "SEVPX": "Guggenheim", "LKBAX": "LKCM",  "NBGNX": "Neuberger",
    "QLEIX": "AQR",  "VLSCX": "Virtus",  "PUREX": "PGIM",  "PCAFX": "Prospector",  "PURGX": "PGIM",  "NEAIX": "Needham", "VSTCX": "Vanguard", "AGVDX": "American Funds", "CSRIX": "Cohen & Steers",
    "CGVBX": "American Funds ", "SEVAX": "Guggenheim", "QLERX": "AQR", "CGVEX": "American Funds", "AGVFX": "American Funds","AGVEX": "American Funds", "CGVYX": "American Funds","RGLEX": "American Funds ",
    "HHDFX": "Hamlin", "FOBPX": "Tributary", "HHDVX": "Hamlin", "CSJCX": "Cohen & Steers", "FCGCX": "Fidelity", "WCMSX": "WCM","CSJIX": "Cohen & Steers",
    "CSRSX": "Cohen & Steers",  "CSJAX": "Cohen & Steers",  "CSJRX": "Cohen & Steers", "CSJZX": "Cohen & Steers", "FFGTX": "Fidelity", "MNHCX": "Manning & Napier","FOBAX": "Tributary","MNHRX": "Manning & Napier","GQGPX": "GQG Partners","PHRAX": "Virtus","VRREX": "Virtus","GQGIX": "GQG Partners",   "GQGRX": "GQG Partners",
    "FMIJX": "FMI",  "VLSAX": "Virtus",  "JDBAX": "Janus Henderson","GURCX": "Guggenheim", "FMIYX": "FMI",   "JABAX": "Janus Henderson",
    "JABNX": "Janus Henderson", "JBALX": "Janus Henderson", "SCVEX": "Hartford", "DIEMX": "Driehaus", "GURAX": "Guggenheim", "GURPX": "Guggenheim",  "VLSRX": "Virtus", "ICSIX": "Dynamic",  "ICSNX": "Dynamic",  "RYAVX": "Rydex",
    "EAFVX": "Eaton Vance",  "RYMVX": "Rydex", "VASGX": "Vanguard",  "GTSCX": "Glenmede",  "GURIX": "Guggenheim",  "EIFVX": "Eaton Vance",  "RAIWX": "Manning & Napier",  "JABCX": "Janus Henderson",
    "BBHLX": "BBH Partner Fund", "RYMMX": "Rydex", "RAIRX": "Manning & Napier",  "JDBRX": "Janus Henderson",  "UGTCX": "Victory Growth and Tax",
    "BTBFX": "Boston Trust Asset Management","JABRX": "Janus Henderson",   "UGTAX": "Victory Growth and Tax",  "UGTIX": "Victory Growth and Tax",  "JANBX": "Janus Henderson",  "SEBLX": "Touchstone",  "SBACX": "Touchstone",  "FSCRX": "Fidelity","NEEGX": "Needham"

}




filtered_company_mapping = {
    symbol: sector for symbol, sector in company_mapping.items() if symbol in symbols
}


filtered_clusters_df = clusters_df[~clusters_df['Symbol'].isin(outlier_stocks)]

company_df = pd.DataFrame(list(filtered_company_mapping.items()), columns=['Symbol', 'Company'])
merged_df_filtered = pd.merge(filtered_clusters_df, company_df, on='Symbol')

sector_composition_filtered = merged_df_filtered.groupby(['Cluster', 'Company']).size().unstack(fill_value=0)

sector_composition_filtered.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
plt.title('Company Composition within Clusters (Excluding Outliers)')
plt.xlabel('Cluster')
plt.ylabel('Number of Stocks')
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
plt.title('Cluster Composition by Company [K-Medoids - DTW]', fontsize=20)
plt.xlabel('Company', fontsize=14)
plt.ylabel('Cluster', fontsize=14)
plt.xticks(rotation=90, fontsize=10)  # Rotate labels to 90 degrees for better readability
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()  # Adjust layout to fit elements better
plt.show()



filtered_symbols = [symbol for symbol in symbols if symbol not in outlier_stocks]
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

# Plot
plt.figure(figsize=(14, 8))  # Increase figure size for better visibility in thesis
ax = company_percentage.plot(kind='bar', stacked=True, colormap='viridis', figsize=(14, 8))
ax.set_title('Cluster Composition by Company - [K-Medoids - DTW]', fontsize=22)
ax.set_xlabel('Cluster', fontsize=18)
ax.set_ylabel('Percentage of Companies', fontsize=18)
ax.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')

# Improve layout and font sizes for readability
plt.xticks(rotation=0, fontsize=14)  # Rotate x-labels for better readability
plt.yticks(fontsize=14)
plt.tight_layout()  # Adjust layout to make room for legend

plt.show()