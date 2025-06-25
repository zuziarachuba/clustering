import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA

def xie_beni_index_kmeans(X, labels, centers):
    """
    Oblicza wskaźnik Xie-Beni dla klasteryzacji KMeans.
    
    Parametry:
    - X: ndarray, dane wejściowe (n_samples, n_features)
    - labels: ndarray, etykiety klastrów (n_samples,)
    - centers: ndarray, współrzędne centroidów (n_clusters, n_features)

    Zwraca:
    - XB: float, wartość wskaźnika Xie-Beni
    """
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    intra_dist = 0.0
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        dists = np.linalg.norm(cluster_points - centers[i], axis=1)
        intra_dist += np.sum(dists ** 2)


    center_dists = pairwise_distances(centers)
    np.fill_diagonal(center_dists, np.inf)
    min_inter_dist = np.min(center_dists) ** 2


    XB = intra_dist / (n_samples * min_inter_dist)
    return XB



df = pd.read_csv("winequality-red.csv", sep=";")
X = df.drop("quality", axis=1).values
X_scaled = StandardScaler().fit_transform(X)

kmeans_full = KMeans(n_clusters=3, random_state=0).fit(X_scaled)
xb_full = xie_beni_index_kmeans(X_scaled, kmeans_full.labels_, kmeans_full.cluster_centers_)
print("XB (bez PCA):", round(xb_full, 3))

X_pca = PCA(n_components=2).fit_transform(X_scaled)
kmeans_pca = KMeans(n_clusters=3, random_state=0).fit(X_pca)
xb_pca = xie_beni_index_kmeans(X_pca, kmeans_pca.labels_, kmeans_pca.cluster_centers_)
print("XB (po PCA):", round(xb_pca, 3))
