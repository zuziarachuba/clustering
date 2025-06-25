import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("winequality-red.csv", sep=';')
X = df.drop("quality", axis=1)
y = df["quality"]


X_scaled = StandardScaler().fit_transform(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
pca_df["quality"] = y


data = pca_df[['PC1', 'PC2']].T.values


n_clusters = 3
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    data, c=n_clusters, m=2.0, error=0.005, maxiter=1000, init=None)

cluster_labels = np.argmax(u, axis=0)


custom_colors = ["#1B5BA3", "#8b2655", "#36165e"] 
color_map = np.array(custom_colors)


plt.figure(figsize=(10, 6))
plt.scatter(
    pca_df['PC1'], pca_df['PC2'],
    c=color_map[cluster_labels],
    alpha=0.6
)

plt.title('')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.tight_layout()
plt.savefig("fcm_pca.png", dpi=300)
plt.show()
