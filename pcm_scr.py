import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pcm import pcm  # zakładam, że Twoja funkcja pcm jest w pliku pcm.py

# Wczytaj dane
df = pd.read_csv("winequality-red.csv", sep=';')
X = df.drop("quality", axis=1)
y = df["quality"]

# Standaryzacja
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

n_clusters = 3
cntr, U, T, obj_fcn = pcm(X_pca, c=n_clusters, expo=2, max_iter=300, min_impro=0.005, a=1, b=4, nc=3)


labels = np.argmax(U, axis=0)

# Kolory pastelowe
colors = ["#1B5BA3", "#8b2655", "#36165e"] 

# Wykres
plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    plt.scatter(
        X_pca[labels == i, 0], X_pca[labels == i, 1],
        c=colors[i],
        label=f'Grupa {i+1}',
        alpha=0.6
    )

plt.title("")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.savefig("pcm_pca.png", dpi=300)
plt.show()
