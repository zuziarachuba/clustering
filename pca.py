import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Wczytaj dane
df = pd.read_csv("winequality-red.csv", sep=';')
X = df.drop("quality", axis=1)
y = df["quality"]

# Standaryzuj dane
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# DataFrame z wynikami
df_pca = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
df_pca["quality"] = y

# Wykres
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="quality", palette="coolwarm", s=50)
plt.title("")
plt.xlabel("PC1 (28,2% wyjaśnianej wariancji)")
plt.ylabel("PC2 ( 17,5% wyjaśnianej wariancji)")
plt.legend(title="Jakość wina")
plt.tight_layout()
plt.savefig("pca_quality.png", dpi=300)
plt.show()
