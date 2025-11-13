"""
Auto-generated script for an unsupervised learning task using the Wine dataset.
Each script is self-contained and uses sklearn.datasets.load_wine()
Run with: python3 <script_name>.py
"""
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import csv

X = load_wine().data
Xs = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=5, random_state=42)
Xp = pca.fit_transform(Xs)

# Clustering
k = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xp)
sil = silhouette_score(Xp, k.labels_)

print('End-to-end pipeline silhouette (PCA(5) + KMeans):', round(sil,4))
# Save simple report
with open('Q19_report.txt','w') as f:
    f.write(f'Silhouette:{sil}\nExplained_variance_ratio_sum:{pca.explained_variance_ratio_.sum()}\n')
print('Saved Q19_report.txt')