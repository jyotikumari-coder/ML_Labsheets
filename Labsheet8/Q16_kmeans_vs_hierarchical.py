"""
Auto-generated script for an unsupervised learning task using the Wine dataset.
Each script is self-contained and uses sklearn.datasets.load_wine()
Run with: python3 <script_name>.py
"""
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt

X = load_wine().data
Xs = StandardScaler().fit_transform(X)

k = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)
a = AgglomerativeClustering(n_clusters=3).fit(Xs)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.scatter(Xs[:,0], Xs[:,1], c=k.labels_)
plt.title('KMeans (first two features)')
plt.subplot(1,2,2)
plt.scatter(Xs[:,0], Xs[:,1], c=a.labels_)
plt.title('Agglomerative (first two features)')
plt.savefig('Q16_kmeans_vs_hierarchical.png')
print('Saved Q16_kmeans_vs_hierarchical.png')