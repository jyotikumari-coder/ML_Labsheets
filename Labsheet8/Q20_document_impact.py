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
import matplotlib.pyplot as plt

X = load_wine().data
Xs = StandardScaler().fit_transform(X)

components = list(range(2, 11))
sils = []
explained = []
for n in components:
    p = PCA(n_components=n, random_state=42).fit(Xs)
    Xp = p.transform(Xs)
    k = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xp)
    sils.append(silhouette_score(Xp, k.labels_))
    explained.append(p.explained_variance_ratio_.sum())

plt.figure()
plt.plot(components, sils, marker='o', label='Silhouette')
plt.plot(components, explained, marker='x', label='Cumulative Explained Variance')
plt.xlabel('n PCA components')
plt.legend()
plt.title('Impact of PCA dimensionality on clustering quality & explained variance')
plt.savefig('Q20_impact_plot.png')
print('Saved Q20_impact_plot.png')