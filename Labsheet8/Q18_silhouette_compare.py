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

X = load_wine().data
Xs = StandardScaler().fit_transform(X)

k_orig = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)
s_orig = silhouette_score(Xs, k_orig.labels_)

Xp = PCA(n_components=2, random_state=42).fit_transform(Xs)
k_pca = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xp)
s_pca = silhouette_score(Xp, k_pca.labels_)

print('Silhouette original scaled:', round(s_orig,4))
print('Silhouette PCA(2):', round(s_pca,4))