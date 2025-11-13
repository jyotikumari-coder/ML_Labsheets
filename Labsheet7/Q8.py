from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

data = load_iris().data
kmeans_raw = KMeans(n_clusters=3).fit(data)
raw_score = silhouette_score(data, kmeans_raw.labels_)

scaled = StandardScaler().fit_transform(data)
kmeans_scaled = KMeans(n_clusters=3).fit(scaled)
scaled_score = silhouette_score(scaled, kmeans_scaled.labels_)

print("Before Scaling:", raw_score)
print("After Scaling:", scaled_score)
