from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

data = load_iris().data
kmeans = KMeans(n_clusters=3, random_state=42).fit(data)
score = silhouette_score(data, kmeans.labels_)
print("Silhouette Score:", score)
