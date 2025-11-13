import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

data = load_iris().data
scores = []

for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42).fit(data)
    scores.append(silhouette_score(data, km.labels_))

plt.plot(range(2, 11), scores, marker='o')
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.show()
