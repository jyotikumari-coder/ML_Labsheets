"""
Auto-generated script for an unsupervised learning task using the Wine dataset.
Each script is self-contained and uses sklearn.datasets.load_wine()
Run with: python3 <script_name>.py
"""
from sklearn.datasets import load_wine
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X = load_wine().data
Xs = StandardScaler().fit_transform(X)

model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(Xs)
print('Labels distribution:', {i: list(labels).count(i) for i in set(labels)})

plt.figure()
plt.scatter(Xs[:,0], Xs[:,1], c=labels)
plt.title('Agglomerative Clustering labels (2-feature projection)')
plt.savefig('Q03_agglomerative_labels.png')
print('Saved Q03_agglomerative_labels.png')