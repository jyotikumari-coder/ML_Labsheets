from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

methods = ['ward', 'complete', 'average']
data = load_iris().data

for method in methods:
    plt.figure()
    dendrogram(linkage(data, method=method))
    plt.title(f"Dendrogram - {method} linkage")
    plt.show()
