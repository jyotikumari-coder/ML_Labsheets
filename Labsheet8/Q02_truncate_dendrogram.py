"""
Auto-generated script for an unsupervised learning task using the Wine dataset.
Each script is self-contained and uses sklearn.datasets.load_wine()
Run with: python3 <script_name>.py
"""
from sklearn.datasets import load_wine
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

X = load_wine().data
Z = linkage(X, method='ward')

# Truncate and display last p merges (p controls level)
plt.figure(figsize=(8,4))
dendrogram(Z, truncate_mode='lastp', p=12, show_leaf_counts=True)
plt.title('Truncated dendrogram (p=12)')
plt.savefig('Q02_truncated_dendrogram.png')
print('Saved Q02_truncated_dendrogram.png')

# Example: form clusters by cutting at maxclust=3
labels = fcluster(Z, t=3, criterion='maxclust')
print('Cluster counts (truncated cut to 3):', {i: list(labels).count(i) for i in set(labels)})