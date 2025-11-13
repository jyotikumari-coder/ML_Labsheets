"""
Auto-generated script for an unsupervised learning task using the Wine dataset.
Each script is self-contained and uses sklearn.datasets.load_wine()
Run with: python3 <script_name>.py
"""
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import numpy as np

data = load_wine()
X = data.data

# Unscaled linkage
Z_unscaled = linkage(X, method='ward')
labels_unscaled = fcluster(Z_unscaled, t=3, criterion='maxclust')

# Scaled linkage
X_scaled = StandardScaler().fit_transform(X)
Z_scaled = linkage(X_scaled, method='ward')
labels_scaled = fcluster(Z_scaled, t=3, criterion='maxclust')

print("Unique clusters (unscaled):", np.unique(labels_unscaled))
print("Unique clusters (scaled):", np.unique(labels_scaled))

# Save simple plot comparing dendrogram shapes (truncated for readability)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Dendrogram (unscaled) - truncated")
from scipy.cluster.hierarchy import dendrogram
dendrogram(Z_unscaled, truncate_mode='lastp', p=20, no_labels=True)
plt.subplot(1,2,2)
plt.title("Dendrogram (scaled) - truncated")
dendrogram(Z_scaled, truncate_mode='lastp', p=20, no_labels=True)
plt.tight_layout()
plt.savefig('Q01_dendrograms_comparison.png')
print('Saved Q01_dendrograms_comparison.png')