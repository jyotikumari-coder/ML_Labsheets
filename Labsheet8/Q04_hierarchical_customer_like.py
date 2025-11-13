"""
Auto-generated script for an unsupervised learning task using the Wine dataset.
Each script is self-contained and uses sklearn.datasets.load_wine()
Run with: python3 <script_name>.py
"""
# Using Wine features as 'customer' features for demonstration of hierarchical segmentation
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

X = load_wine().data
Xc = X[:, [0, 6]]  # pick two interpretable features (alcohol, color_intensity)
Xs = StandardScaler().fit_transform(Xc)

Z = linkage(Xs, method='ward')
labels = fcluster(Z, t=4, criterion='maxclust')
plt.figure()
plt.scatter(Xs[:,0], Xs[:,1], c=labels)
plt.title('Customer-like Segments (hierarchical) on Wine features')
plt.xlabel('feature 0 (scaled)')
plt.ylabel('feature 6 (scaled)')
plt.savefig('Q04_customer_segments.png')
print('Saved Q04_customer_segments.png')