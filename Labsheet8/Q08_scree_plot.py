"""
Auto-generated script for an unsupervised learning task using the Wine dataset.
Each script is self-contained and uses sklearn.datasets.load_wine()
Run with: python3 <script_name>.py
"""
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X = load_wine().data
Xs = StandardScaler().fit_transform(X)
pca = PCA().fit(Xs)
evr = pca.explained_variance_ratio_

plt.figure()
plt.plot(range(1, len(evr)+1), evr, marker='o')
plt.xlabel('Principal Component'); plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.savefig('Q08_scree_plot.png')
print('Saved Q08_scree_plot.png')