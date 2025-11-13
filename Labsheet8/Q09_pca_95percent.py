"""
Auto-generated script for an unsupervised learning task using the Wine dataset.
Each script is self-contained and uses sklearn.datasets.load_wine()
Run with: python3 <script_name>.py
"""
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

X = load_wine().data
Xs = StandardScaler().fit_transform(X)
pca = PCA().fit(Xs)
cum = np.cumsum(pca.explained_variance_ratio_)
n95 = np.argmax(cum >= 0.95) + 1
print('Components to retain for 95% cumulative explained variance:', n95)
print('Cumulative ratios:', np.round(cum,3))