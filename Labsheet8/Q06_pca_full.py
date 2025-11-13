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
pca = PCA()
pca.fit(Xs)
print('Explained variance ratio (first 10):', np.round(pca.explained_variance_ratio_[:10], 4))