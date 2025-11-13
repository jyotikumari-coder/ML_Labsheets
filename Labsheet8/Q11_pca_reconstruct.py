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
pca = PCA(n_components=5, random_state=42)  # choose 5 for demonstration
Xred = pca.fit_transform(Xs)
Xrec = pca.inverse_transform(Xred)
mse = np.mean((Xs - Xrec)**2)
print(f'Reconstruction MSE with 5 components: {mse:.6f}')