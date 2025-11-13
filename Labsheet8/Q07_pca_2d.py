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
pca = PCA(n_components=2, random_state=42)
X2 = pca.fit_transform(Xs)

plt.figure()
plt.scatter(X2[:,0], X2[:,1], c=load_wine().target)
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('PCA (2 components)')
plt.savefig('Q07_pca_2d_scatter.png')
print('Saved Q07_pca_2d_scatter.png')