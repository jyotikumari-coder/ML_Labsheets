"""
Auto-generated script for an unsupervised learning task using the Wine dataset.
Each script is self-contained and uses sklearn.datasets.load_wine()
Run with: python3 <script_name>.py
"""
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = load_wine()
X = StandardScaler().fit_transform(data.data)
y = data.target

pca = PCA(n_components=2, random_state=42)
Xp = pca.fit_transform(X)

plt.figure()
plt.scatter(Xp[:,0], Xp[:,1], c=y)
plt.title('Wine PCA (2D)')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.savefig('Q10_wine_pca_2d.png')
print('Saved Q10_wine_pca_2d.png')