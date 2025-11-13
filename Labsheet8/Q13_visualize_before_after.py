"""
Auto-generated script for an unsupervised learning task using the Wine dataset.
Each script is self-contained and uses sklearn.datasets.load_wine()
Run with: python3 <script_name>.py
"""
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = load_wine().data
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=42)
Xp = pca.fit_transform(Xs)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.scatter(Xs[:,0], Xs[:,1])
plt.title('Original (first two features, scaled)')
plt.subplot(1,2,2)
plt.scatter(Xp[:,0], Xp[:,1])
plt.title('After PCA (2 components)')
plt.savefig('Q13_before_after_pca.png')
print('Saved Q13_before_after_pca.png')