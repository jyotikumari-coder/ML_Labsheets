"""
Auto-generated script for an unsupervised learning task using the Wine dataset.
Each script is self-contained and uses sklearn.datasets.load_wine()
Run with: python3 <script_name>.py
"""
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_wine()
X = data.data; y = data.target
Xs = StandardScaler().fit_transform(X)

Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=42)
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
acc_orig = accuracy_score(yte, clf.predict(Xte))

pca = PCA(n_components=5, random_state=42)
Xp = pca.fit_transform(Xs)
Xptr, Xpte, ytr, yte = train_test_split(Xp, y, test_size=0.3, random_state=42)
clf2 = LogisticRegression(max_iter=1000).fit(Xptr, ytr)
acc_pca = accuracy_score(yte, clf2.predict(Xpte))

print('Accuracy without PCA:', round(acc_orig,4))
print('Accuracy with PCA (5 comps):', round(acc_pca,4))