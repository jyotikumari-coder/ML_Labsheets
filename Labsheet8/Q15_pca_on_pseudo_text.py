"""
Auto-generated script for an unsupervised learning task using the Wine dataset.
Each script is self-contained and uses sklearn.datasets.load_wine()
Run with: python3 <script_name>.py
"""
from sklearn.datasets import load_wine
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

data = load_wine()
X = data.data

# Convert numeric features into binned 'tokens' per sample (pseudo-text)
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
Xb = kbd.fit_transform(X).astype(int)

corpus = []
for row in Xb:
    tokens = [f'F{i}_B{val}' for i, val in enumerate(row)]
    corpus.append(' '.join(tokens))

tfidf = TfidfVectorizer()
Xtf = tfidf.fit_transform(corpus)

# Use TruncatedSVD as PCA for sparse TF-IDF
svd = TruncatedSVD(n_components=2, random_state=42)
Xred = svd.fit_transform(Xtf)

k = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xred)
sil = silhouette_score(Xred, k.labels_)

print('Silhouette on pseudo-text TF-IDF + SVD + KMeans:', round(sil,4))