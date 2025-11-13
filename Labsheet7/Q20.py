from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
agg = AgglomerativeClustering(n_clusters=3)
df['Cluster'] = agg.fit_predict(df)
score = adjusted_rand_score(iris.target, df['Cluster'])
print("Adjusted Rand Index (Comparison with Ground Truth):", score)
