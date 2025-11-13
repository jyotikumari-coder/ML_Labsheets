import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
kmeans = KMeans(n_clusters=3).fit(df)
df['Cluster'] = kmeans.labels_

def cluster_stats(df, label_col):
    return df.groupby(label_col).agg(['count', 'mean'])

print(cluster_stats(df, 'Cluster'))
