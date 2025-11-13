import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
agg = AgglomerativeClustering(n_clusters=3)
df['Cluster'] = agg.fit_predict(df)

plt.scatter(df.iloc[:,0], df.iloc[:,1], c=df['Cluster'], cmap='rainbow')
plt.title("Hierarchical Clustering Results")
plt.show()
