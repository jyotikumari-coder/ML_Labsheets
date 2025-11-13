import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")  # Example dataset
x = df[['Annual Income (k$)', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters=5, random_state=42).fit(x)
df['Cluster'] = kmeans.labels_
print(df.head())
