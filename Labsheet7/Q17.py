"""
Labsheet7 - Introduction to Machine Learning
Problem 17: Use a dendrogram to visualize hierarchical clustering using scipy.cluster.hierarchy.
"""

# Package Name: Labsheet7

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Your implementation here
print("Problem 17: Use a dendrogram to visualize hierarchical clustering using scipy.cluster.hierarchy.")
