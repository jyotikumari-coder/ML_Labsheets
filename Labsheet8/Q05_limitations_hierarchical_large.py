"""
Auto-generated script for an unsupervised learning task using the Wine dataset.
Each script is self-contained and uses sklearn.datasets.load_wine()
Run with: python3 <script_name>.py
"""
import time
import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.datasets import make_blobs

# Create increasing sizes and time linkage on a sample subset
sizes = [200, 500, 1000]
times = {}
for n in sizes:
    X, _ = make_blobs(n_samples=n, n_features=5, random_state=42)
    t0 = time.time()
    # we run linkage but be aware of runtime/memory
    linkage(X, method='ward')
    t1 = time.time()
    times[n] = t1 - t0
    print(f'Linkage time for n={n}: {times[n]:.2f} s')

print("\nConclusion: hierarchical linkage scales poorly (time & memory). For large n use
 - mini-batch KMeans, DBSCAN, or approximate methods, or sample/aggregate first.")