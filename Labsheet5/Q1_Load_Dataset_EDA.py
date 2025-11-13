# Q1: Load dataset and perform EDA
import pandas as pd
from sklearn.datasets import load_boston

# Load dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Basic EDA
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nSummary Statistics:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())
