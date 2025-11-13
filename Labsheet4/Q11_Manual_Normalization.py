# Q11: Manual normalization function
import pandas as pd

def normalize_column(col):
    return (col - col.min()) / (col.max() - col.min())

data = pd.DataFrame({'Values': [10, 12, 15, 20, 25]})
data['Normalized'] = normalize_column(data['Values'])
print(data)
