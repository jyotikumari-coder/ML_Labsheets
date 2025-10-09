import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [25, 30, 35, 40],
    "Salary": [50000, 60000, 70000, 80000]
}

df = pd.DataFrame(data)
print(df)
filtered = df[df['Age'] > 30]
print(filtered)
