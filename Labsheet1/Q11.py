import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length','sepal_width','petal_length','petal_width','class']
iris = pd.read_csv(url, names=columns)
print(iris.head())
