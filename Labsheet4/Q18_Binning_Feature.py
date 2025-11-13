# Q18: Bin continuous feature into categories
import pandas as pd

ages = pd.DataFrame({'Age': [5, 12, 17, 24, 35, 45, 60]})
bins = [0, 12, 19, 35, 60, 100]
labels = ['Child','Teen','Young Adult','Adult','Senior']
ages['AgeGroup'] = pd.cut(ages['Age'], bins=bins, labels=labels)
print(ages)
