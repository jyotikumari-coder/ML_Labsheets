# Q3: Custom function to detect outliers using IQR
import pandas as pd

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return df[(df[column] < lower) | (df[column] > upper)]

data = pd.DataFrame({'Values': [10, 12, 11, 13, 12, 15, 200, 13, 11, 12]})
print("Outliers Detected:\n", detect_outliers_iqr(data, 'Values'))
