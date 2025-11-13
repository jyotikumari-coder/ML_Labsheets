# Q14: Extract day, month, year from date
import pandas as pd

df = pd.DataFrame({'Date': pd.to_datetime(['2023-01-15', '2023-05-20', '2023-12-01'])})
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
print(df)
