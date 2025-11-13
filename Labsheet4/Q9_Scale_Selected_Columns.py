# Q9: Scale selected columns and join back
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.DataFrame({'A': [10,20,30], 'B':[40,50,60], 'C':['x','y','z']})
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[['A','B']] = scaler.fit_transform(df[['A','B']])
print(df_scaled)
