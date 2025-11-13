# Q15: Create new column as product or ratio
import pandas as pd

df = pd.DataFrame({'Price': [100,200,300], 'Quantity': [2,4,6]})
df['Total'] = df['Price'] * df['Quantity']
df['Price_Per_Unit'] = df['Price'] / df['Quantity']
print(df)
