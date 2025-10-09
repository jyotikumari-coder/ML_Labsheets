data2 = {
    "Department": ["HR","IT","HR","IT"],
    "Salary": [50000,60000,55000,65000]
}
df2 = pd.DataFrame(data2)
grouped = df2.groupby('Department')['Salary'].mean()
print(grouped)
