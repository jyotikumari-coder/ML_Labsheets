df.to_csv("output.csv", index=False)
new_df = pd.read_csv("output.csv")
print(new_df.head())
