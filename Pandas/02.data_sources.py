import pandas as pd
# Reading data from CSVs

df = pd.read_csv('purchases.csv')
print(df)

df2 = pd.read_json('purchases.json')
print(df2)

#In fact, we could use set_index() on any DataFrame using any column at any time.
# Indexing Series and DataFrames is a very common task, and the different ways of doing it is worth remembering.

#df = df.set_index('index')
df = df.set_index('Unnamed: 0')
print(df)

df.to_csv('new_purchases.csv')

df2.to_json('new_purchases.json')