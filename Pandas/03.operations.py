import pandas as pd

movies_df = pd.read_csv("IMDB-Movie-Data.csv", index_col="Title")

# Viewing your data
print(movies_df.head()) #5 rows by default
#print(movies_df.head(10))
#print(movies_df.tail(2))

print(movies_df.info()) # How many non null entries per column, incl. index. Column type
print(movies_df.shape)

temp_df = movies_df.append(movies_df)
print(temp_df.shape)
temp_df = temp_df.drop_duplicates()
print(temp_df.shape)

temp_df.drop_duplicates(inplace=True)
temp_df = movies_df.append(movies_df)  # make a new copy
temp_df.drop_duplicates(inplace=True, keep=False) # first, last or False
print(temp_df.shape)

print(movies_df.columns)

movies_df.rename(columns={
        'Runtime (Minutes)': 'Runtime', 
        'Revenue (Millions)': 'Revenue_millions'
    }, inplace=True)

print(movies_df.columns)

movies_df.columns = [col.lower() for col in movies_df]
# movies_df.columns = [col.lower() for col in movies_df.columns] # It's the same
print(movies_df.columns)


# Most commonly you'll see Python's None or NumPy's np.nan, each of which are handled differently in some situations.
print(movies_df.isnull()) # Prints the map of null values

print(movies_df.isnull().sum()) # For each column, collapses the rows

#movies_df.dropna(inplace=True)
#print(movies_df.dropna(axis=1)) #Removes the columns

# Imputation, i.e., replacing null values
revenue = movies_df['revenue_millions']
print(revenue.head()) # N.B., the index column is always present
revenue_mean = revenue.mean()
print(revenue_mean)
revenue.fillna(revenue_mean, inplace=True)
print(movies_df.isnull().sum())

print(movies_df.describe()) # Numeric only
print(movies_df['genre'].describe()) # Categoric
print(movies_df['genre'].value_counts().head(10)) # The ten most common values

# Relationships between continuous variables
print(movies_df.corr())

genre_col = movies_df['genre']
print(type(genre_col)) # Series
genre_col = movies_df[['genre']]
print(type(genre_col)) #DataFrame

subset = movies_df[['genre', 'rating']]
print(subset.head())

#location
prom = movies_df.loc["Prometheus"] #index column
print(prom)

prom = movies_df.iloc[1] #numerical index
print(prom) # Prints the same row as above

movie_subset = movies_df.loc['Prometheus':'Sing'] # Sing included
print(movie_subset)
movie_subset = movies_df.iloc[1:4] # 4 not included
print(movie_subset)

#Queries
condition = (movies_df['director'] == "Ridley Scott") # Important for vectorization
print(condition.head()) # False, True, False, etc.

print(movies_df[movies_df['director'] == "Ridley Scott"])
print(movies_df[movies_df['rating'] >= 8.6].head(3))
print(movies_df[(movies_df['director'] == 'Christopher Nolan') | (movies_df['director'] == 'Ridley Scott')].head())
# More coincise with isin() method of a DataFrame or of a Series
print(movies_df[movies_df['director'].isin(['Christopher Nolan', 'Ridley Scott'])].head())
#quantiles
print(movies_df[
    ((movies_df['year'] >= 2005) & (movies_df['year'] <= 2010))
    & (movies_df['rating'] > 8.0)
    & (movies_df['revenue_millions'] < movies_df['revenue_millions'].quantile(0.25))
])

# Applying functions (to a column). Apply return a new series
#  pandas is utilizing vectorization, a style of computer programming
# where operations are applied to whole arrays instead of individual elements
def rating_function(x):
    if x >= 8.0:
        return "good"
    else:
        return "bad"
#movies_df["rating_category"] = movies_df["rating"].apply(rating_function)
#print(movies_df.head(2))
#The same, with lambda functions
movies_df["rating_category"] = movies_df["rating"].apply(lambda x: 'good' if x >= 8.0 else 'bad')
movies_df.head(2)