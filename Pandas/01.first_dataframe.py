import pandas as pd
# https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/ 

# There are many ways to create a DataFrame from scratch, but a great option is to just use a simple dict.
data = {
    'apples': [3, 2, 0, 1], 
    'oranges': [0, 3, 7, 2]
}

purchases = pd.DataFrame(data)
print(purchases)

# The Index of this DataFrame was given to us on creation as the numbers 0-3,
# but we could also create our own when we initialize the DataFrame.
# Let's have customer names as our index:

# Each (key, value) item in data corresponds to a column in the resulting DataFrame.
purchases = pd.DataFrame(data, index=['June', 'Robert', 'Lily', 'David'])

print(purchases)

# So now we could locate a customer's order by using their name:

a = purchases.loc['June']
print(a)