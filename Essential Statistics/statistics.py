# https://www.learndatasci.com/tutorials/data-science-statistics-using-python/

# https://seaborn.pydata.org/
# Seaborn is a Python data visualization library based on matplotlib.
# It provides a high-level interface for drawing attractive and informative statistical graphics.

import pandas as pd
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set(font_scale=1.5)

# Read in TN middle school dataset from GitHub
df = pd.read_csv('middle_tn_schools.csv')
pd.options.display.max_columns = df.shape[1] #to show all columns
print(df.describe(include='all')) #to show numerical and categorical properties

# measurements = outcomes or indicators
# reduced_lunch is % of students on reduced lunch. It may indicate poverty. It is an indicator.
print(df[['reduced_lunch', 'school_rating']].groupby(['school_rating']).describe())

# The descriptive statistics consistently reveal that schools with more students on reduced lunch under-perform when compared to their peers. 
print(df[['reduced_lunch', 'school_rating']].corr())

fig, ax = plt.subplots(figsize=(14,8))

ax.set_ylabel('school_rating')
# ax.set_xlabel('reduced_lunch') #It has no effects, don't know why

# boxplot with only these two variables
_ = df[['reduced_lunch', 'school_rating']].boxplot(by='school_rating', figsize=(13,8), vert=False, sym='b.', ax=ax)
plt.show()
# Above a 3-star rating, more predictors are needed to explain school_rating due to an increasing spread in reduced_lunch.

plt.figure(figsize=(14,8)) # set the size of the graph
_ = sns.regplot(data=df, x='reduced_lunch', y='school_rating')
plt.show()

# create tabular correlation matrix
corr = df.corr()
_, ax = plt.subplots(figsize=(13,10)) 

# graph correlation matrix
_ = sns.heatmap(corr, ax=ax,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                cmap='coolwarm')
plt.show()