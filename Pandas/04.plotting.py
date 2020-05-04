import pandas as pd
import matplotlib.pyplot as plt
#df = pd.DataFrame({'lab':['A', 'B', 'C'], 'val':[10, 30, 20]})
#df.plot.bar(x='lab', y='val', rot=0)
#plt.show()


movies_df = pd.read_csv("IMDB-Movie-Data.csv", index_col="Title")
movies_df.rename(columns={
        'Runtime (Minutes)': 'Runtime', 
        'Revenue (Millions)': 'Revenue_millions'
    }, inplace=True)
movies_df.columns = [col.lower() for col in movies_df]
print(movies_df.columns)
print(movies_df.describe())

movies_df["rating_category"] = movies_df["rating"].apply(lambda x: 'good' if x >= 8.0 else 'bad')
plt.rcParams.update({'font.size': 20, 'figure.figsize': (10, 8)}) # set font and plot size to be larger

#For categorical variables utilize Bar Charts* and Boxplots.
#For continuous variables utilize Histograms, Scatterplots, Line graphs, and Boxplots.

movies_df.plot(kind='scatter', x='rating', y='revenue_millions', title='Revenue (millions) vs Rating')
plt.show()

movies_df['rating'].plot(kind='hist', title='Rating')
plt.show() # Floor int values (see below for the bar chart)

print(movies_df['rating'].describe())
#Using a Boxplot we can visualize this data:
movies_df['rating'].plot(kind="box", title='Rating boxplot')
plt.show()


movies_df.boxplot(column='revenue_millions', by='rating_category')
title_boxplot = 'Revenue millions by Rating category'
plt.title( title_boxplot )
plt.suptitle('') # that's what you're after
plt.show()

# Plot a categorical variable
#movies_df['rating'].plot(kind='chart', title='Rating') # ValueError: chart is not a valid plot kind
# movies_df['rating'].plot(kind='chart', title='Rating') # ValueErrorchart is not a valid plot kind
movies_df['rating'][:12].plot(kind='bar', title='Rating') # This is a bar, not histogram!
plt.show()
# hist (above) is ok.

# Categorical histogram
#movies_df['rating_category'].plot(kind='hist', title='Rating category') # TypeError: no numeric data to plot
movies_df['rating_category'].value_counts().plot(kind='bar', title='Rating category')
plt.show()