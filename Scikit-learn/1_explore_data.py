# https://www.dataquest.io/blog/sci-kit-learn-tutorial/

#import necessary modules
import pandas as pd
#store the url in a variable
# url = "https://community.watsonanalytics.com/wp-content/uploads/2015/04/WA_Fn-UseC_-Sales-Win-Loss.csv"
# url = "https://raw.githubusercontent.com/vkrit/data-science-class/master/WA_Fn-UseC_-Sales-Win-Loss.csv"
url = "WA_Fn-UseC_-Sales-Win-Loss.csv"

# Read in the data with `read_csv()`
sales_data = pd.read_csv(url)

# Using head() method with an argument which helps us to restrict the number of initial records that should be displayed
print(sales_data.head(n=2))

pd.options.display.max_columns = None
# Using .tail() method to view the last few records from the dataframe
print(sales_data.tail())

print(sales_data.dtypes)

# import the seaborn module
import seaborn as sns
# import the matplotlib module
import matplotlib.pyplot as plt
# set the background colour of the plot to white
sns.set(style="whitegrid", color_codes=True)
# setting the plot size for all plots
sns.set(rc={'figure.figsize':(11.7,8.27)})
# create a countplot with two categories
sns.countplot('Route To Market',data=sales_data,hue = 'Opportunity Result')
# Remove the top and down margin
sns.despine(offset=10, trim=True)
# display the plot
plt.show()

# setting the plot size for all plots
sns.set(rc={'figure.figsize':(16.7,13.27)})

# plotting the violinplot
sns.violinplot(x="Opportunity Result",y="Client Size By Revenue", data=sales_data)
plt.show()

