# 4_

#import necessary modules
import pandas as pd
#file = "WA_Fn-UseC_-Sales-Win-Loss.csv"
## Read in the data with `read_csv()`
#sales_data = pd.read_csv(file)
import preprocess as pp

# select columns other than 'Opportunity Number','Opportunity Result'
cols = [col for col in pp.sales_data.columns if col not in ['Opportunity Number','Opportunity Result']]
# dropping the 'Opportunity Number'and 'Opportunity Result' columns
# create the feature set
data = pp.sales_data[cols]
#assigning the Oppurtunity Result column as target
target = pp.sales_data['Opportunity Result']
print(data.head(n=2))
print(target.head(n=2))

#import the necessary module
from sklearn.model_selection import train_test_split

global data_train, data_test, target_train, target_test
#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.30, random_state = 10)
# The fourth argument ‘random_state’ just ensures that we get reproducible results every time.

