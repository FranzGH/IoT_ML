# https://towardsdatascience.com/time-series-analysis-in-python-an-introduction-70d5a5b1d52a

# quandl for financial data
import quandl
# pandas for data manipulation
import pandas as pd
quandl.ApiConfig.api_key = 'BXJgiHoMSwr13ewWpbj1'

import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

'''
# Retrieve TSLA data from Quandl
tesla = quandl.get('WIKI/TSLA', limit=500)
# Retrieve the GM data from Quandl
gm = quandl.get('WIKI/GM', limit=500)

tesla.to_pickle('data/tesla.pck')
gm.to_pickle('data/gm.pck')
'''

tesla = pd.read_pickle('data/tesla.pck')
gm = pd.read_pickle('data/gm.pck')

gm.head(5)

# The adjusted close accounts for stock splits, so that is what we should graph
plt.plot(gm.index, gm['Adj. Close'])
plt.title('GM Stock Price')
plt.ylabel('Price ($)')
plt.show()
plt.plot(tesla.index, tesla['Adj. Close'], 'r')
plt.title('Tesla Stock Price')
plt.ylabel('Price ($)')
plt.show()

# Market cap= share price * number of shares
# Yearly average number of shares outstanding for Tesla and GM
tesla_shares = {2018: 168e6, 2017: 162e6, 2016: 144e6, 2015: 128e6, 2014: 125e6, 2013: 119e6, 2012: 107e6, 2011: 100e6, 2010: 51e6}
gm_shares = {2018: 1.42e9, 2017: 1.50e9, 2016: 1.54e9, 2015: 1.59e9, 2014: 1.61e9, 2013: 1.39e9, 2012: 1.57e9, 2011: 1.54e9, 2010:1.50e9}

# Create a year column 
tesla['Year'] = tesla.index.year
# Take Dates from index and move to Date column 
tesla.reset_index(level=0, inplace = True)
tesla['cap'] = 0
# Calculate market cap for all years
for i, year in enumerate(tesla['Year']):
    # Retrieve the shares for the year
    shares = tesla_shares.get(year)
    
    # Update the cap column to shares times the price
    tesla.loc[i, 'cap'] = shares * tesla.loc[i, 'Adj. Close']

# Create a year column 
gm['Year'] = gm.index.year
# Take Dates from index and move to Date column 
gm.reset_index(level=0, inplace = True)
gm['cap'] = 0
# Calculate market cap for all years
for i, year in enumerate(gm['Year']):
    # Retrieve the shares for the year
    shares = gm_shares.get(year)
    
    # Update the cap column to shares times the price
    gm.loc[i, 'cap'] = shares * gm.loc[i, 'Adj. Close']

# Merge the two datasets and rename the columns
cars = gm.merge(tesla, how='inner', on='Date')
cars.rename(columns={'cap_x': 'gm_cap', 'cap_y': 'tesla_cap'}, inplace=True)
# Select only the relevant columns
cars = cars.loc[:, ['Date', 'gm_cap', 'tesla_cap']]
# Divide to get market cap in billions of dollars
cars['gm_cap'] = cars['gm_cap'] / 1e9
cars['tesla_cap'] = cars['tesla_cap'] / 1e9
cars.head()

plt.figure(figsize=(10, 8))
plt.plot(cars['Date'], cars['gm_cap'], 'b-', label = 'GM')
plt.plot(cars['Date'], cars['tesla_cap'], 'r-', label = 'TESLA')
plt.xlabel('Date')
plt.ylabel('Market Cap (Billions $)')
plt.title('Market Cap of GM and Tesla')
plt.legend()
plt.show()

import numpy as np
# Find the first and last time Tesla was valued higher than GM
first_date = cars.loc[np.min(list(np.where(cars['tesla_cap'] > cars['gm_cap'])[0])), 'Date']
last_date = cars.loc[np.max(list(np.where(cars['tesla_cap'] > cars['gm_cap'])[0])), 'Date']
print("Tesla was valued higher than GM from {} to {}.".format(first_date.date(), last_date.date()))
#Tesla was valued higher than GM from 2017-04-10 to 2017-09-21.

indices = np.where(cars['tesla_cap'] > cars['gm_cap'])
differences = np.diff(indices)[0]

d = np.add(np.where(differences>1), 1)
d = np.append(np.array([0]), d)
startIdx = np.take(indices, d) 

endIdx = np.take(indices, np.where(differences>1)).flatten()
if endIdx.size < startIdx.size:
    endIdx = np.append(endIdx, indices[0][-1])

for st, end in zip(startIdx, endIdx):
    d1 = cars.loc[st, 'Date']
    d2 = cars.loc[end, 'Date']
    print(f'From {d1.date()} to {d2.date()}')