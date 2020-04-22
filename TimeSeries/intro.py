# https://maelfabien.github.io/statistics/TimeSeries1/#ii-illustration-using-open-data

### General import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import statsmodels.api as sm

### Time Series
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
#from pandas.tools.plotting import autocorrelation_plot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.sarimax_model import SARIMAX

### LSTM Time Series
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.layers import Dropout 
from sklearn.preprocessing import MinMaxScaler 

df = pd.read_csv('data/opsd_germany_daily.csv', index_col=0)
print(df.head(10))

df.index = pd.to_datetime(df.index)

print(df.describe())

'''
# Distribution of the consumption
plt.figure(figsize=(12,8))
plt.hist(df['Consumption'], bins=100)
plt.title("Distribution of the consumption")
plt.xlabel("Electricity consumption in GWh")
#plt.show()

# Distribution of the wind power production
plt.figure(figsize=(12,8))
plt.hist(df['Wind'], bins=100)
plt.title("Distribution of the wind power production")
plt.xlabel("Wind power production in GWh")
#plt.show()

# Distribution of the solar power production
plt.figure(figsize=(12,8))
plt.hist(df['Solar'], bins=100)
plt.title("Distribution of the solar power production")
plt.xlabel("Solar power production in GWh")
plt.show()

#Time series
plt.figure(figsize=(12,8))
plt.plot(df['Consumption'], linewidth = 0.5)
plt.plot(df['Wind+Solar'], linewidth = 0.5)
plt.title("Consumption vs. Production")
plt.show()

plt.figure(figsize=(12,8))
plt.plot(df['Consumption'], linewidth = 0.5)
plt.title("Consumption over time")
plt.show()

# Marker, no line
plt.figure(figsize=(12,8))
plt.plot(df['Consumption'], linewidth = 0.5, linestyle = "None", marker='.')
plt.title("Consumption over time")
plt.show()

plt.figure(figsize=(12,8))
plt.plot(df['Wind'], linewidth = 0.5)
plt.title("Wind production over time")
plt.show()

plt.figure(figsize=(12,8))
plt.plot(df['Solar'], linewidth = 0.5)
plt.title("Solar production over time")
plt.show()

########
# 4 Change scale
########
plt.figure(figsize=(12,8))
plt.plot(df.loc['2017-01':'2017-12']['Consumption'], linewidth = 0.5)
plt.title("Electricity Consumption in 2017")
plt.show()

plt.figure(figsize=(12,8))
plt.plot(df.loc['2017-01':'2017-12']['Wind'], linewidth = 0.5)
plt.title("Wind Production in 2017")
plt.show()

plt.figure(figsize=(12,8))
plt.plot(df.loc['2017-01':'2017-12']['Solar'], linewidth = 0.5)
plt.title("Solar Production in 2017")
plt.show()

# Weekly
# To understand the weekly trends, we can take a period of 3 weeks.
# I’ve added red line on Sundays, to better understand the pattern through the week :

plt.figure(figsize=(12,8))
plt.plot(df.loc['2017-12-09':'2017-12-31']['Consumption'], linewidth = 0.5)
plt.title("Electricity Consumption in December 2017")
plt.axvline("2017-12-10", c='r')
plt.axvline("2017-12-17", c='r')
plt.axvline("2017-12-24", c='r')
plt.show()

######
# 5 Boxplots
######

# Boxplot interpretation
# https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.weekday_name

plt.figure(figsize=(12,8))
sns.boxplot(data=df, x='year', y='Consumption')
plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(data=df, x='month', y='Consumption')
plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(data=df, x='day', y='Consumption')
plt.show()
'''

#####
# 6 Handling missing values
#####

# Forward filling
df = df.fillna(method='ffill')
# bfill, for backward filling

# Rolling mean
plt.figure(figsize=(12,8))
plt.plot(df.loc['2017-11':'2017-12']['Consumption'].rolling('7D').mean())
plt.axvline('2017-11-09', color = 'red')
plt.axvline('2017-11-16', color = 'red')
plt.show()

plt.figure(figsize=(12,8))
plt.plot(df['Consumption'], label="Consumption")
plt.plot(df['Consumption'].rolling('90D').mean(), label="Rolling Mean")
plt.legend()
plt.show()

# Expanding, which is: average until the point
plt.figure(figsize=(12,8))
plt.plot(df['Consumption'], label="Consumption")
plt.plot(df['Consumption'].expanding().mean(), label="Rolling Mean")
plt.legend()
plt.show()