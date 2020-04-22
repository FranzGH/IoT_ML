# https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/

# Time series decomposition involves thinking of a series as a combination of 
# level, trend, seasonality, and noise components.

# Level: The average value in the series.
# Trend: The increasing or decreasing value in the series.
# Seasonality: The repeating short-term cycle in the series.
# Noise: The random variation in the series.

# Additive Model
# Multiplicative Model

# Additive Decomposition
# We can create a time series comprised of a linearly increasing trend from 1 to 99 and
# some random noise and decompose it as an additive model.

# Because the time series was contrived and was provided as an array of numbers,
# we must specify the frequency of the observations (the freq=1 argument).
# If a Pandas Series object is provided, this argument is not required.

from random import randrange
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
'''
series = [i+randrange(10) for i in range(1,100)]
result = seasonal_decompose(series, model='additive', freq=1)
result.plot()
pyplot.show()

from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
series = [i**2.0 for i in range(1,100)]
result = seasonal_decompose(series, model='multiplicative', freq=1)
result.plot()
pyplot.show()
'''

from pandas import read_csv
from matplotlib import pyplot
series = read_csv('data/airline-passengers.csv', header=0, index_col=0)
series.plot()
pyplot.show()


from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
'''
series = read_csv('data/airline-passengers.csv', header=0, index_col=0)
result = seasonal_decompose(series, model='multiplicative')
result.plot()
pyplot.show()

result = seasonal_decompose(series, model='additive')
result.plot()
pyplot.show()
'''

# parse the month attribute as date and make it the index as:

series = read_csv('data/airline-passengers.csv', header = 0, parse_dates = ['Month'],
index_col = ['Month'])

# pass the series into the seasonal_decompose function
result = seasonal_decompose(series, model = 'Multiplicative')

#plot all the components
result.plot()
pyplot.show()

result = seasonal_decompose(series, model = 'Additive')

#plot all the components
result.plot()
pyplot.show()

