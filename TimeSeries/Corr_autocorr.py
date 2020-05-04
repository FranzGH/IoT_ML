# https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

from pandas import read_csv
from matplotlib import pyplot
series = read_csv('data/daily-minimum-temperatures.csv', header=0, index_col=0)
figsize=(12, 7)
series.plot(figsize=figsize)
pyplot.show()

# We can calculate the correlation for time series observations with observations with previous time steps, called lags.


from statsmodels.graphics.tsaplots import plot_acf
with pyplot.rc_context(): #Just to have a window of different size
    pyplot.rc("figure", figsize=figsize)
    plot_acf(series, lags=2000) 
pyplot.show()

plot_acf(series, lags=500)
pyplot.show()

