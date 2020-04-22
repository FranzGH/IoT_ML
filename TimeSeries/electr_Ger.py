# https://www.ethanrosenthal.com/2018/01/28/time-series-for-scikit-learn-people-part1/

# https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-9-time-series-analysis-in-python-a270cb05e0b3

import pandas as pd
print(pd.to_datetime('2018-01-15 3:45pm'))

print(pd.to_datetime('7/8/1952'))

print(pd.to_datetime('7/8/1952', dayfirst=True))

print(pd.to_datetime(['2018-01-05', '7/8/1952', 'Oct 10, 1995']))

print(pd.to_datetime(['2/25/10', '8/6/17', '12/15/12'], format='%m/%d/%y'))

opsd_daily = pd.read_csv('./data/opsd_germany_daily.csv')
print(opsd_daily.shape)

print(opsd_daily.head(3))

print(opsd_daily.tail(3))

print(opsd_daily.dtypes)

opsd_daily = opsd_daily.set_index('Date')
print(opsd_daily.head(3))

print(opsd_daily.index)

# Alternatively, in one line!
opsd_daily = pd.read_csv('./data/opsd_germany_daily.csv', index_col=0, parse_dates=True)
print(opsd_daily.head(3))

# Add columns with year, month, and weekday name
opsd_daily['Year'] = opsd_daily.index.year
opsd_daily['Month'] = opsd_daily.index.month
opsd_daily['Weekday Name'] = opsd_daily.index.weekday_name
# Display a random sampling of 5 rows
print(opsd_daily.sample(5, random_state=0))

#####
# Time-based indexing
#####

'''
print(opsd_daily.loc['2017-08-10'])

# Slicing
print(opsd_daily.loc['2014-01-20':'2014-01-22'])

# Partial-string indexing
print(opsd_daily.loc['2012-02'])
'''

######
# Visualizing time series data
######

import matplotlib.pyplot as plt
import seaborn as sns
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})

'''
opsd_daily['Consumption'].plot(linewidth=0.5) # Direct plot of a Panda series
plt.show()

# Let's plot better...
cols_plot = ['Consumption', 'Solar', 'Wind']
axes = opsd_daily[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
# A plot() creates an axis!
for ax in axes:
    ax.set_ylabel('Daily Totals (GWh)')
plt.show()

# Power of partial-string indexing! Try also with months
ax = opsd_daily.loc['2017', 'Consumption'].plot()
ax.set_ylabel('Daily Consumption (GWh)')
plt.show()

ax = opsd_daily.loc['2017-01':'2017-02', 'Consumption'].plot(marker='o', linestyle='-')
ax.set_ylabel('Daily Consumption (GWh)');
plt.show()


# https://stackoverflow.com/questions/42365766/pandas-matplotlib-plot-with-multiple-y-axes
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

opsd_daily.loc['2017-01':'2017-02', 'Consumption'].plot(marker='o', linestyle='-', ax=ax1)
ax1.set_ylabel('Daily Consumption (GWh)')
ax1.set_xlabel('2017')
opsd_daily.loc['2016-01':'2016-02', 'Consumption'].plot(marker='o', linestyle='-', ax=ax2)
ax2.set_ylabel('Daily Consumption (GWh)')
ax2.set_xlabel('2016')
plt.show()
'''

#######
# Customizing time series plots
#######

import matplotlib.dates as mdates
'''
# Because date/time ticks are handled a bit differently in matplotlib.dates
# compared with the DataFrame’s plot() method, let’s create the plot directly in matplotlib.
fig, ax = plt.subplots()
ax.plot(opsd_daily.loc['2017-01':'2017-02', 'Consumption'], marker='o', linestyle='-')
ax.set_ylabel('Daily Consumption (GWh)')
ax.set_title('Jan-Feb 2017 Electricity Consumption')
# Set x-axis major ticks to weekly interval, on Mondays
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
# Format x-tick labels as 3-letter month name and day number
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.show()
'''

#####
# Seasonality
#####
'''
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for name, ax in zip(['Consumption', 'Solar', 'Wind'], axes):
    sns.boxplot(data=opsd_daily, x='Month', y=name, ax=ax)
    ax.set_ylabel('GWh')
    ax.set_title(name)
    # Remove the automatic x-axis label from all but the bottom subplot
    if ax != axes[-1]:
        ax.set_xlabel('')
plt.show()

sns.boxplot(data=opsd_daily, x='Weekday Name', y='Consumption')
plt.show()
'''

#####
# Frequencies
#####
'''
print(pd.date_range('1998-03-10', '1998-03-15', freq='D'))

print(pd.date_range('2004-09-20', periods=8, freq='H'))
# Attention: ValueError: Of the four parameters: start, end, periods, and freq, exactly three must be specified
print(pd.date_range('2004-09-20', '2004-09-21', freq='4H'))

# If we know that our data should be at a specific frequency,
# we can use the DataFrame’s asfreq() method to assign a frequency.

# To select an arbitrary sequence of date/time values from a pandas time series,
# we need to use a DatetimeIndex, rather than simply a list of date/time strings
times_sample = pd.to_datetime(['2013-02-03', '2013-02-06', '2013-02-08'])
# Select the specified dates and just the Consumption column
consum_sample = opsd_daily.loc[times_sample, ['Consumption']].copy()
print(consum_sample)

# Convert the data to daily frequency, without filling any missings
consum_freq = consum_sample.asfreq('D')
# Create a column with missings forward filled
consum_freq['Consumption - Forward Fill'] = consum_sample.asfreq('D', method='ffill')
print(consum_freq)

# If you’re doing any time series analysis which requires uniformly spaced data without any missings,
# you’ll want to use asfreq() to convert your time series to the specified frequency and
# fill any missings with an appropriate method.
'''

#####
# Resampling
#####


# The resample() method returns a Resampler object, similar to a pandas GroupBy object.
# We can then apply an aggregation method such as mean(), median(), sum(), etc.,
# to the data group for each time bin.

# For example, let’s resample the data to a weekly mean time series.
# Specify the data columns we want to include (i.e. exclude Year, Month, Weekday Name)
data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']

'''
# Resample to weekly frequency, aggregating with mean
opsd_weekly_mean = opsd_daily[data_columns].resample('W').mean()
print(opsd_weekly_mean.head(3))
print(opsd_weekly_mean.sample(3))

print(opsd_daily.shape[0])
print(opsd_weekly_mean.shape[0])

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Start and end of the date range to extract
start, end = '2017-01', '2017-06'
# Plot daily and weekly resampled time series together
fig, ax = plt.subplots()
ax.plot(opsd_daily.loc[start:end, 'Solar'],
marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(opsd_weekly_mean.loc[start:end, 'Solar'],
marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.set_ylabel('Solar Production (GWh)')
ax.legend()
plt.show()

# Compute the monthly sums, setting the value to NaN for any month which has
# fewer than 28 days of data
opsd_monthly = opsd_daily[data_columns].resample('M').sum(min_count=28)
print(opsd_monthly.head(3))
print(opsd_monthly.sample(3))

fig, ax = plt.subplots()
#ax.plot(opsd_monthly['Consumption'], color='black', label='Consumption') # same as below
opsd_monthly['Consumption'].plot(ax = ax, color='black', label='Consumption')
opsd_monthly[['Wind', 'Solar']].plot.area(ax=ax, linewidth=0)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend()
ax.set_ylabel('Monthly Total (GWh)');
plt.show()

# Compute the annual sums, setting the value to NaN for any year which has
# fewer than 360 days of data
opsd_annual = opsd_daily[data_columns].resample('A').sum(min_count=360)
# The default index of the resampled DataFrame is the last day of each year,
# ('2006-12-31', '2007-12-31', etc.) so to make life easier, set the index
# to the year component
opsd_annual = opsd_annual.set_index(opsd_annual.index.year)
opsd_annual.index.name = 'Year'
# Compute the ratio of Wind+Solar to Consumption
opsd_annual['Wind+Solar/Consumption'] = opsd_annual['Wind+Solar'] / opsd_annual['Consumption']
opsd_annual.tail(3)

# Plot from 2012 onwards, because there is no solar production data in earlier years
ax = opsd_annual.loc[2012:, 'Wind+Solar/Consumption'].plot.bar(color='C0')
ax.set_ylabel('Fraction')
ax.set_ylim(0, 0.3)
ax.set_title('Wind + Solar Share of Annual Electricity Consumption')
plt.xticks(rotation=0)
plt.show()
'''

######
# Rolling windows
######

# Similar to downsampling, rolling windows split the data into time windows and 
# the data in each window is aggregated with a function such as mean(), median(), sum(), etc.
# However, unlike downsampling, where the time bins do not overlap and the output is at a lower frequency than
# the input, rolling windows overlap and “roll” along at the same frequency as the data,
# so the transformed time series is at the same frequency as the original time series.

# Compute the centered 7-day rolling mean
opsd_7d = opsd_daily[data_columns].rolling(7, center=True).mean()
print(opsd_7d.head(10))

opsd_7d = opsd_daily.loc['2012-01-01':, data_columns].rolling(7, center=True).mean()
print(opsd_7d.head(10))

opsd_weekly_mean = opsd_daily[data_columns].resample('W').mean() # re-taken from above

# Start and end of the date range to extract
start, end = '2017-01', '2017-06'
# Plot daily, weekly resampled, and 7-day rolling mean time series together
fig, ax = plt.subplots()
ax.plot(opsd_daily.loc[start:end, 'Solar'],
marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(opsd_weekly_mean.loc[start:end, 'Solar'],
marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.plot(opsd_7d.loc[start:end, 'Solar'],
marker='.', linestyle='-', label='7-d Rolling Mean')
ax.set_ylabel('Solar Production (GWh)')
ax.legend()
plt.show()

######
# Trends
######

# Time series data often exhibit some slow, gradual variability in addition to higher frequency variability
# such as seasonality and noise.
# An easy way to visualize these trends is with rolling means at different time scales.

# The min_periods=360 argument accounts for a few isolated missing days in the
# wind and solar production time series
opsd_365d = opsd_daily[data_columns].rolling(window=365, center=True, min_periods=360).mean()

# Plot daily, 7-day rolling mean, and 365-day rolling mean time series
fig, ax = plt.subplots()
ax.plot(opsd_daily['Consumption'], marker='.', markersize=2, color='0.6',
linestyle='None', label='Daily')
ax.plot(opsd_7d['Consumption'], linewidth=2, label='7-d Rolling Mean')
ax.plot(opsd_365d['Consumption'], color='0.2', linewidth=3,
label='Trend (365-d Rolling Mean)')
# Set x-ticks to yearly interval and add legend and labels
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend()
ax.set_xlabel('Year')
ax.set_ylabel('Consumption (GWh)')
ax.set_title('Trends in Electricity Consumption')
plt.show()

# We can see that the 7-day rolling mean has smoothed out all the weekly seasonality,
# while preserving the yearly seasonality.
# The 7-day rolling mean reveals that while electricity consumption is typically higher in winter and
# lower in summer, there is a dramatic decrease for a few weeks every winter
# at the end of December and beginning of January, during the holidays.

#Looking at the 365-day rolling mean time series,
# we can see that the long-term trend in electricity consumption is pretty flat,
# with a couple of periods of anomalously low consumption around 2009 and 2012-2013.

# Plot 365-day rolling mean time series of wind and solar power
fig, ax = plt.subplots()
for nm in ['Wind', 'Solar', 'Wind+Solar']:
    ax.plot(opsd_365d[nm], label=nm)
# Set x-ticks to yearly interval, adjust y-axis limits, add legend and labels
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.set_ylim(0, 400)
ax.legend()
ax.set_ylabel('Production (GWh)')
ax.set_title('Trends in Electricity Production (365-d Rolling Means)')
plt.show()