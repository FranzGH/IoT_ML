# https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-9-time-series-analysis-in-python-a270cb05e0b3

# There’s nothing unusual here, as always we have to choose a loss function suitable for the task,
# that will tell us how close the model approximates data.
# Then using cross-validation we will evaluate our chosen loss function for given model parameters, calculate gradient, 
# adjust model parameters and so forth, bravely descending to the global minimum of error.

import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

# %matplotlib inline

ads = pd.read_csv('data/ads.csv', index_col=['Time'], parse_dates=['Time'])
currency = pd.read_csv('data/currency.csv', index_col=['Time'], parse_dates=['Time'])

# If the process is stationary that means it doesn’t change its statistical properties over time,
# namely mean and variance do not change over time (constancy of variance is also called homoscedasticity),
# also covariance function does not depend on the time (should only depend on the distance between observations). 

'''
white_noise = np.random.normal(size=1000)
with plt.style.context('bmh'):  
    plt.figure(figsize=(15, 5))
    plt.plot(white_noise)
    plt.show()

def plotProcess(n_samples=1000, rho=0):
    x = w = np.random.normal(size=n_samples)
    for t in range(n_samples):
        x[t] = rho * x[t-1] + w[t]

    with plt.style.context('bmh'):  
        plt.figure(figsize=(10, 5))
        plt.plot(x)
        plt.title("Rho {}\n Dickey-Fuller p-value: {}".format(rho, round(sm.tsa.stattools.adfuller(x)[1], 3)))
        
for rho in [0, 0.6, 0.9, 1]:
    plotProcess(rho=rho)
plt.show()
'''

# We can fight non-stationarity using different approaches — various order differences,
# trend and seasonality removal, smoothing, also using transformations like Box-Cox or logarithmic.

# Null hypothesis of the test — time series is non-stationary, was rejected on the first three charts and
# was accepted on the last one. We’ve got to say that the first difference
# is not always enough to get stationary series as the process might be integrated of order d, d > 1 (and have multiple unit roots),
# in such cases the augmented Dickey-Fuller test is used that checks multiple lags at once.

# Chart rendering code
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        # Dickey-Fuller test for stationarity
        # Null hypothesis of the test — time series is non-stationary
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        

 # Autocorrelation, also known as serial correlation,
 # is the correlation of a signal with a delayed copy of itself as a function of delay.
 # Informally, it is the similarity between observations as a function of the time lag between them.

# https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

'''
# Just to show white noise autocorrelation
n_samples = 1000
x = w = np.random.normal(size=n_samples)
# Null hyp (non stationarity) accepted, with p=0.6
tsplot(w, lags=60) # until 60 samples
plt.show()

# Just to show random walk's autocorrelation
n_samples = 1000
x = w = np.random.normal(size=n_samples)
# Null hyp (non stationarity) accepted, with p=0.6
for t in range(n_samples):
    x[t] = x[t-1] + w[t]
tsplot(x, lags=60) # until 60 samples
plt.show()
'''

tsplot(ads.Ads, lags=60) # Unit is hour, so we consider 2 days and 1/2
plt.show()

# seasonal difference
ads_diff = ads.Ads - ads.Ads.shift(24)
tsplot(ads_diff[24:], lags=60)
plt.show()
# That’s better, visible seasonality (24 hours) is gone,
# however autocorrelation function still has too many significant lags. 

# To remove them we’ll take first differences — subtraction of series from itself with lag 1
ads_diff = ads_diff - ads_diff.shift(1)
tsplot(ads_diff[24+1:], lags=60)
plt.show()
# Our series now look like something undescribable, oscillating around zero,
# Dickey-Fuller indicates that it’s stationary and the number of significant peaks in ACF has dropped.



# Autoregression
# https://www.youtube.com/watch?v=AN0a58F6cxA

# AutoRegression (AR), MovingAverage (MA)
#  

# setting initial values and some bounds for them
ps = range(3,4) #range(2, 5)
d=1 
qs = range(3,4)#range(2, 5)
Ps = range(1,2)#range(0, 3)
D=1 
Qs = range(1, 2)#range(0, 2)
s = 24 # season length is still 24

import itertools
import pandas as pd

# creating list with all the possible combinations of parameters
parameters = itertools.product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

def optimizeSARIMA(parameters_list, d, D, s):
    """
        Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """
    
    results = []
    best_aic = float("inf")

    #for param in tqdm_notebook(parameters_list):
    for param in parameters_list:
        # we need try-except because on some combinations model fails to converge
        try:
            model=sm.tsa.statespace.SARIMAX(ads.Ads[:300], order=(param[0], d, param[1]), #Attention! Cut to 1000
                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1) # disable convergence messages
        except:
            continue
        # Score: Akaike’s Information Criterion (the lower the better)
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table
    

result_table = optimizeSARIMA(parameters_list, d, D, s)




# set the parameters that give the lowest AIC
p, q, P, Q = result_table.parameters[0]

best_model=sm.tsa.statespace.SARIMAX(ads.Ads[:1000], order=(p, d, q), 
                                        seasonal_order=(P, D, Q, s)).fit(disp=-1)
print(best_model.summary())

tsplot(best_model.resid[24+1:], lags=60)
plt.show()

# Well, it’s clear that the residuals are stationary, there are no apparent autocorrelations, let’s make predictions using our model

from sklearn.utils.validation import check_array
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true = check_array(y_true.values.reshape(-1,1))
    y_pred = check_array(y_pred.values.reshape(-1,1))

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plotSARIMA(series, model, n_steps):
    """
        Plots model vs predicted values
        
        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future
        
    """
    # adding model values
    data = series.copy()
    data.columns = ['actual']
    data['arima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['arima_model'][:s+d] = np.NaN
    
    # forecasting on n_steps forward 
    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps) # Predict starting from the final sample
    forecast = data.arima_model.append(forecast) # These are the fitted values (see above)
    # calculate error, again having shifted on s+d steps from the beginning
    from sklearn import metrics
    error = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])

    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey') #vertical span. N.B. Index is exposed
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True)
    
plotSARIMA(ads, best_model, 50)
plt.show()