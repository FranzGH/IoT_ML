# https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/

#Importing libraries. The same will be used throughout the article.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10

#Define input array with angles from 60deg to 300deg converted to radians
x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = np.sin(x) + np.random.normal(0,0.15,len(x))
data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])
plt.plot(data['x'],data['y'],'.')
#plt.show()

for i in range(2,16):  #power of 1 is already there
    colname = 'x_%d'%i      #new var will be x_power
    data[colname] = data['x']**i
print(data.head())


##############
## Let’s try to estimate the sine function using polynomial regression with powers of x from 1 to 15.
## I.e., for every point x, we also consider as a dimension x^2, x^3, ..., x^15
##############


########
# linear_regression model
########

#Import Linear Regression model from scikit-learn.
from sklearn.linear_model import LinearRegression
def linear_regression(data, power, models_to_plot):
    #initialize predictors:
    predictors=['x'] # Power of 1
    if power>=2: # For higher level powers
        predictors.extend(['x_%d'%i for i in range(2,power+1)]) #x_2, x_3, created at the beginning when creating data.
    
    #Fit the model
    linreg = LinearRegression(normalize=True)
    # See here about normalize: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    # About vector norms: https://machinelearningmastery.com/vector-norms-machine-learning/
    linreg.fit(data[predictors],data['y'])
    y_pred = linreg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered power
    if power in models_to_plot:
        plt.subplot(models_to_plot[power])
        plt.tight_layout()
        plt.plot(data['x'],y_pred) # line joining the predictions (N.B., x is ordered)
        plt.plot(data['x'],data['y'],'.') # original x,y samples
        #plt.plot(data['x'],data['y'])
        #plt.scatter(data['x'],data['y'])
        plt.title('Plot for power: %d'%power)

    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2) # residual sum of squares (sum of squared residuals)
    #RSS is a measure of how good the model approximates the data while OLS is a method of constructing a good model
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    return ret

    #Initialize a dataframe to store the results:
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)] # A powerful way of dding columns
ind = ['model_pow_%d'%i for i in range(1,16)]
coef_matrix_simple = pd.DataFrame(index=ind, columns=col)

#Define the powers for which a plot is required:
models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236} #This is a dictionary!

#Iterate through all powers and assimilate results
for i in range(1,16):
    coef_matrix_simple.iloc[i-1,0:i+2] = linear_regression(data, power=i, models_to_plot=models_to_plot)
    # i+2: i is the power, +2 is for rss and intercept

plt.show()

#Set the display format to be scientific for ease of analysis
pd.options.display.float_format = '{:,.2g}'.format
print(coef_matrix_simple)
# Simple, because no regularization yet.
# One row for each model (power), Columns are the coefficients, as for ridge and lasso,
# but rows do not have all coefficients, obivosuly: triangular matrix


########
# ridge_regression model
########

# Objective = RSS + α * (sum of square of coefficients)
# Residual sum of sqares, see above

from sklearn.linear_model import Ridge
def ridge_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['y'])
    y_pred = ridgereg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred) # line joining the predictions (N.B., x is ordered)
        plt.plot(data['x'],data['y'],'.') # original samples
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret

    #Initialize predictors to be set of 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Set the different values of alpha to be tested
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)] # 10 are the alpha_ridge values
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
for i in range(10):
    coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot) # Now we consider always all the predictors (i.e., powers)...
    # We have to fill the whole row...
plt.show()

#Set the display format to be scientific for ease of analysis
pd.options.display.float_format = '{:,.2g}'.format
print(coef_matrix_ridge)

print(coef_matrix_ridge.apply(lambda x: sum(x.values==0),axis=1))
# Powerful column-level operation on a dataframe


########
# lasso_regression model
########

# Objective = RSS + α * (sum of absolute value of coefficients)

from sklearn.linear_model import Lasso
def lasso_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    lassoreg.fit(data[predictors],data['y'])
    y_pred = lassoreg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred) # line joining the predictions (N.B., x is ordered)
        plt.plot(data['x'],data['y'],'.') # original values
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret

#Initialize predictors to all 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Define the alpha values to test
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]

#Initialize the dataframe to store coefficients
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Define the models to plot
models_to_plot = {1e-10:231, 1e-5:232,1e-4:233, 1e-3:234, 1e-2:235, 1:236}

#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(data, predictors, alpha_lasso[i], models_to_plot)
    # We have to fill the whole row...

plt.show()
print(coef_matrix_lasso)
print(coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1))

# See the article's conclusions!
print(data.corr())
print()
print(data.head())
print(data.tail())