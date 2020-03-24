# https://scikit-learn.org/stable/modules/linear_model.html

# https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
# Linear regression uses mean squared error as its cost function
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0, 1], [1, 2], [2, 3]], [1, 2, 3]) # 2 features, 3 training samples

print(reg.coef_)
print(reg.intercept_)

test_set = [[3,3.5], [4,5], [2,6], [0,-2]] # 4 testing samples
test_pred = reg.predict(test_set)
print(test_pred)

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
n_train = 80
n_test = 20
n_features = 1
noise = 10
random_seed = 15
X, y, coef = make_regression(n_samples=n_train + n_test, n_features=n_features, noise=noise, coef=True, random_state=random_seed)

import matplotlib.pyplot as plt
plt.scatter(X[:,0], y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_train, test_size=n_test, random_state=random_seed)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print ("Regression value: ",r2_score(y_pred, y_test))

# The coefficients
print('Coefficients: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)

plt.figure(figsize=(8, 8))
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='blue', linewidth=3)
#plt.plot([5,2,4], [20,5,80], color='red', linewidth=1) #JUst as an example...
plt.show()
