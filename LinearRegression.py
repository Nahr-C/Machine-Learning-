import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

### Create dataset
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

### Show sample figures
#fig = plt.figure(figsize=(8,6))
#plt.scatter(X, y)
#plt.show()

### Define linear regression class
class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initial parameters
        n_samples, n_feathers = X.shape
        self.weights = np.zeros(n_feathers)
        self.bias = 0

        # gradient decent
        for _ in range(self.n_iters):
            # compute predicted value y = wx + b
            y_predicted = np.dot(X,self.weights) + self.bias

            # compute derivative of cost function respect to weight and bias
            # cost function use Mean Square Error
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            # update weights and bias 
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

### Create regressor instance to predict
regressor = LinearRegression(lr=0.01)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

# define MSE to compute final loss
def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)

mse_value = mse(y_test, predicted)
print(mse_value)

### Visualization of curve
# get all predicted value
y_pred_line = regressor.predict(X)
# create figure
fig = plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color='red')
plt.scatter(X_test, y_test, color='green')
plt.plot(X, y_pred_line, label = "Prediction")
plt.show()

