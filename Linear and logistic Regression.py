import numpy as np


class BaseRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        # Assign the variables
        self.lr = lr
        self.n_iters = n_iters

        # weights and bias
        self.weights, self.bias = None, None

    def fit(self, X, y):
        # initial parameters
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0
        
        # minimizing loss, update weights and bias using gradient decent
        for _ in range(self.n_iters):
            # compute predicted value via wanted model
            y_predicted = self._model(X, self.weights, self.bias)

            # compute derivative of cost function respect to weight and bias
            # cost function use Mean Square Error
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
    def predict(self, X):
        # compute prediced labels via wanted model
        y_predicted = self._predict(X, self.weights, self.bias)
        return y_predicted

    def _model(self, X, w, b):
        raise NotImplementedError

    def _predict(self, X, w, b):
        raise NotImplementedError

class LinearRegression(BaseRegression):
    def _model(self, X, w, b):
        # linear model y = wx + b
        linear_model = np.dot(X, w) + b
        return linear_model

    def _predict(self, X, w, b):
        # linear model y = wx + b, same as in fit method
        linear_model = np.dot(X, w) + b
        return linear_model

class LogisticRegression(BaseRegression):
    def _model(self, X, w, b):
        linear_model = np.dot(X, w) + b

        # applying sigmoid function on linear model
        logistic_model = self._sigmoid(linear_model)

        return logistic_model

    def _predict(self, X, w, b):
        linear_model = np.dot(X, w) + b
        logistic_model = self._sigmoid(linear_model)

        # output label upon the value of logistic model
        pred_label = [1 if i > 0.5 else 0 for i in logistic_model]

        return pred_label

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Testing
if __name__ == "__main__":

    # utils
    def s2_score(y_true, y_pred):
        co_matrix = np.corrcoef(y_true, y_pred)
        return co_matrix[0,1] ** 2

    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def acc(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    # Linear Regression
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    X1, y1 = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=123
    )

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.2, random_state=123
    )

    reg1 = LinearRegression(lr=0.01)
    reg1.fit(X1_train, y1_train)
    prediction1 = reg1.predict(X1_test)

    print(f"Linear Regression loss is : {mse(y1_test, prediction1)}")
    print(f"Linear Regression accuracy is : {s2_score(y1_test, prediction1)}")

    # Logistic Regression
    bc = datasets.load_breast_cancer()
    X2, y2 = bc.data, bc.target

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=123
    )

    reg2 = LogisticRegression(lr=0.01)
    reg2.fit(X2_train, y2_train)
    prediction2 = reg2.predict(X2_test)

    print(f"Logistic Regression loss is : {mse(y2_test, prediction2)}")
    print(f"Logistic Regression accuracy is : {acc(y2_test, prediction2)}")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1,2)
    predicted_line = reg1.predict(X1)
    ax1.scatter(X1, y1, c='blue')
    ax1.plot(X1, predicted_line)
    ax1.set_title("Linear Regression")

    y2_ = ["green" if i == 0 else "red" for i in y2_test]
    prediction2_ = ["green" if i == 0 else "red" for i in prediction2]
    ax2.scatter(X2_test[:,0], X2_test[:,1], c=y2_, marker="o")
    ax2.scatter(X2_test[:,0], X2_test[:,1], c=prediction2_, marker="x", s=30)
    ax2.set_title("Logistic Regression")
    
    plt.show()