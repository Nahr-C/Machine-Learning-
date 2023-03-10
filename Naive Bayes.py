import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        # get sample numbers and feature numbers
        n_samples, n_features = X.shape

        # get unique class labels and numbers
        self._class = np.unique(y)
        n_classes = len(self._class)

        # init mean, var, priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._prior = np.zeros(n_classes, dtype=np.float64)

        # loop for each class
        for idx, cls in enumerate(self._class):
            # get samples of current class
            X_c = X[cls == y]

            # compute mean, variance, prior of this class
            self._mean[idx, :] = np.mean(X_c, axis=0)
            self._var[idx, :] = np.var(X_c, axis=0)
            self._prior[idx] = X_c.shape[0] / np.float64(n_samples) 

    def predict(self, X):
        # use helper function to compute each sample 
        y_predicted = [self._predict(x) for x in X]
        return y_predicted

    def _predict(self, x):
        # biuld empty list for posteriors
        posteriors = []

        # loop for each class, calculate the possibilities, then add to biulded list
        for idx, cls in enumerate(self._class):
            prior = np.log(self._prior[idx])
            likelihood = np.sum(np.log(self._gaussian(idx, x)))
            posterior = likelihood + prior
            posteriors.append(posterior)

        # find the indices of maximum value, return the corresponding class label
        return self._class[np.argmax(posteriors)]

    def _gaussian(self, idx, x):
        # compute mean and variance 
        mean = self._mean[idx]
        var = self._var[idx]

        # variance equals to squre of std
        numerator = np.exp(- ((x-mean)**2) / (2*(var)))
        demoninator = np.sqrt(2 * np.pi * var)
        return numerator / demoninator


if __name__ == "__main__":

    # utils
    def acc(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    # dataset
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print("Naive Bayes classification accuracy", acc(y_test, predictions))

    # plot
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8,6))
    y_ = ["green" if i == 1 else "red" for i in y_test]
    predictions_ = ["green" if i == 1 else "red" for i in predictions]

    plt.scatter(X_test[:,0], X_test[:,1], c=y_, marker="o")
    plt.scatter(X_test[:,0], X_test[:,1], c=predictions_, marker="x", s=25)

    plt.show()
