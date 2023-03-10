import numpy as np

class SVM:

    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i,  y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)


if __name__ == "__main__":

    # import 
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # dataset
    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=650
    )

    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # utils
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    # instance 
    clf = SVM()
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X[:,0], X[:,1], marker='o', c=y)
    plt.scatter(X_test[:,0], X_test[:,1], marker='x', c=prediction, s=20)

    p1_x = np.amin(X[:,0])
    p2_x = np.amax(X[:,0])

    p1_y = get_hyperplane_value(p1_x, clf.w, clf.b, 0)
    p2_y = get_hyperplane_value(p2_x, clf.w, clf.b, 0)

    p1_y_m = get_hyperplane_value(p1_x, clf.w, clf.b, -1)
    p2_y_m = get_hyperplane_value(p2_x, clf.w, clf.b, -1)

    p1_y_p = get_hyperplane_value(p1_x, clf.w, clf.b, 1)
    p2_y_p = get_hyperplane_value(p2_x, clf.w, clf.b, 1)

    ax.plot([p1_x, p2_x], [p1_y, p2_y], 'y--')
    ax.plot([p1_x, p2_x], [p1_y_m, p2_y_m], 'k')
    ax.plot([p1_x, p2_x], [p1_y_p, p2_y_p], 'k')

    y_min = np.amin(X[:,1])
    y_max = np.amax(X[:,1])
    ax.set_ylim([y_min-3, y_max+3])

    plt.show()
