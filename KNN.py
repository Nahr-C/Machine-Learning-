import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib.colors import ListedColormap

### Biuld dataset
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# split training data and test data
iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

# plot the first 2 features as a 2D image
#plt.figure()
#plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap, edgecolors='k', s=80)
#plt.show()
#print(X_test.shape)

### Define euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

### Create k nearest neighbots class
class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_label = [self._predict(x) for x in X]
        return np.array(predicted_label)

    def _predict(self, x):
        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote, most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

### Create classifier instance
classifier = KNN(k=3)
# fit sample point 
classifier.fit(X_train, y_train)
# predict label of test point
preidictions = classifier.predict(X_test)

### Compute accuracy of prediction
acc = np.sum(preidictions == y_test) / len(y_test)
print(acc)