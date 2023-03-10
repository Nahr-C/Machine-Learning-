import numpy as np
import matplotlib.pyplot as plt

class Kmeans:

    def __init__(self, K=5, n_iters=150, plot=False):
        self.K = K
        self.n_iters = n_iters
        self.plot = plot
        self.clusters, self.centroids = None, None

    def predict(self, X):
        self.n_samples, self.n_features = X.shape
        self.X = X

        random_sample_idx = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[i] for i in random_sample_idx]

        for _ in range(self.n_iters):

            # update clusters
            self.clusters = self._get_clusters(self.centroids)
            if self.plot:
                self._plot()

            # update centroids
            centroid_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self.plot:
                self._plot()

            # check if converged
            if self._is_converged(centroid_old, self.centroids):
                break

        return self._get_label(self.clusters)

    def _get_label(self, clusters):
        label = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                label[sample_idx] = cluster_idx
        return label

    def _get_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for sample_idx, sample in enumerate(self.X):
            closest_idx = self._get_closest_centroid(sample, centroids)
            clusters[closest_idx].append(sample_idx)
        return clusters

    def _get_closest_centroid(self, sample, centroids):
        distances = [self._euclidean_distance(sample, centroid) for centroid in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            centroid_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = centroid_mean
        return centroids

    def _is_converged(self, centroid_old, centroid_new):
        distance = [self._euclidean_distance(centroid_old[i], centroid_new[i]) for i in range(self.K)]
        return sum(distance) == 0

    def _plot(self):
        fig, ax = plt.subplots(figsize=(12,8))

        for cluster_idx, cluster in enumerate(self.clusters):
            point = self.X[cluster]
            ax.scatter(point[:,0], point[:,1])

        for centroid in self.centroids:
            ax.scatter(centroid[0], centroid[1], c="black", marker="x", linewidths=2)

        plt.show()


if __name__ == "__main__":

    # import 
    from sklearn.datasets import make_blobs

    # dataset
    X, y = make_blobs(
        n_samples=150, n_features=2, centers=8, shuffle=True, random_state=145
    )

    # instance
    k = Kmeans(plot=True)
    k.predict(X)
