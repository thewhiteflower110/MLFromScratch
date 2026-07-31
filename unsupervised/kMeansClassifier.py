import numpy as np

def euclidean_distance(A, B):
    return np.linalg.norm(A - B)

class KMeansClassifier:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []
        self.labels = None

    def fit(self, X):
        self.num_samples, self.num_features = X.shape
        
        # Randomly choose k data points as initial centroids
        random_indices = np.random.choice(self.num_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Assign samples to nearest centroids
            labels = self._assign_clusters(X)
            
            # Calculate new centroids from means
            new_centroids = self._get_new_centroids(X, labels)
            
            # Check for convergence (if centroids stop changing)
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids
            
        self.labels = labels

    def _assign_clusters(self, X):
        labels = []
        for point in X:
            distances = [euclidean_distance(point, c) for c in self.centroids]
            closest_centroid_idx = np.argmin(distances)
            labels.append(closest_centroid_idx)
        return np.array(labels)

    def _get_new_centroids(self, X, labels):
        centroids = np.zeros((self.k, self.num_features))
        for k in range(self.k):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Handle empty cluster by keeping previous centroid
                centroids[k] = self.centroids[k]
        return centroids

    def predict(self, X):
        return self._assign_clusters(X)