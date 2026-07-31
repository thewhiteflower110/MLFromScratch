import numpy as np
def euclidean_distance(A, B):
    return np.linalg.norm(A - B)

class hieararchicalClustering:
    # Using Aglomerative Strategy
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.labels_ = None
    
    def _cluster_distance(self, cluster1, cluster2):
        """Single linkage: Minimum distance between any point in cluster1 and cluster2."""
        min_dist = float('inf')
        for p1 in cluster1:
            for p2 in cluster2:
                dist = euclidean_distance(p1, p2)
                if dist < min_dist:
                    min_dist = dist
        return min_dist
    
    def fit_predict(self, X):
        # num_samples,num_features = X.shape

        #consider each point as a cluster.
        clusters = [i for i in X]
        cluster_indices = [[i] for i in range(len(X))]
        while(len(clusters) > self.n_clusters):
            min_dist = float('inf')
            closest_pair = (0, 1)
            
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    dist = self._cluster_distance(clusters[i],clusters[j])
                    if dist < min_dist:
                        min_dist =  dist
                        closest_pair = (i, j)
            
            i, j = closest_pair

            # Merge cluster j into cluster i
            clusters[i].extend(clusters[j])
            cluster_indices[i].extend(cluster_indices[j])

            # Remove merged cluster j
            clusters.pop(j)
            cluster_indices.pop(j)

        # Assign final cluster labels
        self.labels_ = np.zeros(len(X), dtype=int)
        for cluster_label, indices in enumerate(cluster_indices):
            for idx in indices:
                self.labels_[idx] = cluster_label

        return self.labels_