import numpy as np
class DBSCAN:
    def __init__(self, epsilon, minpts):
        self.minpts = minpts
        self.epsilon = epsilon
    
    def fit_predict(self, X):
        num_samples = len(X)
        # Initialize labels: 0 = unvisited, -1 = noise, >0 = cluster ID
        self.labels_ = np.zeros(num_samples, dtype=int)
        cluster_id = 0
        self.X = X
        for i in range(num_samples):
            if self.labels_[i] !=0:
                continue
                
            #does any other point exist in nearby radius ?
            neighbors = self._get_neighbors(i)
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                # Expand cluster from this Core Point
                cluster_id += 1
                self.labels_[i] = cluster_id
                self._expand_cluster( neighbors, cluster_id)

        return self.labels_
    
    def _expand_cluster(self,neighbors, cluster_id):
        i = 0
        neighbors = list(neighbors)

        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            # If point was marked as noise, reassign it as a border point of this cluster
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
            
            # If point has not been visited yet, process it
            elif self.labels_[neighbor_idx] == 0:
                self.labels_[neighbor_idx] = cluster_id
            
                # Check if this neighbor is also a Core Point
                new_neighbors = self._get_neighbors(self.X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)
            
            # we dont check the points if they are already assigned, because
            # there won't be any, if there would be, they would have already
            # processed our point to, due to epsilon radius distance
            i+=1
        pass

    def _get_neighbors(self,i):
        distances = np.linalg.norm(self.X - self.X[i], axis=1)
        return np.where(distances <= self.epsilon)[0]
