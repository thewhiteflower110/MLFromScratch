import numpy as np
import heapq 
from collections import Counter
def euclidean_distance (A,B):
    #A = [x1,x2,x3]
    #B = [y1,y1,y3]
    return np.linalg.norm(A-B)
    
class knnClassifier:
    def __init__(self, k):
        self.k = k
        self.X = None
        self.y = None
    
    def fit(self, X,y):
        self.X = X
        self.y = y
    
    def _predict_single(self, x):
        heap = []#(-dist,i)
        #get distance of each points
        for i in range(len(self.X)):
            dist = euclidean_distance(self.X[i],x)
            heapq.heappush(heap,(dist,i))
        
        points_y = [] #(y,dist)
        for j in range(self.k):
            dist,point_idx = heap.heappop(heap)
            points_y.append((self.y[point_idx],dist))
        
        #empty heap
        heap =[]
        #sort points according to distance
        points_y.sort(key=lambda i: i[1])
        # return the class that occurs the max, in case of tie, 
        # use the nearest distance point
        return Counter([label for label, dist in points_y]).most_common(1)

    def predict(self, X):
        return np.array([self._predict_single(X[i]) for i in range(X)])