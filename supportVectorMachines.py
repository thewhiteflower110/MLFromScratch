import numpy as np
class SVM:
    def __init__(self, c, lr):
        self.w = None
        self.b = None
        self.c = c
        self.lr = lr

    def fit(self,X,y,iterations):
        num_samples,num_features = X.shape 
        self.b = 0.0
        self.w = np.zeros(num_features)
        
        for _ in iterations:
            #run through all the features
            for i, xi in X:
                z = np.dot(self.w.T, xi) + self.b
                condition = y[i]*z[i]
                if condition >= 1:
                    # Only apply regularization gradient
                    dw = self.w 
                    db = 0
                else:
                    # Apply regularization + misclassification loss gradient
                    dw = self.w - self.c * y[i]*xi
                    db = - self.c*y[i]
                # Parameter updates
                self.w -= self.lr * dw
                self.b -= self.lr * db
            