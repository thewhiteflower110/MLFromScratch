#linear regression with one or multiple features and regularization
import numpy as np

class linearRegression:
    def __init__(self,learning_rate, lambda_ = None):
        self.lr = learning_rate
        self.weights = None
        self.bias = None
        self.lambda_ = lambda_

    def fit(self,X,y,iterations, regularization):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0.0
        if regularization == True and self.lambda_==None:
            self.lambda_ = 0.001
        for _ in iterations:
            y_hat = np.dot(X,self.weights) + self.bias
            # (samples,features) dot (features,1 )
            # cost_fn = 0.5*np.sum((y_hat - y)**2) /num_samples
            dw = (1/num_samples) * np.dot(X.T,y_hat - y)
            db = (1/num_samples) * np.sum(y_hat - y)
            if regularization==True:
                #L2 regularization (Ridge)
                dw += (self.lambda_ / num_samples) * self.weights
                #L1 regularixation (Lasso)
                # dw += (self.lambda_ / num_samples) * np.sign(self.weights)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self,X):
        return np.dot(X,self.weights)+ self.bias