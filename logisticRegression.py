import numpy as np

class linearRegression:
    def __init__(self,learning_rate, lambda_ = None):
        self.lr = learning_rate
        self.w = None
        self.b = None
        self.lambda_ = lambda_
    
    def _sigmoid(self, z):
        # Clip values to prevent overflow errors in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self,X,y,iterations, regularization):
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0.0

        for _ in iterations:
            z = np.dot(X,self.w) + self.b #array of (samples,1)
            y_hat = self._sigmoid(z)

            loss = - (1/num_features)* np.sum(np.dot(y,np.log(y_hat)) + np.dot((1-y),np.log(1 - y_hat)))
            #use loss to make a graph
            dw = (1/num_features) * np.dot(X.T,(y_hat-y))
            db = (1/num_samples) * np.sum(y_hat - y)
            if regularization==True:
                #L2 regularization (Ridge)
                dw += (self.lambda_ / num_samples) * self.w
                #L1 regularixation (Lasso)
                # dw += (self.lambda_ / num_samples) * np.sign(self.weights)
            self.w = - self.lr * dw
            self.b = - self.lr * db

    def predict_prob(self,X):
        z = np.dot(X,self.w) + self.b #array of (samples,1)
        y_hat = self._sigmoid(z)
        return y_hat

    def predict(self, y_hat, threshold):
        if y_hat>=threshold:
            return 1
        else:
            return 0
