import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class TwoLayerNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.w1 = np.random.random(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(1, hidden_size)

        self.w2 = np.zeros(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros(1,output_size)

        self.lr = learning_rate
    
    def _relu(self, Z):
        return np.maximum(0, Z)
    
    def _relu_derivative(self, Z):
        return (Z > 0).astype(float)
    
    def forward(self, X):
        self.Z1 = np.dot(self.w1.T,X) + self.b1
        # applying activation function
        self.A1 = self._relu(self.Z1)
        self.Z2 = np.dot(self.w2.T,self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
    
    def backward(self, X,y):
        m = X.shape[0]
        dZ2 = self.A2 - y
        dw2 = (1/m)* np.dot(dZ2,self.A2)
        db2 = (1/m)* np.sum(dZ2, axis = 1, keepdims = True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1* self._relu_derivative(self.A1)
        dw1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        
    def compute_loss(self,Y,Y_hat):
        #binary cross entropy loss
        m = Y.shape[0]
        # Binary Cross-Entropy Loss with epsilon to avoid log(0)
        eps = 1e-15
        Y_hat = np.clip(Y_hat, eps, 1 - eps)
        loss = - (1 / m) * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        return loss


    def fit(self, X,y,epochs):
        for epoch in epochs:
            Y_hat = self.forward(X)
            loss = self.compute_loss(Y, Y_hat)
            self.backward(y,Y_hat)
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")
            
    def predict(self, X):
        preds = self.forward(X)
        return (preds >= 0.5).astype(int)
