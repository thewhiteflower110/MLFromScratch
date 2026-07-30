#P(xi|Y) = Count(xi,Y) + alpha / Count(Y) + alpha |V|
# Alpha adds Laplace Smooothening, becuase if one of the P(xi) is not presen
#t, it will be 0, hence making everything else as 0
#V =(where V is the total number of unique features/vocabulary size).
import numpy as np
class naiveBayes:
    def __init__(self):
        pass
    
    def fit(self, X,y):
        self.classes = np.unique(y)
        self.means = {}
        self.variances = {}
        self.priors = {}

        # we find P(y|xi) for all classes
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + 1e-9  # added smooth to avoid division by zero
            #P(Y)
            self.priors[c] = X_c.shape[0] / X.shape[0]
    
    # We assume that each feature xi is distributed normally
    def _calculate_likelihood(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.variances[class_idx]
        # Gaussian Probability Density Function
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    # P(x_i|C)
    def _predict_single(self, x):
        posteriors = []
        
        for c in self.classes:
            prior = np.log(self.priors[c])
            # Sum log-likelihoods to avoid numerical underflow
            likelihood = np.sum(np.log(self._calculate_likelihood(c, x)))
            # P(C|X) -> becuase we used log, we are adding not multiplying
            posterior = prior + likelihood
            posteriors.append((posterior, c))
            
        # Return class with the highest posterior probability
        return max(posteriors, key=lambda item: item[0])[1]

    def predict(self,X):
        return [self._predict_single(x) for x in X]