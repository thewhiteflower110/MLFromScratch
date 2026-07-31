import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.X = X
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        #get covariance matrix of the columns
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
 
        # 4. Sort eigenvalues and corresponding eigenvectors in descending order
        # np.linalg.eigh returns them in ascending order, so we reverse them
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        # Select the top n_components eigenvectors (columns)
        self.components = sorted_eigenvectors[:, :self.n_components]
        
        # Calculate explained variance ratio
        # total_variance = np.sum(sorted_eigenvalues)
        # self.explained_variance_ratio_ = sorted_eigenvalues[:self.n_components] / total_variance


    def transform(self, X):
        # Center the data using the saved training mean
        X_centered = X - self.mean
        # Project data onto the principal components: (N, features) @ (features, k) -> (N, k)
        return np.dot(X_centered, self.components)
    
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
