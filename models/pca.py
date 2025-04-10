import numpy as np
from models.model import Model

class PCAModel(Model):
    def __init__(self, n_components=2):
        """
        Principal Component Analysis (PCA) implementation.
        
        Parameters:
        -----------
        n_components : int, default=2
            Number of components to keep after dimensionality reduction.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None
    
    def fit(self, X: np.ndarray, y: np.ndarray=None):
        """
        Fit the PCA model with the given data.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray, default=None
            Not used, present for API consistency.
        
        Returns:
        --------
        self : object
            Returns self.
        """
        # Store the mean of the data
        self.mean = np.mean(X, axis=0)
        
        # Center the data
        X_centered = X - self.mean
        
        # Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by decreasing eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store the first n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]
        
        # Store explained variance and ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / total_variance
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X to the principal component space.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.
        
        Returns:
        --------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        # Center the data
        X_centered = X - self.mean
        
        # Project the data onto the principal components
        X_transformed = np.dot(X_centered, self.components)
        
        return X_transformed
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to the original space.
        
        Parameters:
        -----------
        X_transformed : ndarray of shape (n_samples, n_components)
            Data in the principal component space.
        
        Returns:
        --------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Reconstructed data in the original space.
        """
        # Project back to the original space
        X_reconstructed = np.dot(X_transformed, self.components.T) + self.mean
        
        return X_reconstructed
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict function implemented for compatibility with Model interface.
        For PCA, it transforms and then reconstructs the data.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.
        
        Returns:
        --------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Reconstructed data.
        """
        # This is a unique case where 'predict' doesn't make traditional sense for PCA
        # We'll define it as transform followed by inverse_transform
        return self.inverse_transform(self.transform(X))