import numpy as np
from models.model import Model

class KNNModel(Model):
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model with the given data."""
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for the given input data."""
        predictions = np.array([self._predict_single(x) for x in X])
        return predictions
    
    def _predict_single(self, x):
        """Predict class for a single sample"""
        # Calculate distances from sample to all training points
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        # Return most common label
        values, counts = np.unique(k_nearest_labels, return_counts=True)
        return values[np.argmax(counts)]