import numpy as np
from models.model import Model

class LogisticRegressionModel(Model):
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model with the given data."""
        # Add bias feature
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features + 1)
        y = y.astype(float)
        
        # Gradient descent
        for _ in range(self.epochs):
            # Calculate predictions
            z = np.dot(X, self.weights[1:]) + self.weights[0]
            predictions = self._sigmoid(z)
            
            # Calculate error
            errors = y - predictions
            
            # Update weights
            self.weights[0] += self.learning_rate * np.sum(errors)
            self.weights[1:] += self.learning_rate * np.dot(X.T, errors)
        
        return self
    
    def _sigmoid(self, z):
        """Apply sigmoid function"""
        return 1 / (1 + np.exp(-z))
    
    def predict_proba(self, X):
        """Predict probabilities"""
        z = np.dot(X, self.weights[1:]) + self.weights[0]
        return self._sigmoid(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for the given input data."""
        return np.round(self.predict_proba(X)).astype(int)