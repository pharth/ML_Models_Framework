from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model with the given data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for the given input data."""
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy: (correct predictions / total samples)."""
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy