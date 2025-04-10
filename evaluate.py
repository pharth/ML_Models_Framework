import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models.model import Model
from models.knn import KNNModel
from models.logistic import LogisticRegressionModel
from models.pca import PCAModel

from data.data_processing import read_csv, split_data

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate machine learning models')
    parser.add_argument('model_name', type=str, choices=['knn', 'logistic', 'pca'], 
                        help='Model to evaluate (knn, logistic, or pca)')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset CSV file')
    parser.add_argument('shuffle', type=str, choices=['true', 'false'], 
                        help='Whether to shuffle data before splitting (true/false)')
    
    # KNN specific parameters
    parser.add_argument('--k', type=int, default=3, help='Number of neighbors for KNN model')
    
    # Logistic Regression specific parameters
    parser.add_argument('--learning-rate', type=float, default=0.01, 
                        help='Learning rate for Logistic Regression model')
    parser.add_argument('--epochs', type=int, default=1000, 
                        help='Number of epochs for Logistic Regression model')
    
    # PCA specific parameters
    parser.add_argument('--n-components', type=int, default=2, 
                        help='Number of components for PCA model')
    parser.add_argument('--visualize', type=str, choices=['true', 'false'], default='false',
                        help='Whether to visualize PCA results (true/false)')
    
    return parser.parse_args()

def create_model(args) -> Model:
    """Create and return the specified model with the provided parameters."""
    if args.model_name == 'knn':
        return KNNModel(k=args.k)
    elif args.model_name == 'logistic':
        return LogisticRegressionModel(learning_rate=args.learning_rate, epochs=args.epochs)
    elif args.model_name == 'pca':
        return PCAModel(n_components=args.n_components)
    else:
        raise ValueError(f"Unknown model: {args.model_name}")

def visualize_pca(pca_model, X_transformed, y):
    """Visualize PCA results."""
    plt.figure(figsize=(10, 8))
    
    # Convert y to numeric if it's not already
    y_numeric = y.astype(float)
    
    # Get unique classes
    unique_classes = np.unique(y_numeric)
    
    # Plot each class
    for cls in unique_classes:
        mask = (y_numeric == cls)
        plt.scatter(X_transformed[mask, 0], 
                    X_transformed[mask, 1] if X_transformed.shape[1] > 1 else np.zeros(np.sum(mask)), 
                    label=f'Class {cls}',
                    alpha=0.7)
    
    plt.title('PCA Visualization')
    plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio[0]*100:.2f}%)')
    
    if X_transformed.shape[1] > 1:
        plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio[1]*100:.2f}%)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig('pca_visualization.png')
    print("PCA visualization saved to 'pca_visualization.png'")

if __name__ == "__main__":
    args = parse_arguments()
    
    try:
        print(f"Loading data from {args.dataset_path}")
        data = read_csv(args.dataset_path)
        shuffle_flag = args.shuffle.lower() == 'true'
        train, test = split_data(data, shuffle_flag)
        
        X_train = train[:, :-1].astype(float)
        y_train = train[:, -1]
        X_test = test[:, :-1].astype(float)
        y_test = test[:, -1]
        
        model = create_model(args)
        model_name = args.model_name.capitalize()
        
        print(f"Training {model_name} model...")
        model.fit(X_train, y_train)
        
        # Handle PCA differently
        if args.model_name == 'pca':
            print(f"Explained variance ratio: {model.explained_variance_ratio}")
            
            # Transform the data
            X_train_transformed = model.transform(X_train)
            X_test_transformed = model.transform(X_test)
            
            print(f"Original data shape: {X_train.shape}")
            print(f"Transformed data shape: {X_train_transformed.shape}")
            
            # Calculate reconstruction error
            X_train_reconstructed = model.inverse_transform(X_train_transformed)
            reconstruction_error = np.mean(np.sum((X_train - X_train_reconstructed) ** 2, axis=1))
            print(f"Mean squared reconstruction error: {reconstruction_error:.6f}")
            
            # Visualize if requested
            if args.visualize.lower() == 'true':
                visualize_pca(model, X_train_transformed, y_train)
        else:
            print(f"Evaluating {model_name} model...")
            accuracy = model.evaluate(X_test, y_test)
            print(f"{model_name} Model Accuracy: {accuracy * 100:.2f}%")
        
        # Print model-specific parameters
        if args.model_name == 'knn':
            print(f"Model parameters: k={args.k}")
        elif args.model_name == 'logistic':
            print(f"Model parameters: learning_rate={args.learning_rate}, epochs={args.epochs}")
        elif args.model_name == 'pca':
            print(f"Model parameters: n_components={args.n_components}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)