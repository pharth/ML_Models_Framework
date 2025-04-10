import sys
import argparse
import numpy as np

from models.model import Model
from models.knn import KNNModel
from models.logistic import LogisticRegressionModel

from data.data_processing import read_csv, split_data

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate machine learning models')
    parser.add_argument('model_name', type=str, choices=['knn', 'logistic'], 
                        help='Model to evaluate (knn or logistic)')
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
    
    return parser.parse_args()

def create_model(args) -> Model:
    """Create and return the specified model with the provided parameters."""
    if args.model_name == 'knn':
        return KNNModel(k=args.k)
    elif args.model_name == 'logistic':
        return LogisticRegressionModel(learning_rate=args.learning_rate, epochs=args.epochs)
    else:
        raise ValueError(f"Unknown model: {args.model_name}")

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
        
        print(f"Evaluating {model_name} model...")
        accuracy = model.evaluate(X_test, y_test)
        
        print(f"{model_name} Model Accuracy: {accuracy * 100:.2f}%")
        
        # Print model-specific parameters
        if args.model_name == 'knn':
            print(f"Model parameters: k={args.k}")
        elif args.model_name == 'logistic':
            print(f"Model parameters: learning_rate={args.learning_rate}, epochs={args.epochs}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)