# 📊 Machine Learning Models Framework

Hey there! 👋 Welcome to my Applied Data Science Assignment project. This repository contains a flexible machine learning framework I built as part of my coursework, featuring different classification algorithms with a simple interface.

## ✨ What's This All About?

This project implements a bunch of machine learning models from scratch using NumPy! I've created a modular framework where all models follow the same interface, making it easy to swap between algorithms and compare their performance on different datasets.


## ✅ Features

- **Common Interface**: All models share the same methods, so they're easy to use
- **Pure Python/NumPy**: Models implemented from scratch (no scikit-learn!)
- **Easy Testing**: Simple command-line tool to evaluate model performance
- **Customizable**: Tweak hyperparameters through command-line arguments


## 🚀 How to Use It

It's super simple! Just use the command-line interface:

```bash
python evaluate.py <model_name> <dataset_path> <true/false for shuffle> [--extra-parameters]
```

### Examples

```bash
# Try KNN with default settings
python evaluate.py knn data/dataset2.csv true

# Or customize it a bit
python evaluate.py knn data/dataset3.csv false --k 7

# Let's test logistic regression
python evaluate.py logistic data/dataset1.csv true

# Fine-tune logistic regression
python evaluate.py logistic data/dataset2.csv false --learning-rate 0.005 --epochs 2000
```

## 📋 Data Format

The program expects:
- CSV files with features followed by a target column
- Numeric values (please encode categorical features first)
- The last column should be the label/target

## 🔧 Extending with New Models

Want to add your own model? It's easy:

1. Create a new file in the `models/` folder
2. Make a class that inherits from `Model`
3. Implement the required `train()` and `predict()` methods
4. Update the `create_model()` function in `evaluate.py`

## 📦 Requirements

- Python 3.6+
- NumPy

## 🤝 Contributions Welcome!

This project is part of my Applied Data Science assignment, but I'm definitely open to contributions! If you're interested in helping out, here are some ways:

- **Add new models**: SVM, Decision Trees, Random Forest, etc.
- **Improve existing implementations**: Optimizations, better documentation
- **Add features**: Cross-validation, confusion matrix, F1 scores
- **Fix bugs**: Found something that doesn't work? Let me know!

To contribute:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-model`)
3. Commit your changes (`git commit -m 'Add amazing new model'`)
4. Push to the branch (`git push origin feature/amazing-model`)
5. Open a Pull Request


## 📝 Assignment Context

This project was created as part of my Applied Data Science coursework. The goal was to implement machine learning algorithms from scratch and create a flexible framework for model evaluation and comparison.
