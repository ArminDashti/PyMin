# PyMin Classification Module

A comprehensive collection of scikit-learn classification algorithms wrapped for easy use with pandas DataFrames. All classifiers expect the target column to be named 'y' by default.

## Features

- **Unified Interface**: All classifiers inherit from `BasePyMinClassifier` for consistent usage
- **DataFrame Support**: Works directly with pandas DataFrames
- **Automatic Preprocessing**: Built-in feature scaling, selection, and encoding
- **Comprehensive Evaluation**: Built-in metrics and evaluation tools
- **Easy to Use**: Simple, intuitive API for all algorithms

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import pandas as pd
from PyMin.classification import LogisticRegressionClassifier, RandomForestClassifierWrapper

# Create sample data
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [2, 4, 6, 8, 10],
    'y': [0, 0, 1, 1, 1]  # Target column must be named 'y'
})

# Create and train classifier
clf = LogisticRegressionClassifier()
clf.fit(df)

# Make predictions
predictions = clf.predict(df)
probabilities = clf.predict_proba(df)

# Evaluate model
accuracy = clf.score(df)
```

## Available Classifiers

### Linear Models
- `LogisticRegressionClassifier` - Logistic Regression
- `SGDClassifierWrapper` - Stochastic Gradient Descent
- `PerceptronClassifier` - Perceptron
- `PassiveAggressiveClassifierWrapper` - Passive Aggressive
- `RidgeClassifierWrapper` - Ridge Classifier
- `RidgeClassifierCVWrapper` - Ridge Classifier with CV
- `LinearModelsWithScaling` - Linear models with automatic scaling

### Tree Models
- `DecisionTreeClassifierWrapper` - Decision Tree
- `RandomForestClassifierWrapper` - Random Forest
- `GradientBoostingClassifierWrapper` - Gradient Boosting
- `ExtraTreesClassifierWrapper` - Extra Trees
- `AdaBoostClassifierWrapper` - AdaBoost
- `HistGradientBoostingClassifierWrapper` - Histogram-based Gradient Boosting
- `TreeModelsWithFeatureSelection` - Tree models with automatic feature selection

### SVM Models
- `SVCWrapper` - Support Vector Classification
- `LinearSVCWrapper` - Linear Support Vector Classification
- `NuSVCWrapper` - Nu-Support Vector Classification
- `SVMWithScaling` - SVM with automatic scaling
- `SVMWithKernelSelection` - SVM with automatic kernel selection

### Naive Bayes Models
- `GaussianNBWrapper` - Gaussian Naive Bayes
- `MultinomialNBWrapper` - Multinomial Naive Bayes
- `BernoulliNBWrapper` - Bernoulli Naive Bayes
- `ComplementNBWrapper` - Complement Naive Bayes
- `CategoricalNBWrapper` - Categorical Naive Bayes
- `NaiveBayesWithAutoSelection` - Automatic algorithm selection

### Neighbor Models
- `KNeighborsClassifierWrapper` - k-Nearest Neighbors
- `RadiusNeighborsClassifierWrapper` - Radius Neighbors
- `NearestCentroidWrapper` - Nearest Centroid
- `NeighborModelsWithScaling` - Neighbor models with scaling
- `KNNWithAutoK` - k-NN with automatic k selection
- `NeighborModelsWithDistanceWeights` - Distance-weighted neighbors

### Ensemble Models
- `VotingClassifierWrapper` - Voting Classifier
- `BaggingClassifierWrapper` - Bagging Classifier
- `StackingClassifierWrapper` - Stacking Classifier
- `EnsembleWithAutoSelection` - Automatic ensemble selection
- `EnsembleWithCustomWeights` - Custom weighted ensemble

## Convenience Functions

### Create Classifiers
```python
from PyMin.classification import create_classifier, create_linear_model, create_tree_model

# Create any classifier by name
clf = create_classifier('random_forest', n_estimators=100)

# Create with specific options
clf = create_linear_model('logistic', with_scaling=True)
clf = create_tree_model('random_forest', with_feature_selection=True)
clf = create_svm_model('svc', with_scaling=True, auto_kernel=True)
clf = create_naive_bayes_model('auto')  # Auto-select best algorithm
clf = create_neighbor_model('knn', with_scaling=True, auto_k=True)
clf = create_ensemble_model('voting', auto_select=True)
```

## Advanced Features

### Custom Target Column
```python
# Use custom target column name
clf = LogisticRegressionClassifier(target_column='target')
clf.fit(df_with_custom_target)
```

### Train-Test Split
```python
# Built-in train-test split
train_df, test_df = clf.train_test_split(df, test_size=0.2, random_state=42)
```

### Feature Importance
```python
# Get feature importance (for supported models)
importance = clf.get_feature_importance()
print(importance.head())
```

### Detailed Evaluation
```python
# Get comprehensive evaluation metrics
results = clf.evaluate(df, detailed=True)
print(f"Accuracy: {results['accuracy']}")
print(f"Classification Report: {results['classification_report']}")
print(f"Confusion Matrix: {results['confusion_matrix']}")
```

### Predictions and Probabilities
```python
# Make predictions
predictions = clf.predict(df)

# Get class probabilities (for supported models)
probabilities = clf.predict_proba(df)
```

## Example Usage

See `example_usage.py` for comprehensive examples of all classifiers and features.

## Requirements

- scikit-learn >= 1.3.0
- pandas >= 1.5.0
- numpy >= 1.21.0

## API Reference

### BasePyMinClassifier

All classifiers inherit from `BasePyMinClassifier` which provides:

#### Methods
- `fit(df)` - Fit the classifier to training data
- `predict(df)` - Make predictions
- `predict_proba(df)` - Get class probabilities (if supported)
- `score(df)` - Calculate accuracy score
- `evaluate(df, detailed=False)` - Get evaluation metrics
- `train_test_split(df, test_size=0.2, random_state=None)` - Split data
- `get_feature_importance()` - Get feature importance (if supported)

#### Properties
- `is_fitted` - Boolean indicating if model is fitted
- `feature_columns` - List of feature column names
- `target_column` - Name of the target column

## Contributing

Contributions are welcome! Please ensure all new classifiers inherit from `BasePyMinClassifier` and follow the established patterns.

## License

This module is part of the PyMin project and follows the same license terms.
