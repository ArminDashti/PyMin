# PyMin Simple ML Interface

The Simple ML Interface provides an easy-to-use way to test machine learning algorithms with just a few parameters: algorithm name, DataFrame, and test size.

## Quick Start

```python
from PyMin import simple_regression, simple_classification, quick_regression, quick_classification
import pandas as pd

# Your DataFrame must have a column named 'y' for the target variable
df = pd.read_csv('your_data.csv')

# Test a specific algorithm
result = simple_regression('random_forest', df, test_size=0.2)
print(f"R² Score: {result['r2']:.4f}")

# Test all algorithms and get the best ones
all_results = quick_regression(df, test_size=0.2)
print(f"Best algorithm: {all_results[0]['algorithm']} with R² = {all_results[0]['r2']:.4f}")
```

## Available Functions

### `simple_regression(algorithm, df, test_size=0.2, **kwargs)`
Test a specific regression algorithm.

**Parameters:**
- `algorithm` (str or None): Algorithm name (e.g., 'random_forest', 'linear', 'svr'). If None, tests all algorithms.
- `df` (DataFrame): DataFrame with features and target column 'y'
- `test_size` (float): Proportion of data to use for testing (default: 0.2)
- `**kwargs`: Additional parameters for the specific algorithm

**Returns:**
- If algorithm specified: Dictionary with results for that algorithm
- If algorithm is None: List of dictionaries with results for all algorithms (sorted by R² score)

### `simple_classification(algorithm, df, test_size=0.2, **kwargs)`
Test a specific classification algorithm.

**Parameters:**
- `algorithm` (str or None): Algorithm name (e.g., 'random_forest', 'logistic', 'svc'). If None, tests all algorithms.
- `df` (DataFrame): DataFrame with features and target column 'y'
- `test_size` (float): Proportion of data to use for testing (default: 0.2)
- `**kwargs`: Additional parameters for the specific algorithm

**Returns:**
- If algorithm specified: Dictionary with results for that algorithm
- If algorithm is None: List of dictionaries with results for all algorithms (sorted by accuracy)

### `quick_regression(df, test_size=0.2)`
Convenience function to test all regression algorithms quickly.

### `quick_classification(df, test_size=0.2)`
Convenience function to test all classification algorithms quickly.

### `get_available_algorithms()`
Get list of all available algorithm names.

## Available Algorithms

### Regression Algorithms
- `linear`: Linear Regression
- `ridge`: Ridge Regression
- `lasso`: Lasso Regression
- `elastic_net`: Elastic Net Regression
- `decision_tree`: Decision Tree Regressor
- `random_forest`: Random Forest Regressor
- `gradient_boosting`: Gradient Boosting Regressor
- `svr`: Support Vector Regressor
- `linear_svr`: Linear Support Vector Regressor
- `voting`: Voting Regressor
- `bagging`: Bagging Regressor
- `adaboost`: AdaBoost Regressor
- `mlp`: Multi-layer Perceptron Regressor

### Classification Algorithms
- `logistic`: Logistic Regression
- `sgd`: SGD Classifier
- `perceptron`: Perceptron
- `passive_aggressive`: Passive Aggressive Classifier
- `ridge`: Ridge Classifier
- `ridge_cv`: Ridge Classifier with CV
- `decision_tree`: Decision Tree Classifier
- `random_forest`: Random Forest Classifier
- `gradient_boosting`: Gradient Boosting Classifier
- `extra_trees`: Extra Trees Classifier
- `adaboost`: AdaBoost Classifier
- `hist_gradient_boosting`: Histogram Gradient Boosting Classifier
- `svc`: Support Vector Classifier
- `linear_svc`: Linear Support Vector Classifier
- `nu_svc`: Nu-Support Vector Classifier
- `gaussian_nb`: Gaussian Naive Bayes
- `multinomial_nb`: Multinomial Naive Bayes
- `bernoulli_nb`: Bernoulli Naive Bayes
- `complement_nb`: Complement Naive Bayes
- `categorical_nb`: Categorical Naive Bayes
- `knn`: k-Nearest Neighbors
- `radius_neighbors`: Radius Neighbors
- `nearest_centroid`: Nearest Centroid
- `voting`: Voting Classifier
- `bagging`: Bagging Classifier
- `stacking`: Stacking Classifier

## Example Results

### Regression Results
```python
{
    'algorithm': 'random_forest',
    'mse': 0.1234,
    'mae': 0.2345,
    'r2': 0.8765,
    'predictions': array([...]),
    'feature_importance': Series([...]),
    'model': RandomForestRegressor(...),
    'status': 'success'
}
```

### Classification Results
```python
{
    'algorithm': 'random_forest',
    'accuracy': 0.9234,
    'classification_report': {...},
    'confusion_matrix': [[...], [...]],
    'predictions': array([...]),
    'probabilities': array([...]),
    'feature_importance': Series([...]),
    'model': RandomForestClassifier(...),
    'status': 'success'
}
```

## Complete Example

```python
import pandas as pd
import numpy as np
from PyMin import simple_regression, simple_classification

# Create sample data
np.random.seed(42)
X = np.random.randn(1000, 5)
y = 2 * X[:, 0] + 1.5 * X[:, 1] + np.random.normal(0, 0.1, 1000)

df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
df['y'] = y

# Test specific algorithm
result = simple_regression('random_forest', df, test_size=0.2)
print(f"Random Forest R²: {result['r2']:.4f}")

# Test all algorithms
all_results = simple_regression(None, df, test_size=0.2)
print("Top 3 algorithms:")
for i, r in enumerate(all_results[:3]):
    print(f"{i+1}. {r['algorithm']}: R² = {r['r2']:.4f}")
```

## Requirements

- pandas
- numpy
- scikit-learn
- All PyMin regression and classification modules

## Notes

- Your DataFrame must contain a column named 'y' for the target variable
- All algorithms are automatically configured with reasonable default parameters
- Results are automatically sorted by performance (R² for regression, accuracy for classification)
- Feature importance is included when available
- Failed algorithms are included in results with error information
