# PyMin Regression Module

This module provides comprehensive scikit-learn regression model wrappers that work seamlessly with pandas DataFrames. All models expect the target column to be named 'y'.

## Features

- **Unified Interface**: All models follow the same API pattern
- **DataFrame Support**: Works directly with pandas DataFrames
- **Automatic Feature Selection**: Automatically uses all columns except 'y' as features
- **Built-in Evaluation**: Built-in scoring with R², MSE, and MAE metrics
- **Feature Importance**: Automatic feature importance extraction where available
- **CLI Integration**: Full command-line interface support

## Available Models

### Linear Models
- `LinearRegressionWrapper` - Ordinary least squares linear regression
- `RidgeRegressionWrapper` - Ridge regression with L2 regularization
- `LassoRegressionWrapper` - Lasso regression with L1 regularization
- `ElasticNetRegressionWrapper` - ElasticNet regression with L1+L2 regularization

### Tree-based Models
- `DecisionTreeRegressorWrapper` - Decision tree regression
- `RandomForestRegressorWrapper` - Random forest regression
- `GradientBoostingRegressorWrapper` - Gradient boosting regression

### Support Vector Machines
- `SVRWrapper` - Support Vector Regression with various kernels
- `LinearSVRWrapper` - Linear Support Vector Regression

### Ensemble Methods
- `VotingRegressorWrapper` - Voting regressor for combining multiple models
- `BaggingRegressorWrapper` - Bagging regressor for bootstrap aggregating
- `AdaBoostRegressorWrapper` - AdaBoost regressor for adaptive boosting

### Neural Networks
- `MLPRegressorWrapper` - Multi-layer Perceptron regressor

## Usage

### Python API

```python
import pandas as pd
from regression.linear_models import RidgeRegressionWrapper
from regression.tree_models import RandomForestRegressorWrapper

# Load your data (must have column 'y' for target)
df = pd.read_csv('your_data.csv')

# Split data
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Linear regression
ridge = RidgeRegressionWrapper(alpha=1.0)
ridge.fit(train_df)
predictions = ridge.predict(test_df)
score = ridge.score(test_df, metric='r2')
print(f"R² Score: {score:.4f}")

# Tree-based regression
rf = RandomForestRegressorWrapper(n_estimators=100, random_state=42)
rf.fit(train_df)
score = rf.score(test_df)
importance = rf.get_feature_importance()
print(f"R² Score: {score:.4f}")
print("Feature importance:")
print(importance.head())
```

### Command Line Interface

```bash
# Linear regression
pymin regression linear --data data.csv --model ridge --alpha 0.1

# Tree-based models
pymin regression tree --data data.csv --model random_forest --n-estimators 200

# SVM models
pymin regression svm --data data.csv --model svr --kernel rbf --c 1.0
```

## Data Format

Your data must be in CSV format with:
- Feature columns: Any number of columns with feature data
- Target column: Must be named exactly 'y'

Example:
```csv
feature_1,feature_2,feature_3,y
1.2,3.4,5.6,10.1
2.1,4.3,6.5,11.2
...
```

## Model Parameters

All models support their respective scikit-learn parameters. See individual model documentation for details:

- **Linear Models**: `alpha`, `fit_intercept`, `max_iter`, etc.
- **Tree Models**: `max_depth`, `min_samples_split`, `n_estimators`, etc.
- **SVM Models**: `kernel`, `C`, `gamma`, `epsilon`, etc.
- **Ensemble Models**: `n_estimators`, `learning_rate`, `base_estimator`, etc.
- **Neural Networks**: `hidden_layer_sizes`, `activation`, `solver`, etc.

## Evaluation Metrics

Available metrics:
- `r2`: R-squared score (default)
- `mse`: Mean Squared Error
- `mae`: Mean Absolute Error

```python
# Get different metrics
r2_score = model.score(test_df, metric='r2')
mse = model.score(test_df, metric='mse')
mae = model.score(test_df, metric='mae')
```

## Feature Importance

Models that support feature importance will automatically provide it:

```python
importance = model.get_feature_importance()
if importance is not None:
    print("Top features:")
    print(importance.head())
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Examples

See `example_usage.py` for comprehensive examples of all models.

## Error Handling

The module includes robust error handling:
- Validates that 'y' column exists in the DataFrame
- Ensures model is fitted before making predictions
- Provides clear error messages for common issues

## Performance Tips

1. **Data Preprocessing**: Scale your features for better performance with linear models and SVMs
2. **Cross-validation**: Use sklearn's cross-validation tools for robust evaluation
3. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV for optimal parameters
4. **Feature Selection**: Consider feature selection for high-dimensional data

## Contributing

When adding new models:
1. Inherit from `BaseRegressionWrapper`
2. Implement `_create_model()` method
3. Add comprehensive docstring with parameters
4. Update `__init__.py` exports
5. Add CLI support if needed
6. Include examples in `example_usage.py`
