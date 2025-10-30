# PyMin Scikit-Learn Usage Guide

## Simple Usage Pattern

PyMin makes scikit-learn incredibly simple - you only need to define the algorithm and provide your data!

### Basic Pattern

```python
# 1. Import the algorithm you want
from PyMin.classification.linear_models import create_linear_model
from PyMin.regression.linear_models import LinearRegressionWrapper

# 2. Prepare your data (DataFrame with 'y' column)
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [2, 4, 6, 8, 10],
    'y': [0, 0, 1, 1, 1]  # Target column must be named 'y'
})

# 3. Create model and use it
model = create_linear_model('logistic')  # Just specify algorithm
model.fit(data)                          # Fit the model
predictions = model.predict(data)        # Make predictions
accuracy = model.score(data)             # Get accuracy
```

## Available Algorithms

### Classification
- **Linear Models**: `'logistic'`, `'sgd'`, `'perceptron'`, `'ridge'`
- **Tree Models**: `RandomForestClassifierWrapper`
- **SVM Models**: `SVRWrapper`, `LinearSVRWrapper`

### Regression
- **Linear Models**: `LinearRegressionWrapper`, `RidgeRegressionWrapper`, `LassoRegressionWrapper`
- **Tree Models**: `RandomForestRegressorWrapper`, `DecisionTreeRegressorWrapper`
- **Ensemble Models**: `VotingRegressorWrapper`, `BaggingRegressorWrapper`

## Key Features

### 1. Automatic Feature Scaling
```python
# Automatically applies StandardScaler when needed
model = create_linear_model('logistic', with_scaling=True)
```

### 2. Built-in Evaluation
```python
# Get detailed evaluation metrics
evaluation = model.evaluate(data, detailed=True)
print(f"Accuracy: {evaluation['accuracy']}")
print(f"Classification Report: {evaluation['classification_report']}")
```

### 3. Feature Importance
```python
# Get feature importance (when available)
importance = model.get_feature_importance()
print(importance)
```

### 4. Train-Test Split
```python
# Automatic train-test splitting
train_data, test_data = model.train_test_split(data, test_size=0.2)
```

## Data Requirements

- **DataFrame format**: All data must be in pandas DataFrame
- **Target column**: Must be named `'y'`
- **Feature columns**: Any other columns are treated as features
- **No missing values**: Handle missing values before using PyMin

## Complete Example

```python
import pandas as pd
import numpy as np
from PyMin.classification.linear_models import create_linear_model

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 100),
    'feature2': np.random.normal(0, 1, 100),
    'feature3': np.random.normal(0, 1, 100),
    'y': (np.random.normal(0, 1, 100) > 0).astype(int)
})

# Use the model
model = create_linear_model('logistic', with_scaling=True)
model.fit(data)
predictions = model.predict(data)
accuracy = model.score(data)

print(f"Accuracy: {accuracy:.3f}")
```

## That's It!

No complex setup, no parameter tuning (unless you want it), no data preprocessing - just specify the algorithm and provide your data. PyMin handles the rest!
