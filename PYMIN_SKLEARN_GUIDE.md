# PyMin Scikit-Learn Usage Guide

## The Simple Pattern: Algorithm + Data = Results

PyMin makes scikit-learn incredibly simple by following this pattern:

**You only need to define:**
1. **Algorithm** - Just specify the algorithm name
2. **Data** - Provide a DataFrame with target column named 'y'

**That's it!** PyMin handles everything else.

## Quick Start

```python
# 1. Import the algorithm
from PyMin.classification.linear_models import create_linear_model

# 2. Define your data (DataFrame with 'y' column)
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [2, 4, 6, 8, 10],
    'y': [0, 0, 1, 1, 1]  # Target column must be named 'y'
})

# 3. Use the model
model = create_linear_model('logistic')  # Just specify algorithm
model.fit(data)                          # Fit the model
predictions = model.predict(data)        # Make predictions
accuracy = model.score(data)             # Get accuracy
```

## Available Algorithms

### Classification
```python
# Linear Models
from PyMin.classification.linear_models import create_linear_model

model = create_linear_model('logistic')     # Logistic Regression
model = create_linear_model('sgd')          # SGD Classifier
model = create_linear_model('perceptron')   # Perceptron
model = create_linear_model('ridge')        # Ridge Classifier

# Tree Models
from PyMin.classification.tree_models import create_tree_model

model = create_tree_model('random_forest')      # Random Forest
model = create_tree_model('decision_tree')      # Decision Tree
model = create_tree_model('gradient_boosting')  # Gradient Boosting
model = create_tree_model('extra_trees')        # Extra Trees
```

### Regression
```python
# Linear Models
from PyMin.regression.linear_models import LinearRegressionWrapper, RidgeRegressionWrapper

model = LinearRegressionWrapper()  # Linear Regression
model = RidgeRegressionWrapper()   # Ridge Regression

# Tree Models
from PyMin.regression.tree_models import RandomForestRegressorWrapper

model = RandomForestRegressorWrapper()  # Random Forest
```

## Data Requirements

### Format
- **DataFrame**: All data must be in pandas DataFrame
- **Target Column**: Must be named `'y'`
- **Features**: Any other columns are treated as features
- **No Missing Values**: Handle missing values before using PyMin

### Example Data Structure
```python
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],      # Feature 1
    'feature2': [2, 4, 6, 8, 10],     # Feature 2
    'feature3': [0.1, 0.2, 0.3, 0.4, 0.5],  # Feature 3
    'y': [0, 0, 1, 1, 1]              # Target (must be named 'y')
})
```

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

### 5. Probability Predictions
```python
# Get class probabilities (for classification)
probabilities = model.predict_proba(data)
print(probabilities)
```

## Complete Examples

### Classification Example
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

### Regression Example
```python
import pandas as pd
import numpy as np
from PyMin.regression.linear_models import LinearRegressionWrapper

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 100),
    'feature2': np.random.normal(0, 1, 100),
    'feature3': np.random.normal(0, 1, 100),
    'y': 2 * np.random.normal(0, 1, 100) + 3 * np.random.normal(0, 1, 100) + np.random.normal(0, 0.1, 100)
})

# Use the model
model = LinearRegressionWrapper()
model.fit(data)
predictions = model.predict(data)
r2_score = model.score(data, metric='r2')

print(f"R² Score: {r2_score:.3f}")
```

## Advanced Usage

### Custom Parameters
```python
# You can still pass custom parameters if needed
model = create_linear_model('logistic', 
                           with_scaling=True,
                           max_iter=1000,
                           random_state=42)
```

### Model Comparison
```python
# Easy to compare different algorithms
algorithms = ['logistic', 'sgd', 'perceptron']
results = {}

for algo in algorithms:
    model = create_linear_model(algo, with_scaling=True)
    model.fit(data)
    accuracy = model.score(data)
    results[algo] = accuracy

print("Algorithm Comparison:")
for algo, acc in results.items():
    print(f"{algo}: {acc:.3f}")
```

## Why PyMin?

### Before PyMin (Traditional scikit-learn)
```python
# Complex setup required
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Manual data preparation
X = data[['feature1', 'feature2', 'feature3']]
y = data['y']

# Manual scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Manual train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Manual model fitting
model = LogisticRegression()
model.fit(X_train, y_train)

# Manual prediction and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

### With PyMin
```python
# Simple and clean
from PyMin.classification.linear_models import create_linear_model

model = create_linear_model('logistic', with_scaling=True)
model.fit(data)
accuracy = model.score(data)
```

## Summary

PyMin makes machine learning with scikit-learn as simple as:

1. **Choose your algorithm** - Just specify the name
2. **Provide your data** - DataFrame with 'y' column
3. **Get results** - Call fit() and predict()

**No complex setup, no parameter tuning, no data preprocessing needed!**
