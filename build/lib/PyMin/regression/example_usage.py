"""
Example usage of PyMin regression models
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Import all regression models
from .linear_models import (
    LinearRegressionWrapper, 
    RidgeRegressionWrapper, 
    LassoRegressionWrapper, 
    ElasticNetRegressionWrapper
)
from .tree_models import (
    DecisionTreeRegressorWrapper,
    RandomForestRegressorWrapper,
    GradientBoostingRegressorWrapper
)
from .svm_models import SVRWrapper, LinearSVRWrapper
from .ensemble_models import (
    VotingRegressorWrapper,
    BaggingRegressorWrapper,
    AdaBoostRegressorWrapper
)
from .neural_models import MLPRegressorWrapper


def create_sample_data():
    """Create sample regression data."""
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['y'] = y
    return df


def example_linear_regression():
    """Example of using linear regression models."""
    print("=== Linear Regression Example ===")
    
    # Create sample data
    df = create_sample_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr = LinearRegressionWrapper()
    lr.fit(train_df)
    predictions = lr.predict(test_df)
    score = lr.score(test_df)
    print(f"Linear Regression R² Score: {score:.4f}")
    
    # Ridge Regression
    ridge = RidgeRegressionWrapper(alpha=1.0)
    ridge.fit(train_df)
    score = ridge.score(test_df)
    print(f"Ridge Regression R² Score: {score:.4f}")
    
    # Lasso Regression
    lasso = LassoRegressionWrapper(alpha=0.1)
    lasso.fit(train_df)
    score = lasso.score(test_df)
    print(f"Lasso Regression R² Score: {score:.4f}")
    
    # ElasticNet Regression
    elastic = ElasticNetRegressionWrapper(alpha=0.1, l1_ratio=0.5)
    elastic.fit(train_df)
    score = elastic.score(test_df)
    print(f"ElasticNet Regression R² Score: {score:.4f}")


def example_tree_models():
    """Example of using tree-based models."""
    print("\n=== Tree-based Models Example ===")
    
    # Create sample data
    df = create_sample_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Decision Tree
    dt = DecisionTreeRegressorWrapper(max_depth=10, random_state=42)
    dt.fit(train_df)
    score = dt.score(test_df)
    print(f"Decision Tree R² Score: {score:.4f}")
    
    # Random Forest
    rf = RandomForestRegressorWrapper(n_estimators=100, random_state=42)
    rf.fit(train_df)
    score = rf.score(test_df)
    print(f"Random Forest R² Score: {score:.4f}")
    
    # Feature importance
    importance = rf.get_feature_importance()
    print("Top 3 most important features:")
    print(importance.head(3))
    
    # Gradient Boosting
    gb = GradientBoostingRegressorWrapper(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(train_df)
    score = gb.score(test_df)
    print(f"Gradient Boosting R² Score: {score:.4f}")


def example_svm_models():
    """Example of using SVM models."""
    print("\n=== SVM Models Example ===")
    
    # Create sample data
    df = create_sample_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # SVR
    svr = SVRWrapper(kernel='rbf', C=1.0, gamma='scale')
    svr.fit(train_df)
    score = svr.score(test_df)
    print(f"SVR R² Score: {score:.4f}")
    
    # LinearSVR
    linear_svr = LinearSVRWrapper(C=1.0, epsilon=0.1)
    linear_svr.fit(train_df)
    score = linear_svr.score(test_df)
    print(f"LinearSVR R² Score: {score:.4f}")


def example_ensemble_models():
    """Example of using ensemble models."""
    print("\n=== Ensemble Models Example ===")
    
    # Create sample data
    df = create_sample_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Voting Regressor
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    
    estimators = [
        ('lr', LinearRegression()),
        ('dt', DecisionTreeRegressor(max_depth=5, random_state=42))
    ]
    
    voting = VotingRegressorWrapper(estimators=estimators)
    voting.fit(train_df)
    score = voting.score(test_df)
    print(f"Voting Regressor R² Score: {score:.4f}")
    
    # Bagging Regressor
    bagging = BaggingRegressorWrapper(
        base_estimator=DecisionTreeRegressor(max_depth=5),
        n_estimators=10,
        random_state=42
    )
    bagging.fit(train_df)
    score = bagging.score(test_df)
    print(f"Bagging Regressor R² Score: {score:.4f}")
    
    # AdaBoost Regressor
    ada = AdaBoostRegressorWrapper(
        base_estimator=DecisionTreeRegressor(max_depth=3),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    ada.fit(train_df)
    score = ada.score(test_df)
    print(f"AdaBoost Regressor R² Score: {score:.4f}")


def example_neural_network():
    """Example of using neural network models."""
    print("\n=== Neural Network Example ===")
    
    # Create sample data
    df = create_sample_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # MLP Regressor
    mlp = MLPRegressorWrapper(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )
    mlp.fit(train_df)
    score = mlp.score(test_df)
    print(f"MLP Regressor R² Score: {score:.4f}")


def run_all_examples():
    """Run all regression examples."""
    example_linear_regression()
    example_tree_models()
    example_svm_models()
    example_ensemble_models()
    example_neural_network()


if __name__ == "__main__":
    run_all_examples()
