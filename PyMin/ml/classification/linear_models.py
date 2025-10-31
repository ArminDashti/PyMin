"""
Linear classification models for PyMin.
Includes Logistic Regression, SGD Classifier, and other linear models.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.linear_model import (
    LogisticRegression, 
    SGDClassifier, 
    Perceptron, 
    PassiveAggressiveClassifier,
    RidgeClassifier,
    RidgeClassifierCV
)
from sklearn.preprocessing import StandardScaler
from .base_classifier import BasePyMinClassifier


class LogisticRegressionClassifier(BasePyMinClassifier):
    """
    Logistic Regression classifier wrapper.
    
    Logistic regression is a linear model for classification that uses
    the logistic function to model the probability of class membership.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Logistic Regression model."""
        default_params = {
            'random_state': 42,
            'max_iter': 1000,
            'solver': 'lbfgs'
        }
        default_params.update(kwargs)
        self.model = LogisticRegression(**default_params)


class SGDClassifierWrapper(BasePyMinClassifier):
    """
    Stochastic Gradient Descent classifier wrapper.
    
    SGD is an efficient linear classifier that can handle large datasets
    and supports various loss functions and regularization methods.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize SGD Classifier model."""
        default_params = {
            'random_state': 42,
            'max_iter': 1000,
            'loss': 'hinge'
        }
        default_params.update(kwargs)
        self.model = SGDClassifier(**default_params)


class PerceptronClassifier(BasePyMinClassifier):
    """
    Perceptron classifier wrapper.
    
    The Perceptron is a simple linear classifier that learns by
    updating weights based on misclassified examples.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Perceptron model."""
        default_params = {
            'random_state': 42,
            'max_iter': 1000
        }
        default_params.update(kwargs)
        self.model = Perceptron(**default_params)


class PassiveAggressiveClassifierWrapper(BasePyMinClassifier):
    """
    Passive Aggressive classifier wrapper.
    
    Passive Aggressive algorithms are online learning algorithms that
    are well-suited for large-scale learning tasks.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Passive Aggressive Classifier model."""
        default_params = {
            'random_state': 42,
            'max_iter': 1000
        }
        default_params.update(kwargs)
        self.model = PassiveAggressiveClassifier(**default_params)


class RidgeClassifierWrapper(BasePyMinClassifier):
    """
    Ridge classifier wrapper.
    
    Ridge classifier uses Ridge regression with binary classification
    by converting the target to {-1, 1} and treating the problem as
    a regression task.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Ridge Classifier model."""
        default_params = {
            'random_state': 42,
            'alpha': 1.0
        }
        default_params.update(kwargs)
        self.model = RidgeClassifier(**default_params)


class RidgeClassifierCVWrapper(BasePyMinClassifier):
    """
    Ridge classifier with cross-validation wrapper.
    
    Ridge classifier that automatically selects the best alpha parameter
    using cross-validation.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Ridge Classifier CV model."""
        default_params = {
            'cv': 5,
            'random_state': 42
        }
        default_params.update(kwargs)
        self.model = RidgeClassifierCV(**default_params)


class LinearModelsWithScaling(BasePyMinClassifier):
    """
    Linear models with automatic feature scaling.
    
    This wrapper applies StandardScaler to features before training
    and prediction, which is often beneficial for linear models.
    """
    
    def __init__(self, model_type: str = 'logistic', target_column: str = 'y', **kwargs):
        """
        Initialize linear model with scaling.
        
        Args:
            model_type: Type of linear model ('logistic', 'sgd', 'perceptron', 'ridge')
            target_column: Name of the target column
            **kwargs: Additional arguments for the underlying model
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        super().__init__(target_column=target_column, **kwargs)
    
    def _initialize_model(self, **kwargs):
        """Initialize the specified linear model."""
        model_map = {
            'logistic': LogisticRegression,
            'sgd': SGDClassifier,
            'perceptron': Perceptron,
            'ridge': RidgeClassifier
        }
        
        if self.model_type not in model_map:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model_class = model_map[self.model_type]
        default_params = {'random_state': 42}
        default_params.update(kwargs)
        self.model = model_class(**default_params)
    
    def fit(self, df: pd.DataFrame) -> 'LinearModelsWithScaling':
        """
        Fit the model with scaled features.
        
        Args:
            df: DataFrame containing features and target column 'y'
            
        Returns:
            self: Returns self for method chaining
        """
        X, y = self._prepare_data(df)
        
        # Fit scaler and transform features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the underlying model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels with scaled features.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities with scaled features.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            np.ndarray: Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.model_type} does not support predict_proba")
        
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def score(self, df: pd.DataFrame) -> float:
        """
        Calculate accuracy score with scaled features.
        
        Args:
            df: DataFrame containing features and target column 'y'
            
        Returns:
            float: Accuracy score
        """
        X, y = self._prepare_data(df)
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)


# Convenience function to create linear models
def create_linear_model(model_type: str = 'logistic', with_scaling: bool = True, 
                       target_column: str = 'y', **kwargs) -> BasePyMinClassifier:
    """
    Create a linear classification model.
    
    Args:
        model_type: Type of linear model ('logistic', 'sgd', 'perceptron', 'ridge', 'ridge_cv')
        with_scaling: Whether to apply feature scaling
        target_column: Name of the target column
        **kwargs: Additional arguments for the model
        
    Returns:
        BasePyMinClassifier: Configured linear model
    """
    if with_scaling and model_type != 'ridge_cv':
        return LinearModelsWithScaling(model_type=model_type, target_column=target_column, **kwargs)
    
    model_map = {
        'logistic': LogisticRegressionClassifier,
        'sgd': SGDClassifierWrapper,
        'perceptron': PerceptronClassifier,
        'ridge': RidgeClassifierWrapper,
        'ridge_cv': RidgeClassifierCVWrapper
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_map[model_type](target_column=target_column, **kwargs)
