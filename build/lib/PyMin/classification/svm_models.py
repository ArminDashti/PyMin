"""
Support Vector Machine (SVM) classification models for PyMin.
Includes SVC, LinearSVC, NuSVC, and other SVM variants.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.svm import SVC, LinearSVC, NuSVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
from .base_classifier import BasePyMinClassifier


class SVCWrapper(BasePyMinClassifier):
    """
    Support Vector Classification wrapper.
    
    SVC implements support vector classification using libsvm. It supports
    both linear and non-linear kernels and can handle multi-class problems.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize SVC model."""
        default_params = {
            'kernel': 'rbf',
            'random_state': 42,
            'probability': True,
            'C': 1.0,
            'gamma': 'scale'
        }
        default_params.update(kwargs)
        self.model = SVC(**default_params)


class LinearSVCWrapper(BasePyMinClassifier):
    """
    Linear Support Vector Classification wrapper.
    
    LinearSVC implements linear support vector classification using liblinear.
    It's faster than SVC for linear kernels and scales better to large datasets.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Linear SVC model."""
        default_params = {
            'random_state': 42,
            'C': 1.0,
            'max_iter': 1000
        }
        default_params.update(kwargs)
        self.model = LinearSVC(**default_params)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        LinearSVC doesn't support predict_proba by default.
        This method is overridden to raise an informative error.
        """
        raise ValueError("LinearSVC does not support predict_proba. Use SVC with linear kernel instead.")


class NuSVCWrapper(BasePyMinClassifier):
    """
    Nu-Support Vector Classification wrapper.
    
    NuSVC implements support vector classification using libsvm with
    a different parameterization. The nu parameter controls the number
    of support vectors and margin errors.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize NuSVC model."""
        default_params = {
            'kernel': 'rbf',
            'random_state': 42,
            'probability': True,
            'nu': 0.5,
            'gamma': 'scale'
        }
        default_params.update(kwargs)
        self.model = NuSVC(**default_params)


class SVMWithScaling(BasePyMinClassifier):
    """
    SVM models with automatic feature scaling.
    
    This wrapper applies StandardScaler to features before training
    and prediction, which is essential for SVM performance.
    """
    
    def __init__(self, model_type: str = 'svc', target_column: str = 'y', **kwargs):
        """
        Initialize SVM model with scaling.
        
        Args:
            model_type: Type of SVM model ('svc', 'linear_svc', 'nu_svc')
            target_column: Name of the target column
            **kwargs: Additional arguments for the underlying model
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        super().__init__(target_column=target_column, **kwargs)
    
    def _initialize_model(self, **kwargs):
        """Initialize the specified SVM model."""
        model_map = {
            'svc': SVC,
            'linear_svc': LinearSVC,
            'nu_svc': NuSVC
        }
        
        if self.model_type not in model_map:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model_class = model_map[self.model_type]
        default_params = {'random_state': 42}
        
        # Add probability support for SVC and NuSVC
        if model_type in ['svc', 'nu_svc']:
            default_params['probability'] = True
        
        default_params.update(kwargs)
        self.model = model_class(**default_params)
    
    def fit(self, df: pd.DataFrame) -> 'SVMWithScaling':
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


class SVMWithKernelSelection(BasePyMinClassifier):
    """
    SVM with automatic kernel selection based on data characteristics.
    
    This wrapper automatically selects the best kernel based on the
    number of features and samples in the dataset.
    """
    
    def __init__(self, target_column: str = 'y', **kwargs):
        """
        Initialize SVM with kernel selection.
        
        Args:
            target_column: Name of the target column
            **kwargs: Additional arguments for the underlying model
        """
        self.scaler = StandardScaler()
        super().__init__(target_column=target_column, **kwargs)
    
    def _select_kernel(self, n_samples: int, n_features: int) -> str:
        """
        Select the best kernel based on data characteristics.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            str: Selected kernel type
        """
        # Rule of thumb for kernel selection
        if n_samples < 1000:
            return 'rbf'  # Good for small datasets
        elif n_features > n_samples:
            return 'linear'  # Good for high-dimensional data
        else:
            return 'rbf'  # Default choice
    
    def _initialize_model(self, **kwargs):
        """Initialize SVC model with automatic kernel selection."""
        default_params = {
            'random_state': 42,
            'probability': True,
            'C': 1.0,
            'gamma': 'scale'
        }
        default_params.update(kwargs)
        self.model = SVC(**default_params)
    
    def fit(self, df: pd.DataFrame) -> 'SVMWithKernelSelection':
        """
        Fit the model with automatic kernel selection and scaling.
        
        Args:
            df: DataFrame containing features and target column 'y'
            
        Returns:
            self: Returns self for method chaining
        """
        X, y = self._prepare_data(df)
        
        # Select kernel based on data characteristics
        kernel = self._select_kernel(len(X), len(X.columns))
        self.model.set_params(kernel=kernel)
        
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
    
    def get_kernel(self) -> str:
        """
        Get the selected kernel type.
        
        Returns:
            str: Selected kernel type
        """
        return self.model.kernel if self.is_fitted else None


# Convenience function to create SVM models
def create_svm_model(model_type: str = 'svc', with_scaling: bool = True,
                    auto_kernel: bool = False, target_column: str = 'y', **kwargs) -> BasePyMinClassifier:
    """
    Create a Support Vector Machine classification model.
    
    Args:
        model_type: Type of SVM model ('svc', 'linear_svc', 'nu_svc')
        with_scaling: Whether to apply feature scaling
        auto_kernel: Whether to use automatic kernel selection (only for SVC)
        target_column: Name of the target column
        **kwargs: Additional arguments for the model
        
    Returns:
        BasePyMinClassifier: Configured SVM model
    """
    if auto_kernel and model_type == 'svc':
        return SVMWithKernelSelection(target_column=target_column, **kwargs)
    elif with_scaling:
        return SVMWithScaling(model_type=model_type, target_column=target_column, **kwargs)
    
    model_map = {
        'svc': SVCWrapper,
        'linear_svc': LinearSVCWrapper,
        'nu_svc': NuSVCWrapper
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_map[model_type](target_column=target_column, **kwargs)
