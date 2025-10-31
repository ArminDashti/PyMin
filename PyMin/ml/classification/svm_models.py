import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.svm import SVC, LinearSVC, NuSVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
from .base_classifier import BasePyMinClassifier


class SVCWrapper(BasePyMinClassifier):
    
    def _initialize_model(self, **kwargs):
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
    
    def _initialize_model(self, **kwargs):
        default_params = {
            'random_state': 42,
            'C': 1.0,
            'max_iter': 1000
        }
        default_params.update(kwargs)
        self.model = LinearSVC(**default_params)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise ValueError("LinearSVC does not support predict_proba. Use SVC with linear kernel instead.")


class NuSVCWrapper(BasePyMinClassifier):
    
    def _initialize_model(self, **kwargs):
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
    
    def __init__(self, model_type: str = 'svc', target_column: str = 'y', **kwargs):
        self.model_type = model_type
        self.scaler = StandardScaler()
        super().__init__(target_column=target_column, **kwargs)
    
    def _initialize_model(self, **kwargs):
        model_map = {
            'svc': SVC,
            'linear_svc': LinearSVC,
            'nu_svc': NuSVC
        }
        
        if self.model_type not in model_map:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model_class = model_map[self.model_type]
        default_params = {'random_state': 42}
        
        if model_type in ['svc', 'nu_svc']:
            default_params['probability'] = True
        
        default_params.update(kwargs)
        self.model = model_class(**default_params)
    
    def fit(self, df: pd.DataFrame) -> 'SVMWithScaling':
        X, y = self._prepare_data(df)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.model_type} does not support predict_proba")
        
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def score(self, df: pd.DataFrame) -> float:
        X, y = self._prepare_data(df)
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)


class SVMWithKernelSelection(BasePyMinClassifier):
    
    def __init__(self, target_column: str = 'y', **kwargs):
        self.scaler = StandardScaler()
        super().__init__(target_column=target_column, **kwargs)
    
    def _select_kernel(self, n_samples: int, n_features: int) -> str:
        if n_samples < 1000:
            return 'rbf'
        elif n_features > n_samples:
            return 'linear'
        else:
            return 'rbf'
    
    def _initialize_model(self, **kwargs):
        default_params = {
            'random_state': 42,
            'probability': True,
            'C': 1.0,
            'gamma': 'scale'
        }
        default_params.update(kwargs)
        self.model = SVC(**default_params)
    
    def fit(self, df: pd.DataFrame) -> 'SVMWithKernelSelection':
        X, y = self._prepare_data(df)
        
        kernel = self._select_kernel(len(X), len(X.columns))
        self.model.set_params(kernel=kernel)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def score(self, df: pd.DataFrame) -> float:
        X, y = self._prepare_data(df)
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)
    
    def get_kernel(self) -> str:
        return self.model.kernel if self.is_fitted else None


def create_svm_model(model_type: str = 'svc', with_scaling: bool = True,
                    auto_kernel: bool = False, target_column: str = 'y', **kwargs) -> BasePyMinClassifier:
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
