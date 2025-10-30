"""
Base wrapper class for regression models
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BaseRegressionWrapper(ABC):
    """
    Base class for all regression model wrappers.
    All models expect a DataFrame with target column named 'y'.
    """
    
    def __init__(self, **kwargs):
        """Initialize the regression model with given parameters."""
        self.model = None
        self.feature_columns = None
        self.is_fitted = False
        
    @abstractmethod
    def _create_model(self, **kwargs):
        """Create the underlying scikit-learn model."""
        pass
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'BaseRegressionWrapper':
        """
        Fit the regression model.
        
        Args:
            df: DataFrame with features and target column 'y'
            **kwargs: Additional parameters for the fit method
            
        Returns:
            self for method chaining
        """
        if 'y' not in df.columns:
            raise ValueError("DataFrame must contain a column named 'y' for the target variable")
        
        # Separate features and target
        self.feature_columns = [col for col in df.columns if col != 'y']
        X = df[self.feature_columns]
        y = df['y']
        
        # Create and fit the model
        self.model = self._create_model(**kwargs)
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with feature columns (same as training data)
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.feature_columns is None:
            raise ValueError("Model has not been fitted properly")
        
        # Use only the feature columns that were used during training
        X = df[self.feature_columns]
        return self.model.predict(X)
    
    def score(self, df: pd.DataFrame, metric: str = 'r2') -> float:
        """
        Calculate model performance score.
        
        Args:
            df: DataFrame with features and target column 'y'
            metric: Scoring metric ('r2', 'mse', 'mae')
            
        Returns:
            Score value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        if 'y' not in df.columns:
            raise ValueError("DataFrame must contain a column named 'y' for the target variable")
        
        X = df[self.feature_columns]
        y_true = df['y']
        y_pred = self.predict(df)
        
        if metric == 'r2':
            return r2_score(y_true, y_pred)
        elif metric == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}. Choose from 'r2', 'mse', 'mae'")
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance if available.
        
        Returns:
            Series with feature importance or None if not available
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_, 
                index=self.feature_columns
            ).sort_values(ascending=False)
        elif hasattr(self.model, 'coef_'):
            return pd.Series(
                self.model.coef_, 
                index=self.feature_columns
            ).sort_values(key=abs, ascending=False)
        else:
            return None
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is None:
            return {}
        return self.model.get_params()
    
    def set_params(self, **params) -> 'BaseRegressionWrapper':
        """Set model parameters."""
        if self.model is not None:
            self.model.set_params(**params)
        return self
