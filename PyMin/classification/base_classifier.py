"""
Base classifier class for PyMin classification module.
All classifiers inherit from this base class to ensure consistent DataFrame handling.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional, Any, Dict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings


class BasePyMinClassifier(ABC, BaseEstimator, ClassifierMixin):
    """
    Base class for all PyMin classifiers.
    
    This class provides a consistent interface for all classification algorithms
    in PyMin, ensuring they all work with DataFrames where the target column
    is named 'y'.
    
    Attributes:
        model: The underlying scikit-learn model
        feature_columns: List of feature column names
        target_column: Name of the target column (default: 'y')
        is_fitted: Boolean indicating if the model has been fitted
    """
    
    def __init__(self, target_column: str = 'y', **kwargs):
        """
        Initialize the base classifier.
        
        Args:
            target_column: Name of the target column in the DataFrame
            **kwargs: Additional arguments passed to the underlying model
        """
        self.target_column = target_column
        self.feature_columns = None
        self.is_fitted = False
        self.model = None
        self._initialize_model(**kwargs)
    
    @abstractmethod
    def _initialize_model(self, **kwargs):
        """Initialize the underlying scikit-learn model."""
        pass
    
    def _prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare data for training/prediction.
        
        Args:
            df: DataFrame containing features and target
            
        Returns:
            tuple: (X, y) where X is feature matrix and y is target vector
            
        Raises:
            ValueError: If target column 'y' is not found in DataFrame
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")
        
        # Store feature columns if not already stored
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns if col != self.target_column]
        
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        return X, y
    
    def fit(self, df: pd.DataFrame) -> 'BasePyMinClassifier':
        """
        Fit the classifier to the training data.
        
        Args:
            df: DataFrame containing features and target column 'y'
            
        Returns:
            self: Returns self for method chaining
        """
        X, y = self._prepare_data(df)
        
        # Fit the underlying model
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the given data.
        
        Args:
            df: DataFrame containing features (target column not required)
            
        Returns:
            np.ndarray: Predicted class labels
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Use stored feature columns for prediction
        if self.feature_columns is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = df[self.feature_columns]
        return self.model.predict(X)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the given data.
        
        Args:
            df: DataFrame containing features (target column not required)
            
        Returns:
            np.ndarray: Predicted class probabilities
            
        Raises:
            ValueError: If model is not fitted or doesn't support predict_proba
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.__class__.__name__} does not support predict_proba")
        
        X = df[self.feature_columns]
        return self.model.predict_proba(X)
    
    def score(self, df: pd.DataFrame) -> float:
        """
        Calculate accuracy score on the given data.
        
        Args:
            df: DataFrame containing features and target column 'y'
            
        Returns:
            float: Accuracy score
        """
        X, y = self._prepare_data(df)
        return self.model.score(X, y)
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance if available.
        
        Returns:
            pd.Series or None: Feature importance scores
        """
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_columns
            ).sort_values(ascending=False)
        elif hasattr(self.model, 'coef_'):
            # For linear models, return absolute coefficients
            if len(self.model.coef_.shape) == 1:
                return pd.Series(
                    np.abs(self.model.coef_),
                    index=self.feature_columns
                ).sort_values(ascending=False)
            else:
                # Multi-class case
                return pd.Series(
                    np.abs(self.model.coef_).mean(axis=0),
                    index=self.feature_columns
                ).sort_values(ascending=False)
        else:
            return None
    
    def evaluate(self, df: pd.DataFrame, detailed: bool = True) -> Dict[str, Any]:
        """
        Evaluate the model on the given data.
        
        Args:
            df: DataFrame containing features and target column 'y'
            detailed: If True, return detailed metrics
            
        Returns:
            dict: Evaluation metrics
        """
        X, y = self._prepare_data(df)
        y_pred = self.predict(df)
        
        results = {
            'accuracy': accuracy_score(y, y_pred),
            'predictions': y_pred
        }
        
        if detailed:
            results.update({
                'classification_report': classification_report(y, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist()
            })
        
        return results
    
    def train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, 
                        random_state: Optional[int] = None) -> tuple:
        """
        Split the data into training and testing sets.
        
        Args:
            df: DataFrame containing features and target column 'y'
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            tuple: (train_df, test_df)
        """
        X, y = self._prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        return train_df, test_df
    
    def __repr__(self) -> str:
        """String representation of the classifier."""
        return f"{self.__class__.__name__}(target_column='{self.target_column}')"
