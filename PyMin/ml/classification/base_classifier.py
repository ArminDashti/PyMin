import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional, Any, Dict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings


class BasePyMinClassifier(ABC, BaseEstimator, ClassifierMixin):
    
    def __init__(self, target_column: str = 'y', **kwargs):
        self.target_column = target_column
        self.feature_columns = None
        self.is_fitted = False
        self.model = None
        self._initialize_model(**kwargs)
    
    @abstractmethod
    def _initialize_model(self, **kwargs):
        pass
    
    def _prepare_data(self, df: pd.DataFrame) -> tuple:
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")
        
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns if col != self.target_column]
        
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        return X, y
    
    def fit(self, df: pd.DataFrame) -> 'BasePyMinClassifier':
        X, y = self._prepare_data(df)
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.feature_columns is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = df[self.feature_columns]
        return self.model.predict(X)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.__class__.__name__} does not support predict_proba")
        
        X = df[self.feature_columns]
        return self.model.predict_proba(X)
    
    def score(self, df: pd.DataFrame) -> float:
        X, y = self._prepare_data(df)
        return self.model.score(X, y)
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_columns
            ).sort_values(ascending=False)
        elif hasattr(self.model, 'coef_'):
            if len(self.model.coef_.shape) == 1:
                return pd.Series(
                    np.abs(self.model.coef_),
                    index=self.feature_columns
                ).sort_values(ascending=False)
            else:
                return pd.Series(
                    np.abs(self.model.coef_).mean(axis=0),
                    index=self.feature_columns
                ).sort_values(ascending=False)
        else:
            return None
    
    def evaluate(self, df: pd.DataFrame, detailed: bool = True) -> Dict[str, Any]:
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
        X, y = self._prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        return train_df, test_df
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_column='{self.target_column}')"
