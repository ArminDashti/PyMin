"""
Naive Bayes classification models for PyMin.
Includes Gaussian, Multinomial, Bernoulli, and Complement Naive Bayes.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    BernoulliNB,
    ComplementNB,
    CategoricalNB
)
from sklearn.preprocessing import LabelEncoder
from .base_classifier import BasePyMinClassifier


class GaussianNBWrapper(BasePyMinClassifier):
    """
    Gaussian Naive Bayes classifier wrapper.
    
    Gaussian Naive Bayes implements the Gaussian Naive Bayes algorithm
    for classification. It assumes that the likelihood of the features
    is Gaussian.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Gaussian Naive Bayes model."""
        default_params = {
            'var_smoothing': 1e-9
        }
        default_params.update(kwargs)
        self.model = GaussianNB(**default_params)


class MultinomialNBWrapper(BasePyMinClassifier):
    """
    Multinomial Naive Bayes classifier wrapper.
    
    Multinomial Naive Bayes implements the naive Bayes algorithm for
    multinomially distributed data. It's suitable for discrete features
    and text classification.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Multinomial Naive Bayes model."""
        default_params = {
            'alpha': 1.0,
            'fit_prior': True,
            'class_prior': None
        }
        default_params.update(kwargs)
        self.model = MultinomialNB(**default_params)


class BernoulliNBWrapper(BasePyMinClassifier):
    """
    Bernoulli Naive Bayes classifier wrapper.
    
    Bernoulli Naive Bayes implements the naive Bayes training and
    classification algorithms for data that is distributed according
    to multivariate Bernoulli distributions.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Bernoulli Naive Bayes model."""
        default_params = {
            'alpha': 1.0,
            'binarize': 0.0,
            'fit_prior': True,
            'class_prior': None
        }
        default_params.update(kwargs)
        self.model = BernoulliNB(**default_params)


class ComplementNBWrapper(BasePyMinClassifier):
    """
    Complement Naive Bayes classifier wrapper.
    
    Complement Naive Bayes implements the complement naive Bayes
    algorithm. It's particularly suited for imbalanced datasets.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Complement Naive Bayes model."""
        default_params = {
            'alpha': 1.0,
            'fit_prior': True,
            'class_prior': None,
            'norm': False
        }
        default_params.update(kwargs)
        self.model = ComplementNB(**default_params)


class CategoricalNBWrapper(BasePyMinClassifier):
    """
    Categorical Naive Bayes classifier wrapper.
    
    Categorical Naive Bayes implements the categorical naive Bayes
    algorithm for categorically distributed data. It's suitable for
    discrete features with categorical distributions.
    """
    
    def __init__(self, target_column: str = 'y', **kwargs):
        """
        Initialize Categorical Naive Bayes classifier.
        
        Args:
            target_column: Name of the target column
            **kwargs: Additional arguments for the underlying model
        """
        self.label_encoders = {}
        super().__init__(target_column=target_column, **kwargs)
    
    def _initialize_model(self, **kwargs):
        """Initialize Categorical Naive Bayes model."""
        default_params = {
            'alpha': 1.0,
            'fit_prior': True,
            'class_prior': None,
            'min_categories': None
        }
        default_params.update(kwargs)
        self.model = CategoricalNB(**default_params)
    
    def _encode_categorical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features for CategoricalNB.
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the encoders
            
        Returns:
            pd.DataFrame: Encoded features
        """
        X_encoded = X.copy()
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                if fit:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        le = self.label_encoders[col]
                        X_encoded[col] = X[col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        X_encoded[col] = 0  # Default value for unseen categories
        
        return X_encoded
    
    def fit(self, df: pd.DataFrame) -> 'CategoricalNBWrapper':
        """
        Fit the Categorical Naive Bayes model.
        
        Args:
            df: DataFrame containing features and target column 'y'
            
        Returns:
            self: Returns self for method chaining
        """
        X, y = self._prepare_data(df)
        
        # Encode categorical features
        X_encoded = self._encode_categorical_features(X, fit=True)
        
        # Fit the underlying model
        self.model.fit(X_encoded, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = df[self.feature_columns]
        X_encoded = self._encode_categorical_features(X, fit=False)
        return self.model.predict(X_encoded)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            np.ndarray: Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = df[self.feature_columns]
        X_encoded = self._encode_categorical_features(X, fit=False)
        return self.model.predict_proba(X_encoded)
    
    def score(self, df: pd.DataFrame) -> float:
        """
        Calculate accuracy score.
        
        Args:
            df: DataFrame containing features and target column 'y'
            
        Returns:
            float: Accuracy score
        """
        X, y = self._prepare_data(df)
        X_encoded = self._encode_categorical_features(X, fit=False)
        return self.model.score(X_encoded, y)


class NaiveBayesWithAutoSelection(BasePyMinClassifier):
    """
    Naive Bayes with automatic algorithm selection based on data characteristics.
    
    This wrapper automatically selects the best Naive Bayes variant based on
    the data types and characteristics of the features.
    """
    
    def __init__(self, target_column: str = 'y', **kwargs):
        """
        Initialize Naive Bayes with auto selection.
        
        Args:
            target_column: Name of the target column
            **kwargs: Additional arguments for the underlying model
        """
        self.selected_algorithm = None
        self.label_encoders = {}
        super().__init__(target_column=target_column, **kwargs)
    
    def _select_algorithm(self, X: pd.DataFrame) -> str:
        """
        Select the best Naive Bayes algorithm based on data characteristics.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            str: Selected algorithm name
        """
        # Check data types
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Check if all values are non-negative integers (suitable for MultinomialNB)
        all_non_negative = True
        for col in numeric_cols:
            if not np.all((X[col] >= 0) & (X[col] == X[col].astype(int))):
                all_non_negative = False
                break
        
        # Check if all values are binary (suitable for BernoulliNB)
        all_binary = True
        for col in numeric_cols:
            unique_vals = X[col].unique()
            if not (len(unique_vals) == 2 and set(unique_vals).issubset({0, 1})):
                all_binary = False
                break
        
        # Algorithm selection logic
        if len(categorical_cols) > 0:
            return 'categorical'
        elif all_binary:
            return 'bernoulli'
        elif all_non_negative:
            return 'multinomial'
        else:
            return 'gaussian'
    
    def _initialize_model(self, **kwargs):
        """Initialize the selected Naive Bayes model."""
        # This will be set in fit method based on data characteristics
        self.model = None
    
    def _encode_categorical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features for CategoricalNB.
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the encoders
            
        Returns:
            pd.DataFrame: Encoded features
        """
        X_encoded = X.copy()
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                if fit:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        X_encoded[col] = X[col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        X_encoded[col] = 0
        
        return X_encoded
    
    def fit(self, df: pd.DataFrame) -> 'NaiveBayesWithAutoSelection':
        """
        Fit the model with automatic algorithm selection.
        
        Args:
            df: DataFrame containing features and target column 'y'
            
        Returns:
            self: Returns self for method chaining
        """
        X, y = self._prepare_data(df)
        
        # Select algorithm based on data characteristics
        self.selected_algorithm = self._select_algorithm(X)
        
        # Initialize the selected model
        algorithm_map = {
            'gaussian': GaussianNB,
            'multinomial': MultinomialNB,
            'bernoulli': BernoulliNB,
            'complement': ComplementNB,
            'categorical': CategoricalNB
        }
        
        model_class = algorithm_map[self.selected_algorithm]
        self.model = model_class()
        
        # Handle categorical features if needed
        if self.selected_algorithm == 'categorical':
            X = self._encode_categorical_features(X, fit=True)
        
        # Fit the underlying model
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = df[self.feature_columns]
        
        if self.selected_algorithm == 'categorical':
            X = self._encode_categorical_features(X, fit=False)
        
        return self.model.predict(X)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            np.ndarray: Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = df[self.feature_columns]
        
        if self.selected_algorithm == 'categorical':
            X = self._encode_categorical_features(X, fit=False)
        
        return self.model.predict_proba(X)
    
    def score(self, df: pd.DataFrame) -> float:
        """
        Calculate accuracy score.
        
        Args:
            df: DataFrame containing features and target column 'y'
            
        Returns:
            float: Accuracy score
        """
        X, y = self._prepare_data(df)
        
        if self.selected_algorithm == 'categorical':
            X = self._encode_categorical_features(X, fit=False)
        
        return self.model.score(X, y)
    
    def get_selected_algorithm(self) -> str:
        """
        Get the selected algorithm name.
        
        Returns:
            str: Selected algorithm name
        """
        return self.selected_algorithm


# Convenience function to create Naive Bayes models
def create_naive_bayes_model(model_type: str = 'auto', target_column: str = 'y', **kwargs) -> BasePyMinClassifier:
    """
    Create a Naive Bayes classification model.
    
    Args:
        model_type: Type of Naive Bayes model ('gaussian', 'multinomial', 'bernoulli', 
                   'complement', 'categorical', 'auto')
        target_column: Name of the target column
        **kwargs: Additional arguments for the model
        
    Returns:
        BasePyMinClassifier: Configured Naive Bayes model
    """
    if model_type == 'auto':
        return NaiveBayesWithAutoSelection(target_column=target_column, **kwargs)
    
    model_map = {
        'gaussian': GaussianNBWrapper,
        'multinomial': MultinomialNBWrapper,
        'bernoulli': BernoulliNBWrapper,
        'complement': ComplementNBWrapper,
        'categorical': CategoricalNBWrapper
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_map[model_type](target_column=target_column, **kwargs)
