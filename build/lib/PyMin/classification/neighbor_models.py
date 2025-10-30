"""
Neighbor-based classification models for PyMin.
Includes k-NN, Radius Neighbors, and other neighbor-based classifiers.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
from sklearn.neighbors import (
    KNeighborsClassifier,
    RadiusNeighborsClassifier,
    NearestCentroid
)
from sklearn.preprocessing import StandardScaler
from .base_classifier import BasePyMinClassifier


class KNeighborsClassifierWrapper(BasePyMinClassifier):
    """
    k-Nearest Neighbors classifier wrapper.
    
    k-NN is a non-parametric method that classifies data points based on
    the class of their k nearest neighbors in the feature space.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize k-NN Classifier model."""
        default_params = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'leaf_size': 30,
            'p': 2,
            'metric': 'minkowski',
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.model = KNeighborsClassifier(**default_params)


class RadiusNeighborsClassifierWrapper(BasePyMinClassifier):
    """
    Radius Neighbors classifier wrapper.
    
    Radius Neighbors classifier implements learning based on the number
    of neighbors within a fixed radius r of each training point.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Radius Neighbors Classifier model."""
        default_params = {
            'radius': 1.0,
            'weights': 'uniform',
            'algorithm': 'auto',
            'leaf_size': 30,
            'p': 2,
            'metric': 'minkowski',
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.model = RadiusNeighborsClassifier(**default_params)


class NearestCentroidWrapper(BasePyMinClassifier):
    """
    Nearest Centroid classifier wrapper.
    
    Nearest Centroid classifier represents each class by the centroid of
    its members. It's a simple and fast classifier that works well for
    certain types of data.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Nearest Centroid Classifier model."""
        default_params = {
            'metric': 'euclidean',
            'shrink_threshold': None
        }
        default_params.update(kwargs)
        self.model = NearestCentroid(**default_params)


class NeighborModelsWithScaling(BasePyMinClassifier):
    """
    Neighbor-based models with automatic feature scaling.
    
    This wrapper applies StandardScaler to features before training
    and prediction, which is essential for distance-based algorithms.
    """
    
    def __init__(self, model_type: str = 'knn', target_column: str = 'y', **kwargs):
        """
        Initialize neighbor model with scaling.
        
        Args:
            model_type: Type of neighbor model ('knn', 'radius', 'centroid')
            target_column: Name of the target column
            **kwargs: Additional arguments for the underlying model
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        super().__init__(target_column=target_column, **kwargs)
    
    def _initialize_model(self, **kwargs):
        """Initialize the specified neighbor model."""
        model_map = {
            'knn': KNeighborsClassifier,
            'radius': RadiusNeighborsClassifier,
            'centroid': NearestCentroid
        }
        
        if self.model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = model_map[self.model_type]
        default_params = {'n_jobs': -1} if model_type in ['knn', 'radius'] else {}
        default_params.update(kwargs)
        self.model = model_class(**default_params)
    
    def fit(self, df: pd.DataFrame) -> 'NeighborModelsWithScaling':
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


class KNNWithAutoK(BasePyMinClassifier):
    """
    k-NN with automatic k selection using cross-validation.
    
    This wrapper automatically selects the best k value based on
    cross-validation performance.
    """
    
    def __init__(self, k_range: tuple = (1, 20), cv_folds: int = 5,
                 target_column: str = 'y', **kwargs):
        """
        Initialize k-NN with auto k selection.
        
        Args:
            k_range: Tuple of (min_k, max_k) for k selection
            cv_folds: Number of cross-validation folds
            target_column: Name of the target column
            **kwargs: Additional arguments for the underlying model
        """
        self.k_range = k_range
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.best_k = None
        super().__init__(target_column=target_column, **kwargs)
    
    def _initialize_model(self, **kwargs):
        """Initialize k-NN model."""
        default_params = {
            'weights': 'uniform',
            'algorithm': 'auto',
            'leaf_size': 30,
            'p': 2,
            'metric': 'minkowski',
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.model = KNeighborsClassifier(**default_params)
    
    def _select_best_k(self, X: pd.DataFrame, y: pd.Series) -> int:
        """
        Select the best k value using cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            int: Best k value
        """
        from sklearn.model_selection import cross_val_score
        
        best_score = -1
        best_k = self.k_range[0]
        
        for k in range(self.k_range[0], self.k_range[1] + 1):
            self.model.set_params(n_neighbors=k)
            scores = cross_val_score(self.model, X, y, cv=self.cv_folds, scoring='accuracy')
            mean_score = scores.mean()
            
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
        
        return best_k
    
    def fit(self, df: pd.DataFrame) -> 'KNNWithAutoK':
        """
        Fit the model with automatic k selection.
        
        Args:
            df: DataFrame containing features and target column 'y'
            
        Returns:
            self: Returns self for method chaining
        """
        X, y = self._prepare_data(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Select best k
        self.best_k = self._select_best_k(X_scaled, y)
        self.model.set_params(n_neighbors=self.best_k)
        
        # Fit the model with best k
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
    
    def get_best_k(self) -> int:
        """
        Get the selected best k value.
        
        Returns:
            int: Best k value
        """
        return self.best_k


class NeighborModelsWithDistanceWeights(BasePyMinClassifier):
    """
    Neighbor-based models with distance-based weighting.
    
    This wrapper uses distance-based weights for neighbor classification,
    giving more influence to closer neighbors.
    """
    
    def __init__(self, model_type: str = 'knn', target_column: str = 'y', **kwargs):
        """
        Initialize neighbor model with distance weights.
        
        Args:
            model_type: Type of neighbor model ('knn', 'radius')
            target_column: Name of the target column
            **kwargs: Additional arguments for the underlying model
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        super().__init__(target_column=target_column, **kwargs)
    
    def _initialize_model(self, **kwargs):
        """Initialize the specified neighbor model with distance weights."""
        model_map = {
            'knn': KNeighborsClassifier,
            'radius': RadiusNeighborsClassifier
        }
        
        if self.model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = model_map[self.model_type]
        default_params = {
            'weights': 'distance',
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.model = model_class(**default_params)
    
    def fit(self, df: pd.DataFrame) -> 'NeighborModelsWithDistanceWeights':
        """
        Fit the model with distance weights and scaling.
        
        Args:
            df: DataFrame containing features and target column 'y'
            
        Returns:
            self: Returns self for method chaining
        """
        X, y = self._prepare_data(df)
        
        # Scale features
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


# Convenience function to create neighbor models
def create_neighbor_model(model_type: str = 'knn', with_scaling: bool = True,
                         auto_k: bool = False, distance_weights: bool = False,
                         target_column: str = 'y', **kwargs) -> BasePyMinClassifier:
    """
    Create a neighbor-based classification model.
    
    Args:
        model_type: Type of neighbor model ('knn', 'radius', 'centroid')
        with_scaling: Whether to apply feature scaling
        auto_k: Whether to use automatic k selection (only for k-NN)
        distance_weights: Whether to use distance-based weights
        target_column: Name of the target column
        **kwargs: Additional arguments for the model
        
    Returns:
        BasePyMinClassifier: Configured neighbor model
    """
    if auto_k and model_type == 'knn':
        return KNNWithAutoK(target_column=target_column, **kwargs)
    elif distance_weights and model_type in ['knn', 'radius']:
        return NeighborModelsWithDistanceWeights(model_type=model_type, target_column=target_column, **kwargs)
    elif with_scaling:
        return NeighborModelsWithScaling(model_type=model_type, target_column=target_column, **kwargs)
    
    model_map = {
        'knn': KNeighborsClassifierWrapper,
        'radius': RadiusNeighborsClassifierWrapper,
        'centroid': NearestCentroidWrapper
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_map[model_type](target_column=target_column, **kwargs)
