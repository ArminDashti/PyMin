"""
Tree-based classification models for PyMin.
Includes Decision Tree, Random Forest, Gradient Boosting, and other tree models.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier
)
from .base_classifier import BasePyMinClassifier


class DecisionTreeClassifierWrapper(BasePyMinClassifier):
    """
    Decision Tree classifier wrapper.
    
    Decision trees are non-parametric supervised learning methods that
    create a model that predicts the value of a target variable by learning
    simple decision rules inferred from the data features.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Decision Tree Classifier model."""
        default_params = {
            'random_state': 42,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
        default_params.update(kwargs)
        self.model = DecisionTreeClassifier(**default_params)


class RandomForestClassifierWrapper(BasePyMinClassifier):
    """
    Random Forest classifier wrapper.
    
    Random Forest is an ensemble method that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Random Forest Classifier model."""
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.model = RandomForestClassifier(**default_params)


class GradientBoostingClassifierWrapper(BasePyMinClassifier):
    """
    Gradient Boosting classifier wrapper.
    
    Gradient Boosting builds an additive model in a forward stage-wise fashion.
    It generalizes the boosting method by allowing optimization of an arbitrary
    differentiable loss function.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Gradient Boosting Classifier model."""
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
        default_params.update(kwargs)
        self.model = GradientBoostingClassifier(**default_params)


class ExtraTreesClassifierWrapper(BasePyMinClassifier):
    """
    Extra Trees classifier wrapper.
    
    Extra Trees (Extremely Randomized Trees) is an ensemble method that
    builds a large number of unpruned decision trees from the training data.
    It uses random splits at each node rather than the best split.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Extra Trees Classifier model."""
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.model = ExtraTreesClassifier(**default_params)


class AdaBoostClassifierWrapper(BasePyMinClassifier):
    """
    AdaBoost classifier wrapper.
    
    AdaBoost (Adaptive Boosting) is a meta-estimator that begins by fitting
    a classifier on the original dataset and then fits additional copies of
    the classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize AdaBoost Classifier model."""
        default_params = {
            'n_estimators': 50,
            'learning_rate': 1.0,
            'random_state': 42,
            'algorithm': 'SAMME.R'
        }
        default_params.update(kwargs)
        self.model = AdaBoostClassifier(**default_params)


class HistGradientBoostingClassifierWrapper(BasePyMinClassifier):
    """
    Histogram-based Gradient Boosting classifier wrapper.
    
    Histogram-based Gradient Boosting is a much faster variant of gradient
    boosting for large datasets. It bins the input features into discrete
    values and uses these bins to find the best split points.
    """
    
    def _initialize_model(self, **kwargs):
        """Initialize Histogram-based Gradient Boosting Classifier model."""
        default_params = {
            'max_iter': 100,
            'learning_rate': 0.1,
            'random_state': 42,
            'max_depth': None,
            'min_samples_leaf': 20,
            'l2_regularization': 0.0
        }
        default_params.update(kwargs)
        self.model = HistGradientBoostingClassifier(**default_params)


class TreeModelsWithFeatureSelection(BasePyMinClassifier):
    """
    Tree-based models with automatic feature selection.
    
    This wrapper uses feature importance from tree-based models to select
    the most important features for training and prediction.
    """
    
    def __init__(self, model_type: str = 'random_forest', 
                 feature_selection_threshold: float = 0.01,
                 target_column: str = 'y', **kwargs):
        """
        Initialize tree model with feature selection.
        
        Args:
            model_type: Type of tree model ('decision_tree', 'random_forest', 'gradient_boosting', 'extra_trees')
            feature_selection_threshold: Minimum importance threshold for feature selection
            target_column: Name of the target column
            **kwargs: Additional arguments for the underlying model
        """
        self.model_type = model_type
        self.feature_selection_threshold = feature_selection_threshold
        self.selected_features = None
        super().__init__(target_column=target_column, **kwargs)
    
    def _initialize_model(self, **kwargs):
        """Initialize the specified tree model."""
        model_map = {
            'decision_tree': DecisionTreeClassifier,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'extra_trees': ExtraTreesClassifier,
            'adaboost': AdaBoostClassifier,
            'hist_gradient_boosting': HistGradientBoostingClassifier
        }
        
        if self.model_type not in model_map:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model_class = model_map[self.model_type]
        default_params = {'random_state': 42}
        default_params.update(kwargs)
        self.model = model_class(**default_params)
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        """
        Select features based on importance threshold.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            list: Selected feature column names
        """
        # Fit a temporary model to get feature importance
        temp_model = self.model.__class__(**self.model.get_params())
        temp_model.fit(X, y)
        
        if hasattr(temp_model, 'feature_importances_'):
            importances = temp_model.feature_importances_
        else:
            # For models without feature_importances_, use all features
            return X.columns.tolist()
        
        # Select features above threshold
        selected_mask = importances >= self.feature_selection_threshold
        selected_features = X.columns[selected_mask].tolist()
        
        # Ensure at least one feature is selected
        if not selected_features:
            # Select the feature with highest importance
            best_feature_idx = np.argmax(importances)
            selected_features = [X.columns[best_feature_idx]]
        
        return selected_features
    
    def fit(self, df: pd.DataFrame) -> 'TreeModelsWithFeatureSelection':
        """
        Fit the model with feature selection.
        
        Args:
            df: DataFrame containing features and target column 'y'
            
        Returns:
            self: Returns self for method chaining
        """
        X, y = self._prepare_data(df)
        
        # Select important features
        self.selected_features = self._select_features(X, y)
        
        # Fit the model with selected features
        X_selected = X[self.selected_features]
        self.model.fit(X_selected, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels with selected features.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_selected = df[self.selected_features]
        return self.model.predict(X_selected)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities with selected features.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            np.ndarray: Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.model_type} does not support predict_proba")
        
        X_selected = df[self.selected_features]
        return self.model.predict_proba(X_selected)
    
    def get_selected_features(self) -> list:
        """
        Get the list of selected features.
        
        Returns:
            list: Selected feature column names
        """
        return self.selected_features if self.selected_features else []


# Convenience function to create tree models
def create_tree_model(model_type: str = 'random_forest', with_feature_selection: bool = False,
                     target_column: str = 'y', **kwargs) -> BasePyMinClassifier:
    """
    Create a tree-based classification model.
    
    Args:
        model_type: Type of tree model ('decision_tree', 'random_forest', 'gradient_boosting', 
                   'extra_trees', 'adaboost', 'hist_gradient_boosting')
        with_feature_selection: Whether to apply automatic feature selection
        target_column: Name of the target column
        **kwargs: Additional arguments for the model
        
    Returns:
        BasePyMinClassifier: Configured tree model
    """
    if with_feature_selection:
        return TreeModelsWithFeatureSelection(
            model_type=model_type, 
            target_column=target_column, 
            **kwargs
        )
    
    model_map = {
        'decision_tree': DecisionTreeClassifierWrapper,
        'random_forest': RandomForestClassifierWrapper,
        'gradient_boosting': GradientBoostingClassifierWrapper,
        'extra_trees': ExtraTreesClassifierWrapper,
        'adaboost': AdaBoostClassifierWrapper,
        'hist_gradient_boosting': HistGradientBoostingClassifierWrapper
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_map[model_type](target_column=target_column, **kwargs)
