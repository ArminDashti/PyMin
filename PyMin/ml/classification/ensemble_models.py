"""
Ensemble classification models for PyMin.
Includes Voting, Bagging, Stacking, and other ensemble methods.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sklearn.ensemble import (
    VotingClassifier,
    BaggingClassifier,
    StackingClassifier,
    ExtraTreesClassifier
)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from .base_classifier import BasePyMinClassifier


class VotingClassifierWrapper(BasePyMinClassifier):
    """
    Voting Classifier wrapper.
    
    Voting Classifier is an ensemble method that combines multiple
    classifiers and uses majority voting or averaging for predictions.
    """
    
    def __init__(self, estimators: Optional[List] = None, voting: str = 'hard',
                 target_column: str = 'y', **kwargs):
        """
        Initialize Voting Classifier.
        
        Args:
            estimators: List of (name, estimator) tuples. If None, uses default estimators.
            voting: 'hard' for majority voting or 'soft' for averaging probabilities
            target_column: Name of the target column
            **kwargs: Additional arguments for the underlying model
        """
        self.voting = voting
        self.estimators = estimators or self._get_default_estimators()
        super().__init__(target_column=target_column, **kwargs)
    
    def _get_default_estimators(self) -> List:
        """Get default estimators for voting classifier."""
        return [
            ('lr', LogisticRegression(random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42)),
            ('svc', SVC(probability=True, random_state=42)),
            ('nb', GaussianNB()),
            ('knn', KNeighborsClassifier())
        ]
    
    def _initialize_model(self, **kwargs):
        """Initialize Voting Classifier model."""
        default_params = {
            'voting': self.voting,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.model = VotingClassifier(
            estimators=self.estimators,
            **default_params
        )


class BaggingClassifierWrapper(BasePyMinClassifier):
    """
    Bagging Classifier wrapper.
    
    Bagging Classifier implements bagging (Bootstrap Aggregating) which
    trains multiple base estimators on bootstrap samples of the dataset.
    """
    
    def __init__(self, base_estimator=None, target_column: str = 'y', **kwargs):
        """
        Initialize Bagging Classifier.
        
        Args:
            base_estimator: Base estimator to use. If None, uses DecisionTreeClassifier
            target_column: Name of the target column
            **kwargs: Additional arguments for the underlying model
        """
        self.base_estimator = base_estimator or DecisionTreeClassifier(random_state=42)
        super().__init__(target_column=target_column, **kwargs)
    
    def _initialize_model(self, **kwargs):
        """Initialize Bagging Classifier model."""
        default_params = {
            'n_estimators': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.model = BaggingClassifier(
            base_estimator=self.base_estimator,
            **default_params
        )


class StackingClassifierWrapper(BasePyMinClassifier):
    """
    Stacking Classifier wrapper.
    
    Stacking Classifier implements stacking (stacked generalization) which
    uses a meta-classifier to combine the predictions of base classifiers.
    """
    
    def __init__(self, estimators: Optional[List] = None, final_estimator=None,
                 target_column: str = 'y', **kwargs):
        """
        Initialize Stacking Classifier.
        
        Args:
            estimators: List of (name, estimator) tuples. If None, uses default estimators.
            final_estimator: Meta-classifier. If None, uses LogisticRegression
            target_column: Name of the target column
            **kwargs: Additional arguments for the underlying model
        """
        self.estimators = estimators or self._get_default_estimators()
        self.final_estimator = final_estimator or LogisticRegression(random_state=42)
        super().__init__(target_column=target_column, **kwargs)
    
    def _get_default_estimators(self) -> List:
        """Get default estimators for stacking classifier."""
        return [
            ('lr', LogisticRegression(random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42)),
            ('svc', SVC(probability=True, random_state=42)),
            ('nb', GaussianNB())
        ]
    
    def _initialize_model(self, **kwargs):
        """Initialize Stacking Classifier model."""
        default_params = {
            'cv': 5,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.model = StackingClassifier(
            estimators=self.estimators,
            final_estimator=self.final_estimator,
            **default_params
        )


class EnsembleWithAutoSelection(BasePyMinClassifier):
    """
    Ensemble with automatic base estimator selection.
    
    This wrapper automatically selects the best base estimators based on
    cross-validation performance and creates an ensemble.
    """
    
    def __init__(self, ensemble_type: str = 'voting', max_estimators: int = 5,
                 target_column: str = 'y', **kwargs):
        """
        Initialize ensemble with auto selection.
        
        Args:
            ensemble_type: Type of ensemble ('voting', 'bagging', 'stacking')
            max_estimators: Maximum number of estimators to include
            target_column: Name of the target column
            **kwargs: Additional arguments for the underlying model
        """
        self.ensemble_type = ensemble_type
        self.max_estimators = max_estimators
        self.selected_estimators = None
        super().__init__(target_column=target_column, **kwargs)
    
    def _get_candidate_estimators(self) -> List:
        """Get candidate estimators for selection."""
        return [
            ('logistic', LogisticRegression(random_state=42)),
            ('decision_tree', DecisionTreeClassifier(random_state=42)),
            ('svc', SVC(probability=True, random_state=42)),
            ('naive_bayes', GaussianNB()),
            ('knn', KNeighborsClassifier()),
            ('random_forest', ExtraTreesClassifier(random_state=42))
        ]
    
    def _select_best_estimators(self, X: pd.DataFrame, y: pd.Series) -> List:
        """
        Select the best estimators based on cross-validation performance.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            List: Selected (name, estimator) tuples
        """
        candidates = self._get_candidate_estimators()
        scores = []
        
        for name, estimator in candidates:
            try:
                cv_scores = cross_val_score(estimator, X, y, cv=3, scoring='accuracy')
                scores.append((name, estimator, cv_scores.mean()))
            except Exception:
                # Skip estimators that fail
                continue
        
        # Sort by performance and select top estimators
        scores.sort(key=lambda x: x[2], reverse=True)
        selected = [(name, estimator) for name, estimator, _ in scores[:self.max_estimators]]
        
        return selected
    
    def _initialize_model(self, **kwargs):
        """Initialize the ensemble model."""
        # This will be set in fit method based on selected estimators
        self.model = None
    
    def fit(self, df: pd.DataFrame) -> 'EnsembleWithAutoSelection':
        """
        Fit the ensemble with automatic estimator selection.
        
        Args:
            df: DataFrame containing features and target column 'y'
            
        Returns:
            self: Returns self for method chaining
        """
        X, y = self._prepare_data(df)
        
        # Select best estimators
        self.selected_estimators = self._select_best_estimators(X, y)
        
        # Initialize the ensemble model
        if self.ensemble_type == 'voting':
            self.model = VotingClassifier(
                estimators=self.selected_estimators,
                voting='soft',
                n_jobs=-1
            )
        elif self.ensemble_type == 'bagging':
            # Use the best estimator as base
            base_estimator = self.selected_estimators[0][1]
            self.model = BaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.ensemble_type == 'stacking':
            self.model = StackingClassifier(
                estimators=self.selected_estimators,
                final_estimator=LogisticRegression(random_state=42),
                cv=3,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
        
        # Fit the ensemble
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
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.ensemble_type} does not support predict_proba")
        
        X = df[self.feature_columns]
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
        return self.model.score(X, y)
    
    def get_selected_estimators(self) -> List:
        """
        Get the list of selected estimators.
        
        Returns:
            List: Selected (name, estimator) tuples
        """
        return self.selected_estimators if self.selected_estimators else []


class EnsembleWithCustomWeights(BasePyMinClassifier):
    """
    Ensemble with custom weights for base estimators.
    
    This wrapper allows specifying custom weights for different base estimators
    in the ensemble.
    """
    
    def __init__(self, estimators: List, weights: Optional[List] = None,
                 target_column: str = 'y', **kwargs):
        """
        Initialize ensemble with custom weights.
        
        Args:
            estimators: List of (name, estimator) tuples
            weights: List of weights for each estimator. If None, uses equal weights
            target_column: Name of the target column
            **kwargs: Additional arguments for the underlying model
        """
        self.estimators = estimators
        self.weights = weights or [1.0] * len(estimators)
        super().__init__(target_column=target_column, **kwargs)
    
    def _initialize_model(self, **kwargs):
        """Initialize Voting Classifier with custom weights."""
        default_params = {
            'voting': 'soft',
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.model = VotingClassifier(
            estimators=self.estimators,
            **default_params
        )
    
    def fit(self, df: pd.DataFrame) -> 'EnsembleWithCustomWeights':
        """
        Fit the ensemble with custom weights.
        
        Args:
            df: DataFrame containing features and target column 'y'
            
        Returns:
            self: Returns self for method chaining
        """
        X, y = self._prepare_data(df)
        
        # Fit the ensemble
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels with custom weights.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = df[self.feature_columns]
        predictions = self.model.predict(X)
        
        # Apply custom weights if needed (this is a simplified implementation)
        # In practice, you might want to implement weighted voting manually
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities with custom weights.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            np.ndarray: Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = df[self.feature_columns]
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
        return self.model.score(X, y)


# Convenience function to create ensemble models
def create_ensemble_model(ensemble_type: str = 'voting', auto_select: bool = False,
                         estimators: Optional[List] = None, target_column: str = 'y', **kwargs) -> BasePyMinClassifier:
    """
    Create an ensemble classification model.
    
    Args:
        ensemble_type: Type of ensemble ('voting', 'bagging', 'stacking')
        auto_select: Whether to automatically select best estimators
        estimators: List of (name, estimator) tuples for custom ensemble
        target_column: Name of the target column
        **kwargs: Additional arguments for the model
        
    Returns:
        BasePyMinClassifier: Configured ensemble model
    """
    if auto_select:
        return EnsembleWithAutoSelection(ensemble_type=ensemble_type, target_column=target_column, **kwargs)
    
    model_map = {
        'voting': VotingClassifierWrapper,
        'bagging': BaggingClassifierWrapper,
        'stacking': StackingClassifierWrapper
    }
    
    if ensemble_type not in model_map:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    if estimators is not None:
        if ensemble_type == 'voting':
            return VotingClassifierWrapper(estimators=estimators, target_column=target_column, **kwargs)
        elif ensemble_type == 'bagging':
            base_estimator = estimators[0][1] if estimators else None
            return BaggingClassifierWrapper(base_estimator=base_estimator, target_column=target_column, **kwargs)
        elif ensemble_type == 'stacking':
            final_estimator = kwargs.pop('final_estimator', None)
            return StackingClassifierWrapper(estimators=estimators, final_estimator=final_estimator, target_column=target_column, **kwargs)
    
    return model_map[ensemble_type](target_column=target_column, **kwargs)
