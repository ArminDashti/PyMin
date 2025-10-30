"""
Ensemble regression models wrapper
"""

import pandas as pd
from typing import Any, Dict, List, Optional, Union
from sklearn.ensemble import VotingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.base import BaseEstimator
from .base import BaseRegressionWrapper


class VotingRegressorWrapper(BaseRegressionWrapper):
    """
    Wrapper for scikit-learn VotingRegressor.
    
    Parameters:
        estimators: List of (name, estimator) tuples
        weights: Sequence of weights for averaging
        n_jobs: Number of jobs to run in parallel
        verbose: Enable verbose output
    """
    
    def __init__(self, estimators: List[tuple], **kwargs):
        """
        Initialize VotingRegressor wrapper.
        
        Args:
            estimators: List of (name, estimator) tuples
            **kwargs: Additional parameters for VotingRegressor
        """
        self.estimators = estimators
        super().__init__(**kwargs)
    
    def _create_model(self, **kwargs) -> VotingRegressor:
        """Create VotingRegressor model with given parameters."""
        return VotingRegressor(estimators=self.estimators, **kwargs)


class BaggingRegressorWrapper(BaseRegressionWrapper):
    """
    Wrapper for scikit-learn BaggingRegressor.
    
    Parameters:
        base_estimator: Base estimator to fit on random subsets of the dataset
        n_estimators: Number of base estimators in the ensemble
        max_samples: Number of samples to draw from X to train each base estimator
        max_features: Number of features to draw from X to train each base estimator
        bootstrap: Whether samples are drawn with replacement
        bootstrap_features: Whether features are drawn with replacement
        oob_score: Whether to use out-of-bag samples to estimate the generalization error
        warm_start: When set to True, reuse the solution of the previous call
        n_jobs: Number of jobs to run in parallel
        random_state: Random state
        verbose: Enable verbose output
    """
    
    def __init__(self, base_estimator: Optional[BaseEstimator] = None, **kwargs):
        """
        Initialize BaggingRegressor wrapper.
        
        Args:
            base_estimator: Base estimator to fit on random subsets
            **kwargs: Additional parameters for BaggingRegressor
        """
        self.base_estimator = base_estimator
        super().__init__(**kwargs)
    
    def _create_model(self, **kwargs) -> BaggingRegressor:
        """Create BaggingRegressor model with given parameters."""
        return BaggingRegressor(base_estimator=self.base_estimator, **kwargs)


class AdaBoostRegressorWrapper(BaseRegressionWrapper):
    """
    Wrapper for scikit-learn AdaBoostRegressor.
    
    Parameters:
        base_estimator: Base estimator from which the boosted ensemble is built
        n_estimators: Number of estimators to terminate boosting
        learning_rate: Learning rate shrinks the contribution of each classifier
        loss: The loss function to use when updating the weights after each boosting iteration
        random_state: Random state
    """
    
    def __init__(self, base_estimator: Optional[BaseEstimator] = None, **kwargs):
        """
        Initialize AdaBoostRegressor wrapper.
        
        Args:
            base_estimator: Base estimator from which the boosted ensemble is built
            **kwargs: Additional parameters for AdaBoostRegressor
        """
        self.base_estimator = base_estimator
        super().__init__(**kwargs)
    
    def _create_model(self, **kwargs) -> AdaBoostRegressor:
        """Create AdaBoostRegressor model with given parameters."""
        return AdaBoostRegressor(base_estimator=self.base_estimator, **kwargs)
