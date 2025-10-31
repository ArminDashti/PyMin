"""
Tree-based regression models wrapper
"""

import pandas as pd
from typing import Any, Dict, Optional
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from .base import BaseRegressionWrapper


class DecisionTreeRegressorWrapper(BaseRegressionWrapper):
    """
    Wrapper for scikit-learn DecisionTreeRegressor.
    
    Parameters:
        criterion: Function to measure the quality of a split ('squared_error', 'friedman_mse', 'absolute_error', 'poisson')
        splitter: Strategy used to choose the split at each node ('best', 'random')
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum number of samples required to split an internal node
        min_samples_leaf: Minimum number of samples required to be at a leaf node
        min_weight_fraction_leaf: Minimum weighted fraction of the sum total of weights
        max_features: Number of features to consider when looking for the best split
        random_state: Random state
        max_leaf_nodes: Maximum number of leaf nodes
        min_impurity_decrease: Minimum impurity decrease required for a split
        ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning
    """
    
    def _create_model(self, **kwargs) -> DecisionTreeRegressor:
        """Create DecisionTreeRegressor model with given parameters."""
        return DecisionTreeRegressor(**kwargs)


class RandomForestRegressorWrapper(BaseRegressionWrapper):
    """
    Wrapper for scikit-learn RandomForestRegressor.
    
    Parameters:
        n_estimators: Number of trees in the forest
        criterion: Function to measure the quality of a split ('squared_error', 'absolute_error', 'friedman_mse', 'poisson')
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum number of samples required to split an internal node
        min_samples_leaf: Minimum number of samples required to be at a leaf node
        min_weight_fraction_leaf: Minimum weighted fraction of the sum total of weights
        max_features: Number of features to consider when looking for the best split
        max_leaf_nodes: Maximum number of leaf nodes
        min_impurity_decrease: Minimum impurity decrease required for a split
        bootstrap: Whether bootstrap samples are used when building trees
        oob_score: Whether to use out-of-bag samples to estimate the R^2 on unseen data
        n_jobs: Number of jobs to run in parallel
        random_state: Random state
        verbose: Controls the verbosity when fitting and predicting
        warm_start: When set to True, reuse the solution of the previous call
        ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning
        max_samples: Number of samples to draw from X to train each base estimator
    """
    
    def _create_model(self, **kwargs) -> RandomForestRegressor:
        """Create RandomForestRegressor model with given parameters."""
        return RandomForestRegressor(**kwargs)


class GradientBoostingRegressorWrapper(BaseRegressionWrapper):
    """
    Wrapper for scikit-learn GradientBoostingRegressor.
    
    Parameters:
        loss: Loss function to be optimized ('squared_error', 'absolute_error', 'huber', 'quantile')
        learning_rate: Learning rate shrinks the contribution of each tree
        n_estimators: Number of boosting stages to perform
        subsample: Fraction of samples to be used for fitting the individual base learners
        criterion: Function to measure the quality of a split ('squared_error', 'friedman_mse', 'absolute_error')
        min_samples_split: Minimum number of samples required to split an internal node
        min_samples_leaf: Minimum number of samples required to be at a leaf node
        min_weight_fraction_leaf: Minimum weighted fraction of the sum total of weights
        max_depth: Maximum depth of the individual regression estimators
        min_impurity_decrease: Minimum impurity decrease required for a split
        init: An estimator for computing the initial predictions
        random_state: Random state
        max_features: Number of features to consider when looking for the best split
        alpha: The alpha-quantile of the Huber loss function and the quantile loss function
        verbose: Enable verbose output
        max_leaf_nodes: Maximum number of leaf nodes
        warm_start: When set to True, reuse the solution of the previous call
        validation_fraction: Proportion of training data to set aside as validation set
        n_iter_no_change: Number of iterations with no improvement to wait before stopping
        tol: Tolerance for the early stopping
        ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning
    """
    
    def _create_model(self, **kwargs) -> GradientBoostingRegressor:
        """Create GradientBoostingRegressor model with given parameters."""
        return GradientBoostingRegressor(**kwargs)
