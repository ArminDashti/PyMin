"""
Linear regression models wrapper
"""

import pandas as pd
from typing import Any, Dict, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from .base import BaseRegressionWrapper


class LinearRegressionWrapper(BaseRegressionWrapper):
    """
    Wrapper for scikit-learn LinearRegression.
    
    Parameters:
        fit_intercept: Whether to calculate the intercept for this model
        copy_X: Whether to copy X; if False, X may be overwritten
        n_jobs: Number of jobs to use for computation
        positive: Whether to force coefficients to be positive
    """
    
    def _create_model(self, **kwargs) -> LinearRegression:
        """Create LinearRegression model with given parameters."""
        return LinearRegression(**kwargs)


class RidgeRegressionWrapper(BaseRegressionWrapper):
    """
    Wrapper for scikit-learn Ridge regression.
    
    Parameters:
        alpha: Regularization strength; must be a positive float
        fit_intercept: Whether to calculate the intercept for this model
        copy_X: Whether to copy X; if False, X may be overwritten
        max_iter: Maximum number of iterations for conjugate gradient solver
        tol: Precision of the solution
        solver: Solver to use ('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga')
        random_state: Random state for 'sag' and 'saga' solvers
    """
    
    def _create_model(self, **kwargs) -> Ridge:
        """Create Ridge model with given parameters."""
        return Ridge(**kwargs)


class LassoRegressionWrapper(BaseRegressionWrapper):
    """
    Wrapper for scikit-learn Lasso regression.
    
    Parameters:
        alpha: Regularization strength; must be a positive float
        fit_intercept: Whether to calculate the intercept for this model
        copy_X: Whether to copy X; if False, X may be overwritten
        max_iter: Maximum number of iterations
        tol: Precision of the solution
        warm_start: Whether to reuse previous solution
        positive: Whether to force coefficients to be positive
        random_state: Random state for 'saga' solver
        selection: Strategy for updating coefficients ('cyclic', 'random')
    """
    
    def _create_model(self, **kwargs) -> Lasso:
        """Create Lasso model with given parameters."""
        return Lasso(**kwargs)


class ElasticNetRegressionWrapper(BaseRegressionWrapper):
    """
    Wrapper for scikit-learn ElasticNet regression.
    
    Parameters:
        alpha: Regularization strength; must be a positive float
        l1_ratio: The ElasticNet mixing parameter (0 <= l1_ratio <= 1)
        fit_intercept: Whether to calculate the intercept for this model
        copy_X: Whether to copy X; if False, X may be overwritten
        max_iter: Maximum number of iterations
        tol: Precision of the solution
        warm_start: Whether to reuse previous solution
        positive: Whether to force coefficients to be positive
        random_state: Random state for 'saga' solver
        selection: Strategy for updating coefficients ('cyclic', 'random')
    """
    
    def _create_model(self, **kwargs) -> ElasticNet:
        """Create ElasticNet model with given parameters."""
        return ElasticNet(**kwargs)
