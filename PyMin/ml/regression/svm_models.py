"""
Support Vector Machine regression models wrapper
"""

import pandas as pd
from typing import Any, Dict, Optional
from sklearn.svm import SVR, LinearSVR
from .base import BaseRegressionWrapper


class SVRWrapper(BaseRegressionWrapper):
    """
    Wrapper for scikit-learn SVR (Support Vector Regression).
    
    Parameters:
        kernel: Specifies the kernel type to be used in the algorithm ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
        degree: Degree of the polynomial kernel function ('poly')
        gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        coef0: Independent term in kernel function
        tol: Tolerance for stopping criterion
        C: Regularization parameter
        epsilon: Epsilon in the epsilon-SVR model
        shrinking: Whether to use the shrinking heuristic
        cache_size: Specify the size of the kernel cache
        verbose: Enable verbose output
        max_iter: Hard limit on iterations within solver
    """
    
    def _create_model(self, **kwargs) -> SVR:
        """Create SVR model with given parameters."""
        return SVR(**kwargs)


class LinearSVRWrapper(BaseRegressionWrapper):
    """
    Wrapper for scikit-learn LinearSVR (Linear Support Vector Regression).
    
    Parameters:
        epsilon: Epsilon parameter in the epsilon-insensitive loss function
        tol: Tolerance for stopping criterion
        C: Regularization parameter
        loss: Specifies the loss function ('epsilon_insensitive', 'squared_epsilon_insensitive')
        fit_intercept: Whether to calculate the intercept for this model
        intercept_scaling: When self.fit_intercept is True, instance vector x becomes [x, self.intercept_scaling]
        dual: Whether to solve the dual or primal optimization problem
        verbose: Enable verbose output
        random_state: Random state
        max_iter: Maximum number of iterations
    """
    
    def _create_model(self, **kwargs) -> LinearSVR:
        """Create LinearSVR model with given parameters."""
        return LinearSVR(**kwargs)
