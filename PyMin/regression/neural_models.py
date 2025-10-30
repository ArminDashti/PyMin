"""
Neural network regression models wrapper
"""

import pandas as pd
from typing import Any, Dict, List, Optional, Union
from sklearn.neural_network import MLPRegressor
from .base import BaseRegressionWrapper


class MLPRegressorWrapper(BaseRegressionWrapper):
    """
    Wrapper for scikit-learn MLPRegressor (Multi-layer Perceptron Regressor).
    
    Parameters:
        hidden_layer_sizes: Number of neurons in each hidden layer
        activation: Activation function for the hidden layers ('identity', 'logistic', 'tanh', 'relu')
        solver: The solver for weight optimization ('lbfgs', 'sgd', 'adam')
        alpha: L2 penalty (regularization term) parameter
        batch_size: Size of minibatches for stochastic optimizers
        learning_rate: Learning rate schedule for weight updates ('constant', 'invscaling', 'adaptive')
        learning_rate_init: The initial learning rate used
        power_t: The exponent for inverse scaling learning rate
        max_iter: Maximum number of iterations
        shuffle: Whether to shuffle samples in each iteration
        random_state: Random state
        tol: Tolerance for the optimization
        verbose: Whether to print progress messages
        warm_start: When set to True, reuse the solution of the previous call
        momentum: Momentum for gradient descent update
        nesterovs_momentum: Whether to use Nesterov's momentum
        early_stopping: Whether to use early stopping to terminate training
        validation_fraction: Proportion of training data to set aside as validation set
        beta_1: Exponential decay rate for estimates of first moment vector in adam
        beta_2: Exponential decay rate for estimates of second moment vector in adam
        epsilon: Value for numerical stability in adam
        n_iter_no_change: Maximum number of epochs to not meet tol improvement
        max_fun: Only used when solver='lbfgs'. Maximum number of function calls
    """
    
    def _create_model(self, **kwargs) -> MLPRegressor:
        """Create MLPRegressor model with given parameters."""
        return MLPRegressor(**kwargs)
