"""
PyMin Regression Module

This module provides various scikit-learn regression models that work with DataFrames
where the target column is named 'y'.
"""

from .linear_models import LinearRegressionWrapper, RidgeRegressionWrapper, LassoRegressionWrapper, ElasticNetRegressionWrapper
from .tree_models import DecisionTreeRegressorWrapper, RandomForestRegressorWrapper, GradientBoostingRegressorWrapper
from .svm_models import SVRWrapper, LinearSVRWrapper
from .ensemble_models import VotingRegressorWrapper, BaggingRegressorWrapper, AdaBoostRegressorWrapper
from .neural_models import MLPRegressorWrapper

__all__ = [
    'LinearRegressionWrapper',
    'RidgeRegressionWrapper', 
    'LassoRegressionWrapper',
    'ElasticNetRegressionWrapper',
    'DecisionTreeRegressorWrapper',
    'RandomForestRegressorWrapper',
    'GradientBoostingRegressorWrapper',
    'SVRWrapper',
    'LinearSVRWrapper',
    'VotingRegressorWrapper',
    'BaggingRegressorWrapper',
    'AdaBoostRegressorWrapper',
    'MLPRegressorWrapper'
]
