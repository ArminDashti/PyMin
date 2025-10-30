"""
PyMin Classification Module

This module provides a comprehensive set of scikit-learn classification algorithms
wrapped for easy use with pandas DataFrames. All classifiers expect the target
column to be named 'y' by default.

Available Classifiers:
- Linear Models: Logistic Regression, SGD, Perceptron, Ridge, etc.
- Tree Models: Decision Tree, Random Forest, Gradient Boosting, etc.
- SVM Models: SVC, LinearSVC, NuSVC with automatic scaling
- Naive Bayes: Gaussian, Multinomial, Bernoulli, Complement, Categorical
- Neighbor Models: k-NN, Radius Neighbors, Nearest Centroid
- Ensemble Models: Voting, Bagging, Stacking with auto-selection

Usage:
    from PyMin.classification import LogisticRegressionClassifier, RandomForestClassifier
    
    # Create classifier
    clf = LogisticRegressionClassifier()
    
    # Fit on DataFrame with 'y' column
    clf.fit(df)
    
    # Make predictions
    predictions = clf.predict(df_test)
    
    # Get probabilities
    probabilities = clf.predict_proba(df_test)
    
    # Evaluate model
    results = clf.evaluate(df_test)
"""

# Import base classifier
from .base_classifier import BasePyMinClassifier

# Import linear models
from .linear_models import (
    LogisticRegressionClassifier,
    SGDClassifierWrapper,
    PerceptronClassifier,
    PassiveAggressiveClassifierWrapper,
    RidgeClassifierWrapper,
    RidgeClassifierCVWrapper,
    LinearModelsWithScaling,
    create_linear_model
)

# Import tree models
from .tree_models import (
    DecisionTreeClassifierWrapper,
    RandomForestClassifierWrapper,
    GradientBoostingClassifierWrapper,
    ExtraTreesClassifierWrapper,
    AdaBoostClassifierWrapper,
    HistGradientBoostingClassifierWrapper,
    TreeModelsWithFeatureSelection,
    create_tree_model
)

# Import SVM models
from .svm_models import (
    SVCWrapper,
    LinearSVCWrapper,
    NuSVCWrapper,
    SVMWithScaling,
    SVMWithKernelSelection,
    create_svm_model
)

# Import Naive Bayes models
from .naive_bayes import (
    GaussianNBWrapper,
    MultinomialNBWrapper,
    BernoulliNBWrapper,
    ComplementNBWrapper,
    CategoricalNBWrapper,
    NaiveBayesWithAutoSelection,
    create_naive_bayes_model
)

# Import neighbor models
from .neighbor_models import (
    KNeighborsClassifierWrapper,
    RadiusNeighborsClassifierWrapper,
    NearestCentroidWrapper,
    NeighborModelsWithScaling,
    KNNWithAutoK,
    NeighborModelsWithDistanceWeights,
    create_neighbor_model
)

# Import ensemble models
from .ensemble_models import (
    VotingClassifierWrapper,
    BaggingClassifierWrapper,
    StackingClassifierWrapper,
    EnsembleWithAutoSelection,
    EnsembleWithCustomWeights,
    create_ensemble_model
)

# Define what gets imported with "from PyMin.classification import *"
__all__ = [
    # Base class
    'BasePyMinClassifier',
    
    # Linear models
    'LogisticRegressionClassifier',
    'SGDClassifierWrapper',
    'PerceptronClassifier',
    'PassiveAggressiveClassifierWrapper',
    'RidgeClassifierWrapper',
    'RidgeClassifierCVWrapper',
    'LinearModelsWithScaling',
    'create_linear_model',
    
    # Tree models
    'DecisionTreeClassifierWrapper',
    'RandomForestClassifierWrapper',
    'GradientBoostingClassifierWrapper',
    'ExtraTreesClassifierWrapper',
    'AdaBoostClassifierWrapper',
    'HistGradientBoostingClassifierWrapper',
    'TreeModelsWithFeatureSelection',
    'create_tree_model',
    
    # SVM models
    'SVCWrapper',
    'LinearSVCWrapper',
    'NuSVCWrapper',
    'SVMWithScaling',
    'SVMWithKernelSelection',
    'create_svm_model',
    
    # Naive Bayes models
    'GaussianNBWrapper',
    'MultinomialNBWrapper',
    'BernoulliNBWrapper',
    'ComplementNBWrapper',
    'CategoricalNBWrapper',
    'NaiveBayesWithAutoSelection',
    'create_naive_bayes_model',
    
    # Neighbor models
    'KNeighborsClassifierWrapper',
    'RadiusNeighborsClassifierWrapper',
    'NearestCentroidWrapper',
    'NeighborModelsWithScaling',
    'KNNWithAutoK',
    'NeighborModelsWithDistanceWeights',
    'create_neighbor_model',
    
    # Ensemble models
    'VotingClassifierWrapper',
    'BaggingClassifierWrapper',
    'StackingClassifierWrapper',
    'EnsembleWithAutoSelection',
    'EnsembleWithCustomWeights',
    'create_ensemble_model'
]

# Convenience function to create any classifier
def create_classifier(algorithm: str, **kwargs):
    """
    Create a classifier by algorithm name.
    
    Args:
        algorithm: Algorithm name (e.g., 'logistic', 'random_forest', 'svc', etc.)
        **kwargs: Additional arguments for the classifier
        
    Returns:
        BasePyMinClassifier: Configured classifier
        
    Examples:
        >>> clf = create_classifier('logistic')
        >>> clf = create_classifier('random_forest', n_estimators=100)
        >>> clf = create_classifier('svc', kernel='rbf', C=10)
    """
    algorithm_map = {
        # Linear models
        'logistic': LogisticRegressionClassifier,
        'sgd': SGDClassifierWrapper,
        'perceptron': PerceptronClassifier,
        'passive_aggressive': PassiveAggressiveClassifierWrapper,
        'ridge': RidgeClassifierWrapper,
        'ridge_cv': RidgeClassifierCVWrapper,
        
        # Tree models
        'decision_tree': DecisionTreeClassifierWrapper,
        'random_forest': RandomForestClassifierWrapper,
        'gradient_boosting': GradientBoostingClassifierWrapper,
        'extra_trees': ExtraTreesClassifierWrapper,
        'adaboost': AdaBoostClassifierWrapper,
        'hist_gradient_boosting': HistGradientBoostingClassifierWrapper,
        
        # SVM models
        'svc': SVCWrapper,
        'linear_svc': LinearSVCWrapper,
        'nu_svc': NuSVCWrapper,
        
        # Naive Bayes models
        'gaussian_nb': GaussianNBWrapper,
        'multinomial_nb': MultinomialNBWrapper,
        'bernoulli_nb': BernoulliNBWrapper,
        'complement_nb': ComplementNBWrapper,
        'categorical_nb': CategoricalNBWrapper,
        
        # Neighbor models
        'knn': KNeighborsClassifierWrapper,
        'radius_neighbors': RadiusNeighborsClassifierWrapper,
        'nearest_centroid': NearestCentroidWrapper,
        
        # Ensemble models
        'voting': VotingClassifierWrapper,
        'bagging': BaggingClassifierWrapper,
        'stacking': StackingClassifierWrapper
    }
    
    if algorithm not in algorithm_map:
        available = ', '.join(algorithm_map.keys())
        raise ValueError(f"Unknown algorithm: {algorithm}. Available algorithms: {available}")
    
    return algorithm_map[algorithm](**kwargs)

# Version information
__version__ = "1.0.0"
__author__ = "PyMin Team"
__description__ = "Comprehensive scikit-learn classification algorithms for PyMin"
