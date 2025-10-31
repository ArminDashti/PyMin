import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
from .regression import (
    LinearRegressionWrapper, RidgeRegressionWrapper, LassoRegressionWrapper, 
    ElasticNetRegressionWrapper, DecisionTreeRegressorWrapper, 
    RandomForestRegressorWrapper, GradientBoostingRegressorWrapper,
    SVRWrapper, LinearSVRWrapper, VotingRegressorWrapper, 
    BaggingRegressorWrapper, AdaBoostRegressorWrapper, MLPRegressorWrapper
)
from .classification import (
    LogisticRegressionClassifier, SGDClassifierWrapper, PerceptronClassifier,
    PassiveAggressiveClassifierWrapper, RidgeClassifierWrapper, RidgeClassifierCVWrapper,
    DecisionTreeClassifierWrapper, RandomForestClassifierWrapper, GradientBoostingClassifierWrapper,
    ExtraTreesClassifierWrapper, AdaBoostClassifierWrapper, HistGradientBoostingClassifierWrapper,
    SVCWrapper, LinearSVCWrapper, NuSVCWrapper, GaussianNBWrapper, MultinomialNBWrapper,
    BernoulliNBWrapper, ComplementNBWrapper, CategoricalNBWrapper, KNeighborsClassifierWrapper,
    RadiusNeighborsClassifierWrapper, NearestCentroidWrapper, VotingClassifierWrapper,
    BaggingClassifierWrapper, StackingClassifierWrapper
)

def simple_regression(algorithm: Optional[str], df: pd.DataFrame, test_size: float = 0.2, 
                     random_state: Optional[int] = None, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    if 'y' not in df.columns:
        raise ValueError("DataFrame must contain a column named 'y' for the target variable")
    regression_algorithms = {
        'linear': LinearRegressionWrapper,
        'ridge': RidgeRegressionWrapper,
        'lasso': LassoRegressionWrapper,
        'elastic_net': ElasticNetRegressionWrapper,
        'decision_tree': DecisionTreeRegressorWrapper,
        'random_forest': RandomForestRegressorWrapper,
        'gradient_boosting': GradientBoostingRegressorWrapper,
        'svr': SVRWrapper,
        'linear_svr': LinearSVRWrapper,
        'voting': VotingRegressorWrapper,
        'bagging': BaggingRegressorWrapper,
        'adaboost': AdaBoostRegressorWrapper,
        'mlp': MLPRegressorWrapper
    }
    X = df.drop('y', axis=1)
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    def test_algorithm(algo_name: str, algo_class):
        try:
            model = algo_class(**kwargs)
            model.fit(train_df)
            y_pred = model.predict(test_df)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            feature_importance = model.get_feature_importance()
            return {
                'algorithm': algo_name,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred,
                'feature_importance': feature_importance,
                'model': model,
                'status': 'success'
            }
        except Exception as e:
            return {
                'algorithm': algo_name,
                'error': str(e),
                'status': 'failed'
            }
    if algorithm is not None:
        if algorithm not in regression_algorithms:
            available = ', '.join(regression_algorithms.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Available algorithms: {available}")
        return test_algorithm(algorithm, regression_algorithms[algorithm])
    else:
        results = []
        for algo_name, algo_class in regression_algorithms.items():
            print(f"Testing {algo_name}...")
            result = test_algorithm(algo_name, algo_class)
            results.append(result)
        successful_results = [r for r in results if r['status'] == 'success']
        successful_results.sort(key=lambda x: x['r2'], reverse=True)
        return successful_results
        
def simple_classification(algorithm: Optional[str], df: pd.DataFrame, test_size: float = 0.2,
                         random_state: Optional[int] = None, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    if 'y' not in df.columns:
        raise ValueError("DataFrame must contain a column named 'y' for the target variable")

    classification_algorithms = {
        'logistic': LogisticRegressionClassifier,
        'sgd': SGDClassifierWrapper,
        'perceptron': PerceptronClassifier,
        'passive_aggressive': PassiveAggressiveClassifierWrapper,
        'ridge': RidgeClassifierWrapper,
        'ridge_cv': RidgeClassifierCVWrapper,
        'decision_tree': DecisionTreeClassifierWrapper,
        'random_forest': RandomForestClassifierWrapper,
        'gradient_boosting': GradientBoostingClassifierWrapper,
        'extra_trees': ExtraTreesClassifierWrapper,
        'adaboost': AdaBoostClassifierWrapper,
        'hist_gradient_boosting': HistGradientBoostingClassifierWrapper,
        'svc': SVCWrapper,
        'linear_svc': LinearSVCWrapper,
        'nu_svc': NuSVCWrapper,
        'gaussian_nb': GaussianNBWrapper,
        'multinomial_nb': MultinomialNBWrapper,
        'bernoulli_nb': BernoulliNBWrapper,
        'complement_nb': ComplementNBWrapper,
        'categorical_nb': CategoricalNBWrapper,
        'knn': KNeighborsClassifierWrapper,
        'radius_neighbors': RadiusNeighborsClassifierWrapper,
        'nearest_centroid': NearestCentroidWrapper,
        'voting': VotingClassifierWrapper,
        'bagging': BaggingClassifierWrapper,
        'stacking': StackingClassifierWrapper
    }
    X = df.drop('y', axis=1)
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    def test_algorithm(algo_name: str, algo_class):
        try:
            model = algo_class(**kwargs)
            model.fit(train_df)
            y_pred = model.predict(test_df)
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            feature_importance = model.get_feature_importance()
            try:
                y_proba = model.predict_proba(test_df)
            except:
                y_proba = None
            return {
                'algorithm': algo_name,
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'predictions': y_pred,
                'probabilities': y_proba,
                'feature_importance': feature_importance,
                'model': model,
                'status': 'success'
            }
        except Exception as e:
            return {
                'algorithm': algo_name,
                'error': str(e),
                'status': 'failed'
            }
    if algorithm is not None:
        if algorithm not in classification_algorithms:
            available = ', '.join(classification_algorithms.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Available algorithms: {available}")
        return test_algorithm(algorithm, classification_algorithms[algorithm])
    else:
        results = []
        for algo_name, algo_class in classification_algorithms.items():
            print(f"Testing {algo_name}...")
            result = test_algorithm(algo_name, algo_class)
            results.append(result)
        successful_results = [r for r in results if r['status'] == 'success']
        successful_results.sort(key=lambda x: x['accuracy'], reverse=True)
        return successful_results

def get_available_algorithms() -> Dict[str, List[str]]:
    return {
        'regression': [
            'linear', 'ridge', 'lasso', 'elastic_net', 'decision_tree',
            'random_forest', 'gradient_boosting', 'svr', 'linear_svr',
            'voting', 'bagging', 'adaboost', 'mlp'
        ],
        'classification': [
            'logistic', 'sgd', 'perceptron', 'passive_aggressive', 'ridge',
            'ridge_cv', 'decision_tree', 'random_forest', 'gradient_boosting',
            'extra_trees', 'adaboost', 'hist_gradient_boosting', 'svc',
            'linear_svc', 'nu_svc', 'gaussian_nb', 'multinomial_nb',
            'bernoulli_nb', 'complement_nb', 'categorical_nb', 'knn',
            'radius_neighbors', 'nearest_centroid', 'voting', 'bagging', 'stacking'
        ]
    }
def quick_regression(df: pd.DataFrame, test_size: float = 0.2) -> List[Dict[str, Any]]:
    return simple_regression(None, df, test_size)
def quick_classification(df: pd.DataFrame, test_size: float = 0.2) -> List[Dict[str, Any]]:
    return simple_classification(None, df, test_size)
