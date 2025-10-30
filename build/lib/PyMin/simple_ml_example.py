"""
Example usage of PyMin's simple ML interface

This script demonstrates how to use the simple_regression and simple_classification
functions with just algorithm name, DataFrame, and test size.
"""

import pandas as pd
import numpy as np
from PyMin.simple_ml import (
    simple_regression, 
    simple_classification,
    quick_regression,
    quick_classification,
    get_available_algorithms
)


def create_sample_regression_data(n_samples=1000, n_features=5, noise=0.1):
    """Create sample regression dataset."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    # Create target with some linear relationship + noise
    y = 2 * X[:, 0] + 1.5 * X[:, 1] - 0.8 * X[:, 2] + np.random.normal(0, noise, n_samples)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['y'] = y
    
    return df


def create_sample_classification_data(n_samples=1000, n_features=5):
    """Create sample classification dataset."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    # Create target with some decision boundary
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.normal(0, 0.5, n_samples) > 0).astype(int)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['y'] = y
    
    return df


def main():
    print("PyMin Simple ML Interface Example")
    print("=" * 50)
    
    # Show available algorithms
    print("\nAvailable algorithms:")
    algorithms = get_available_algorithms()
    print(f"Regression: {', '.join(algorithms['regression'])}")
    print(f"Classification: {', '.join(algorithms['classification'])}")
    
    # Example 1: Regression - Test specific algorithm
    print("\n" + "="*50)
    print("Example 1: Regression - Test specific algorithm")
    print("="*50)
    
    reg_df = create_sample_regression_data()
    print(f"Created regression dataset with {len(reg_df)} samples and {len(reg_df.columns)-1} features")
    
    # Test Random Forest specifically
    result = simple_regression('random_forest', reg_df, test_size=0.2)
    print(f"\nRandom Forest Results:")
    print(f"R² Score: {result['r2']:.4f}")
    print(f"MSE: {result['mse']:.4f}")
    print(f"MAE: {result['mae']:.4f}")
    
    # Example 2: Regression - Test all algorithms
    print("\n" + "="*50)
    print("Example 2: Regression - Test all algorithms (top 3)")
    print("="*50)
    
    all_reg_results = quick_regression(reg_df, test_size=0.2)
    print("\nTop 3 Regression Algorithms:")
    for i, result in enumerate(all_reg_results[:3]):
        print(f"{i+1}. {result['algorithm']}: R² = {result['r2']:.4f}, MSE = {result['mse']:.4f}")
    
    # Example 3: Classification - Test specific algorithm
    print("\n" + "="*50)
    print("Example 3: Classification - Test specific algorithm")
    print("="*50)
    
    clf_df = create_sample_classification_data()
    print(f"Created classification dataset with {len(clf_df)} samples and {len(clf_df.columns)-1} features")
    
    # Test Random Forest specifically
    result = simple_classification('random_forest', clf_df, test_size=0.2)
    print(f"\nRandom Forest Results:")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Classification Report:")
    print(result['classification_report'])
    
    # Example 4: Classification - Test all algorithms
    print("\n" + "="*50)
    print("Example 4: Classification - Test all algorithms (top 3)")
    print("="*50)
    
    all_clf_results = quick_classification(clf_df, test_size=0.2)
    print("\nTop 3 Classification Algorithms:")
    for i, result in enumerate(all_clf_results[:3]):
        print(f"{i+1}. {result['algorithm']}: Accuracy = {result['accuracy']:.4f}")
    
    # Example 5: Show feature importance
    print("\n" + "="*50)
    print("Example 5: Feature Importance")
    print("="*50)
    
    best_reg = all_reg_results[0]
    if best_reg['feature_importance'] is not None:
        print(f"\nFeature importance from best regression model ({best_reg['algorithm']}):")
        print(best_reg['feature_importance'])
    
    best_clf = all_clf_results[0]
    if best_clf['feature_importance'] is not None:
        print(f"\nFeature importance from best classification model ({best_clf['algorithm']}):")
        print(best_clf['feature_importance'])


if __name__ == "__main__":
    main()
