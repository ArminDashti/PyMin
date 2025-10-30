"""
Example usage of PyMin classification models.

This script demonstrates how to use various classification algorithms
from the PyMin classification module with pandas DataFrames.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, load_iris, load_wine
from sklearn.model_selection import train_test_split

# Import PyMin classification models
from PyMin.classification import (
    # Individual classifiers
    LogisticRegressionClassifier,
    RandomForestClassifierWrapper,
    SVCWrapper,
    GaussianNBWrapper,
    KNeighborsClassifierWrapper,
    VotingClassifierWrapper,
    
    # Convenience functions
    create_classifier,
    create_linear_model,
    create_tree_model,
    create_svm_model,
    create_naive_bayes_model,
    create_neighbor_model,
    create_ensemble_model
)


def create_sample_data():
    """Create sample classification datasets."""
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['y'] = y
    
    return df


def demonstrate_linear_models(df):
    """Demonstrate linear classification models."""
    print("=== Linear Models ===")
    
    # Logistic Regression
    print("\n1. Logistic Regression:")
    clf = LogisticRegressionClassifier()
    clf.fit(df)
    accuracy = clf.score(df)
    print(f"   Accuracy: {accuracy:.3f}")
    
    # SGD Classifier with scaling
    print("\n2. SGD Classifier with scaling:")
    clf = create_linear_model('sgd', with_scaling=True)
    clf.fit(df)
    accuracy = clf.score(df)
    print(f"   Accuracy: {accuracy:.3f}")
    
    # Ridge Classifier
    print("\n3. Ridge Classifier:")
    clf = RidgeClassifierWrapper()
    clf.fit(df)
    accuracy = clf.score(df)
    print(f"   Accuracy: {accuracy:.3f}")


def demonstrate_tree_models(df):
    """Demonstrate tree-based classification models."""
    print("\n=== Tree Models ===")
    
    # Random Forest
    print("\n1. Random Forest:")
    clf = RandomForestClassifierWrapper(n_estimators=50)
    clf.fit(df)
    accuracy = clf.score(df)
    print(f"   Accuracy: {accuracy:.3f}")
    
    # Feature importance
    importance = clf.get_feature_importance()
    print(f"   Top 3 features: {importance.head(3).index.tolist()}")
    
    # Gradient Boosting with feature selection
    print("\n2. Gradient Boosting with feature selection:")
    clf = create_tree_model('gradient_boosting', with_feature_selection=True)
    clf.fit(df)
    accuracy = clf.score(df)
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Selected features: {clf.get_selected_features()}")


def demonstrate_svm_models(df):
    """Demonstrate SVM classification models."""
    print("\n=== SVM Models ===")
    
    # SVC with automatic scaling
    print("\n1. SVC with scaling:")
    clf = create_svm_model('svc', with_scaling=True)
    clf.fit(df)
    accuracy = clf.score(df)
    print(f"   Accuracy: {accuracy:.3f}")
    
    # SVC with automatic kernel selection
    print("\n2. SVC with auto kernel selection:")
    clf = create_svm_model('svc', auto_kernel=True)
    clf.fit(df)
    accuracy = clf.score(df)
    kernel = clf.get_kernel()
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Selected kernel: {kernel}")


def demonstrate_naive_bayes_models(df):
    """Demonstrate Naive Bayes classification models."""
    print("\n=== Naive Bayes Models ===")
    
    # Gaussian Naive Bayes
    print("\n1. Gaussian Naive Bayes:")
    clf = GaussianNBWrapper()
    clf.fit(df)
    accuracy = clf.score(df)
    print(f"   Accuracy: {accuracy:.3f}")
    
    # Auto-selection Naive Bayes
    print("\n2. Auto-selection Naive Bayes:")
    clf = create_naive_bayes_model('auto')
    clf.fit(df)
    accuracy = clf.score(df)
    algorithm = clf.get_selected_algorithm()
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Selected algorithm: {algorithm}")


def demonstrate_neighbor_models(df):
    """Demonstrate neighbor-based classification models."""
    print("\n=== Neighbor Models ===")
    
    # k-NN with scaling
    print("\n1. k-NN with scaling:")
    clf = create_neighbor_model('knn', with_scaling=True)
    clf.fit(df)
    accuracy = clf.score(df)
    print(f"   Accuracy: {accuracy:.3f}")
    
    # k-NN with automatic k selection
    print("\n2. k-NN with auto k selection:")
    clf = create_neighbor_model('knn', auto_k=True)
    clf.fit(df)
    accuracy = clf.score(df)
    best_k = clf.get_best_k()
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Best k: {best_k}")


def demonstrate_ensemble_models(df):
    """Demonstrate ensemble classification models."""
    print("\n=== Ensemble Models ===")
    
    # Voting Classifier
    print("\n1. Voting Classifier:")
    clf = VotingClassifierWrapper()
    clf.fit(df)
    accuracy = clf.score(df)
    print(f"   Accuracy: {accuracy:.3f}")
    
    # Auto-selection Ensemble
    print("\n2. Auto-selection Ensemble:")
    clf = create_ensemble_model('voting', auto_select=True)
    clf.fit(df)
    accuracy = clf.score(df)
    selected_estimators = clf.get_selected_estimators()
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Selected estimators: {[name for name, _ in selected_estimators]}")


def demonstrate_train_test_split(df):
    """Demonstrate train-test split functionality."""
    print("\n=== Train-Test Split ===")
    
    # Create a classifier
    clf = RandomForestClassifierWrapper(n_estimators=50)
    
    # Split the data
    train_df, test_df = clf.train_test_split(df, test_size=0.3, random_state=42)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Train and evaluate
    clf.fit(train_df)
    train_accuracy = clf.score(train_df)
    test_accuracy = clf.score(test_df)
    
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")


def demonstrate_evaluation(df):
    """Demonstrate detailed model evaluation."""
    print("\n=== Detailed Evaluation ===")
    
    # Create and train a classifier
    clf = RandomForestClassifierWrapper(n_estimators=50)
    clf.fit(df)
    
    # Get detailed evaluation
    results = clf.evaluate(df, detailed=True)
    
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Classification Report:")
    print(results['classification_report'])
    print(f"Confusion Matrix:")
    print(np.array(results['confusion_matrix']))


def demonstrate_predictions(df):
    """Demonstrate prediction functionality."""
    print("\n=== Predictions ===")
    
    # Create a classifier
    clf = LogisticRegressionClassifier()
    clf.fit(df)
    
    # Make predictions
    predictions = clf.predict(df)
    probabilities = clf.predict_proba(df)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Sample probabilities: {probabilities[:5]}")


def demonstrate_custom_target_column():
    """Demonstrate using custom target column name."""
    print("\n=== Custom Target Column ===")
    
    # Create data with custom target column
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y  # Custom target column name
    
    # Create classifier with custom target column
    clf = LogisticRegressionClassifier(target_column='target')
    clf.fit(df)
    accuracy = clf.score(df)
    
    print(f"Accuracy with custom target column: {accuracy:.3f}")


def main():
    """Main demonstration function."""
    print("PyMin Classification Models Demo")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['y'].value_counts()}")
    
    # Demonstrate different model types
    demonstrate_linear_models(df)
    demonstrate_tree_models(df)
    demonstrate_svm_models(df)
    demonstrate_naive_bayes_models(df)
    demonstrate_neighbor_models(df)
    demonstrate_ensemble_models(df)
    
    # Demonstrate utility functions
    demonstrate_train_test_split(df)
    demonstrate_evaluation(df)
    demonstrate_predictions(df)
    demonstrate_custom_target_column()
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
