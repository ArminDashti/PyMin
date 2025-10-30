#!/usr/bin/env python3
"""
PyMin - Main entry point for the PyMin CLI tool
"""

import sys
import argparse
from pathlib import Path

# Add the PyMin directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point for PyMin CLI"""
    parser = argparse.ArgumentParser(
        description="PyMin - A Python toolkit for various data science and AI tasks",
        prog="pymin"
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # OpenRouter command
    openrouter_parser = subparsers.add_parser('openrouter', help='OpenRouter API operations')
    openrouter_parser.add_argument('--model', default='gpt-4o-mini', help='Model to use')
    openrouter_parser.add_argument('--message', required=True, help='Message to send')
    openrouter_parser.add_argument('--api-key', help='OpenRouter API key (or set OPENROUTER_API_KEY env var)')
    
    # Database command
    db_parser = subparsers.add_parser('db', help='Database operations')
    db_subparsers = db_parser.add_subparsers(dest='db_command', help='Database subcommands')
    
    # MSSQL subcommand
    mssql_parser = db_subparsers.add_parser('mssql', help='MSSQL database operations')
    mssql_parser.add_argument('--server', required=True, help='MSSQL server')
    mssql_parser.add_argument('--database', required=True, help='Database name')
    mssql_parser.add_argument('--username', help='Username')
    mssql_parser.add_argument('--password', help='Password')
    mssql_parser.add_argument('--query', help='SQL query to execute')
    
    # Image converter command
    image_parser = subparsers.add_parser('image', help='Image conversion operations')
    image_parser.add_argument('--input', required=True, help='Input image path')
    image_parser.add_argument('--output', required=True, help='Output image path')
    image_parser.add_argument('--format', help='Output format (jpg, png, etc.)')
    
    # Regression command
    regression_parser = subparsers.add_parser('regression', help='Regression model operations')
    regression_subparsers = regression_parser.add_subparsers(dest='regression_command', help='Regression subcommands')
    
    # Linear regression subcommand
    linear_parser = regression_subparsers.add_parser('linear', help='Linear regression models')
    linear_parser.add_argument('--data', required=True, help='Path to CSV data file')
    linear_parser.add_argument('--model', choices=['linear', 'ridge', 'lasso', 'elastic'], default='linear', help='Linear model type')
    linear_parser.add_argument('--alpha', type=float, default=1.0, help='Regularization strength (for ridge/lasso/elastic)')
    linear_parser.add_argument('--l1-ratio', type=float, default=0.5, help='L1 ratio for elastic net')
    linear_parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (0-1)')
    
    # Tree regression subcommand
    tree_parser = regression_subparsers.add_parser('tree', help='Tree-based regression models')
    tree_parser.add_argument('--data', required=True, help='Path to CSV data file')
    tree_parser.add_argument('--model', choices=['decision', 'random_forest', 'gradient_boost'], default='random_forest', help='Tree model type')
    tree_parser.add_argument('--n-estimators', type=int, default=100, help='Number of estimators')
    tree_parser.add_argument('--max-depth', type=int, help='Maximum tree depth')
    tree_parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (0-1)')
    
    # SVM regression subcommand
    svm_parser = regression_subparsers.add_parser('svm', help='SVM regression models')
    svm_parser.add_argument('--data', required=True, help='Path to CSV data file')
    svm_parser.add_argument('--model', choices=['svr', 'linear_svr'], default='svr', help='SVM model type')
    svm_parser.add_argument('--kernel', default='rbf', help='Kernel type for SVR')
    svm_parser.add_argument('--c', type=float, default=1.0, help='Regularization parameter')
    svm_parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (0-1)')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show PyMin version')
    
    # Help command
    help_parser = subparsers.add_parser('help', help='Show detailed help')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'openrouter':
            from api.openrouter.openrouter_client import OpenRouterClient
            client = OpenRouterClient(api_key=args.api_key, model=args.model)
            response = client.simple_chat(args.message)
            print(response)
            
        elif args.command == 'db':
            if args.db_command == 'mssql':
                from db.mssql.mssql import MSSQLClient
                client = MSSQLClient(
                    server=args.server,
                    database=args.database,
                    username=args.username,
                    password=args.password
                )
                if args.query:
                    result = client.execute_query(args.query)
                    print(result)
                else:
                    print("No query provided. Use --query to execute SQL.")
                    
        elif args.command == 'image':
            from util.image_converter import ImageConverter
            converter = ImageConverter()
            converter.convert_image(args.input, args.output, args.format)
            print(f"Image converted: {args.input} -> {args.output}")
            
        elif args.command == 'regression':
            import pandas as pd
            from sklearn.model_selection import train_test_split
            
            # Load data
            df = pd.read_csv(args.data)
            if 'y' not in df.columns:
                print("Error: Data must contain a column named 'y' for the target variable")
                return
            
            # Split data
            train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)
            
            if args.regression_command == 'linear':
                from regression.linear_models import (
                    LinearRegressionWrapper, RidgeRegressionWrapper, 
                    LassoRegressionWrapper, ElasticNetRegressionWrapper
                )
                
                if args.model == 'linear':
                    model = LinearRegressionWrapper()
                elif args.model == 'ridge':
                    model = RidgeRegressionWrapper(alpha=args.alpha)
                elif args.model == 'lasso':
                    model = LassoRegressionWrapper(alpha=args.alpha)
                elif args.model == 'elastic':
                    model = ElasticNetRegressionWrapper(alpha=args.alpha, l1_ratio=args.l1_ratio)
                
                model.fit(train_df)
                score = model.score(test_df)
                print(f"{args.model.title()} Regression R² Score: {score:.4f}")
                
            elif args.regression_command == 'tree':
                from regression.tree_models import (
                    DecisionTreeRegressorWrapper, RandomForestRegressorWrapper, 
                    GradientBoostingRegressorWrapper
                )
                
                model_kwargs = {'random_state': 42}
                if args.max_depth:
                    model_kwargs['max_depth'] = args.max_depth
                
                if args.model == 'decision':
                    model = DecisionTreeRegressorWrapper(**model_kwargs)
                elif args.model == 'random_forest':
                    model = RandomForestRegressorWrapper(n_estimators=args.n_estimators, **model_kwargs)
                elif args.model == 'gradient_boost':
                    model = GradientBoostingRegressorWrapper(n_estimators=args.n_estimators, **model_kwargs)
                
                model.fit(train_df)
                score = model.score(test_df)
                print(f"{args.model.replace('_', ' ').title()} R² Score: {score:.4f}")
                
                # Show feature importance if available
                importance = model.get_feature_importance()
                if importance is not None:
                    print("\nTop 5 most important features:")
                    print(importance.head(5))
                    
            elif args.regression_command == 'svm':
                from regression.svm_models import SVRWrapper, LinearSVRWrapper
                
                if args.model == 'svr':
                    model = SVRWrapper(kernel=args.kernel, C=args.c)
                elif args.model == 'linear_svr':
                    model = LinearSVRWrapper(C=args.c)
                
                model.fit(train_df)
                score = model.score(test_df)
                print(f"{args.model.upper()} R² Score: {score:.4f}")
            
        elif args.command == 'version':
            print("PyMin v1.0.0")
            
        elif args.command == 'help':
            parser.print_help()
            print("\nExamples:")
            print("  pymin openrouter --message 'Hello, world!'")
            print("  pymin db mssql --server localhost --database test --query 'SELECT * FROM users'")
            print("  pymin image --input photo.jpg --output photo.png")
            print("  pymin regression linear --data data.csv --model ridge --alpha 0.1")
            print("  pymin regression tree --data data.csv --model random_forest --n-estimators 200")
            print("  pymin regression svm --data data.csv --model svr --kernel rbf --c 1.0")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
