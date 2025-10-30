"""
Example usage of PyMin Time Series Prophet methods

This module demonstrates how to use the various Prophet forecasting and analysis
capabilities provided by the PyMin time series module.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Import PyMin time series modules
from prophet_forecaster import ProphetForecaster
from time_series_utils import TimeSeriesUtils
from prophet_analysis import ProphetAnalysis
from prophet_validation import ProphetValidation


def create_sample_data(start_date: str = '2020-01-01', 
                      periods: int = 1000,
                      freq: str = 'D') -> pd.DataFrame:
    """
    Create sample time series data for demonstration.
    
    Args:
        start_date (str): Start date for the time series
        periods (int): Number of periods to generate
        freq (str): Frequency of the time series
        
    Returns:
        pd.DataFrame: Sample time series data
    """
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Generate synthetic data with trend, seasonality, and noise
    trend = np.linspace(100, 200, periods)
    weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(periods) / 7)
    yearly_seasonality = 20 * np.sin(2 * np.pi * np.arange(periods) / 365.25)
    noise = np.random.normal(0, 5, periods)
    
    # Add some holidays effect
    holiday_effect = np.zeros(periods)
    for i, date in enumerate(dates):
        if date.month == 12 and date.day == 25:  # Christmas
            holiday_effect[i] = 30
        elif date.month == 1 and date.day == 1:  # New Year
            holiday_effect[i] = 25
        elif date.month == 7 and date.day == 4:  # July 4th
            holiday_effect[i] = 15
    
    # Combine all components
    y = trend + weekly_seasonality + yearly_seasonality + holiday_effect + noise
    
    # Create dataframe
    df = pd.DataFrame({
        'ds': dates,
        'y': y
    })
    
    return df


def example_basic_forecasting():
    """Demonstrate basic Prophet forecasting functionality."""
    print("=" * 60)
    print("BASIC PROPHET FORECASTING EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    print(f"Created sample data with {len(df)} observations")
    
    # Initialize Prophet forecaster
    forecaster = ProphetForecaster(
        growth='linear',
        seasonality_mode='additive',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    
    # Fit the model
    print("\nFitting Prophet model...")
    forecaster.fit(df)
    
    # Make predictions
    print("Making predictions...")
    forecast = forecaster.predict(periods=30, freq='D')
    print(f"Generated forecast for {len(forecast)} periods")
    
    # Get forecast summary
    summary = forecaster.get_forecast_summary()
    print("\nForecast Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Plot the forecast
    print("\nGenerating forecast plot...")
    forecaster.plot_forecast(interactive=False)
    
    return forecaster, forecast


def example_advanced_analysis():
    """Demonstrate advanced Prophet analysis capabilities."""
    print("\n" + "=" * 60)
    print("ADVANCED PROPHET ANALYSIS EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    
    # Fit a Prophet model
    forecaster = ProphetForecaster()
    forecaster.fit(df)
    forecast = forecaster.predict(periods=30)
    
    # Initialize analysis
    analysis = ProphetAnalysis(forecaster.model, forecast)
    analysis.set_model(forecaster.model, forecast, df)
    
    # Analyze components
    print("Analyzing model components...")
    components = analysis.analyze_components()
    
    print("\nTrend Analysis:")
    trend = components['trend']
    for key, value in trend.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nSeasonality Analysis:")
    for seasonal in components['seasonality']:
        print(f"  {seasonal['component']}:")
        print(f"    Amplitude: {seasonal['amplitude']:.4f}")
        print(f"    Mean: {seasonal['mean']:.4f}")
    
    # Analyze residuals
    print("\nAnalyzing residuals...")
    residuals = analysis.analyze_residuals()
    
    print("Residual Statistics:")
    for stat, value in residuals['residual_stats'].items():
        print(f"  {stat}: {value:.4f}")
    
    # Calculate accuracy metrics
    print("\nCalculating accuracy metrics...")
    metrics = analysis.calculate_accuracy_metrics()
    
    print("Accuracy Metrics:")
    for metric, value in metrics.items():
        if not np.isnan(value):
            print(f"  {metric.upper()}: {value:.4f}")
    
    # Generate analysis report
    print("\nGenerating analysis report...")
    report = analysis.generate_report()
    print(report)
    
    return analysis


def example_cross_validation():
    """Demonstrate Prophet cross-validation capabilities."""
    print("\n" + "=" * 60)
    print("PROPHET CROSS-VALIDATION EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    
    # Fit a Prophet model
    forecaster = ProphetForecaster()
    forecaster.fit(df)
    
    # Initialize validation
    validation = ProphetValidation()
    
    # Perform cross-validation
    print("Performing cross-validation...")
    cv_results = validation.cross_validate_model(
        forecaster.model, 
        df,
        initial='365 days',
        period='180 days',
        horizon='30 days'
    )
    
    print(f"Cross-validation completed with {len(cv_results)} predictions")
    
    # Calculate metrics
    print("Calculating validation metrics...")
    metrics = validation.calculate_cv_metrics(cv_results)
    
    print("Validation Metrics:")
    for metric, value in metrics.items():
        if not np.isnan(value) and metric != 'model':
            print(f"  {metric.upper()}: {value:.4f}")
    
    # Plot validation results
    print("\nGenerating validation plots...")
    validation.plot_cv_results(cv_results, interactive=False)
    
    # Generate validation report
    print("\nGenerating validation report...")
    report = validation.generate_validation_report(cv_results, metrics)
    print(report)
    
    return validation, cv_results


def example_data_preprocessing():
    """Demonstrate time series data preprocessing utilities."""
    print("\n" + "=" * 60)
    print("TIME SERIES DATA PREPROCESSING EXAMPLE")
    print("=" * 60)
    
    # Create sample data with some issues
    df = create_sample_data()
    
    # Add some missing values and outliers
    df.loc[100:105, 'y'] = np.nan  # Missing values
    df.loc[200, 'y'] = 500  # Outlier
    df.loc[300, 'y'] = -50  # Another outlier
    
    print(f"Original data: {len(df)} observations")
    print(f"Missing values: {df['y'].isna().sum()}")
    
    # Initialize utilities
    utils = TimeSeriesUtils()
    
    # Validate data
    print("\nValidating data...")
    is_valid, issues = utils.validate_time_series_data(df)
    print(f"Data valid: {is_valid}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Clean data
    print("\nCleaning data...")
    df_clean = utils.clean_time_series_data(
        df,
        handle_missing='interpolate',
        remove_outliers=True,
        outlier_method='iqr',
        outlier_threshold=2.0
    )
    
    print(f"Cleaned data: {len(df_clean)} observations")
    print(f"Missing values after cleaning: {df_clean['y'].isna().sum()}")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats = utils.calculate_statistics(df_clean)
    
    print("Data Statistics:")
    for stat, value in stats.items():
        print(f"  {stat}: {value:.4f}")
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    df_anomaly = utils.detect_anomalies(df_clean, method='iqr', threshold=2.0)
    anomalies = df_anomaly['is_anomaly'].sum()
    print(f"Anomalies detected: {anomalies}")
    
    # Create lag features
    print("\nCreating lag features...")
    df_lagged = utils.create_lag_features(df_clean, lags=[1, 7, 30])
    print(f"Added lag features. New columns: {[col for col in df_lagged.columns if col.startswith('lag_')]}")
    
    # Create rolling features
    print("\nCreating rolling features...")
    df_rolling = utils.create_rolling_features(df_clean, windows=[7, 30], functions=['mean', 'std'])
    print(f"Added rolling features. New columns: {[col for col in df_rolling.columns if col.startswith('rolling_')]}")
    
    return utils, df_clean


def example_model_comparison():
    """Demonstrate model comparison capabilities."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    
    # Create different Prophet models
    models = {
        'Linear Growth': ProphetForecaster(growth='linear'),
        'Logistic Growth': ProphetForecaster(growth='logistic'),
        'Additive Seasonality': ProphetForecaster(seasonality_mode='additive'),
        'Multiplicative Seasonality': ProphetForecaster(seasonality_mode='multiplicative')
    }
    
    # Fit all models
    print("Fitting multiple models...")
    for name, model in models.items():
        print(f"  Fitting {name}...")
        model.fit(df)
    
    # Initialize validation
    validation = ProphetValidation()
    
    # Compare models
    print("\nComparing models...")
    comparison_results = validation.compare_models(
        models, 
        df,
        {
            'initial': '365 days',
            'period': '180 days',
            'horizon': '30 days'
        }
    )
    
    print("Model Comparison Results:")
    print(comparison_results[['model', 'mae', 'rmse', 'mape', 'r2']].round(4))
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    validation.plot_model_comparison(comparison_results, interactive=False)
    
    return validation, comparison_results


def main():
    """Run all examples."""
    print("PyMin Time Series Prophet Examples")
    print("=" * 60)
    
    try:
        # Basic forecasting
        forecaster, forecast = example_basic_forecasting()
        
        # Advanced analysis
        analysis = example_advanced_analysis()
        
        # Cross-validation
        validation, cv_results = example_cross_validation()
        
        # Data preprocessing
        utils, df_clean = example_data_preprocessing()
        
        # Model comparison
        comparison_validation, comparison_results = example_model_comparison()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
