# PyMin Time Series Analysis Module

A comprehensive time series analysis toolkit built on Facebook Prophet with advanced forecasting, analysis, and validation capabilities.

## Features

### 🚀 Core Forecasting
- **ProphetForecaster**: Complete Prophet implementation with advanced configuration
- **Flexible Growth Models**: Linear, logistic, and flat growth patterns
- **Custom Seasonality**: Add custom seasonal patterns and holiday effects
- **Uncertainty Quantification**: Built-in confidence intervals and uncertainty analysis
- **Multiple Output Formats**: Interactive Plotly plots and static Matplotlib visualizations

### 📊 Advanced Analysis
- **ProphetAnalysis**: Comprehensive model diagnostics and component analysis
- **Residual Analysis**: Statistical tests for model validation
- **Component Interpretation**: Detailed trend, seasonality, and holiday effect analysis
- **Performance Metrics**: MAE, RMSE, MAPE, MASE, SMAPE, and R²
- **Automated Reporting**: Generate detailed analysis reports

### ✅ Model Validation
- **ProphetValidation**: Cross-validation and model comparison tools
- **Time Series CV**: Proper time series cross-validation with configurable parameters
- **Hyperparameter Tuning**: Automated parameter optimization
- **Model Comparison**: Side-by-side model performance evaluation
- **Statistical Testing**: Significance tests and model diagnostics

### 🛠️ Data Utilities
- **TimeSeriesUtils**: Comprehensive data preprocessing and validation
- **Data Cleaning**: Missing value handling, outlier detection, and anomaly removal
- **Feature Engineering**: Lag features, rolling statistics, and time-based features
- **Data Validation**: Automated data quality checks and issue detection
- **Scaling and Transformation**: Multiple scaling methods and data normalization

## Installation

```bash
# Install Prophet and dependencies
pip install -r requirements.txt

# Or install individually
pip install prophet plotly matplotlib seaborn scikit-learn scipy statsmodels
```

## Quick Start

### Basic Forecasting

```python
from PyMin.timeseries import ProphetForecaster
import pandas as pd

# Create sample data
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=365, freq='D'),
    'y': np.random.randn(365).cumsum() + 100
})

# Initialize and fit Prophet model
forecaster = ProphetForecaster(
    growth='linear',
    seasonality_mode='additive',
    yearly_seasonality=True,
    weekly_seasonality=True
)

forecaster.fit(df)

# Make predictions
forecast = forecaster.predict(periods=30)

# Plot results
forecaster.plot_forecast(interactive=True)
```

### Advanced Analysis

```python
from PyMin.timeseries import ProphetAnalysis

# Analyze model components
analysis = ProphetAnalysis(forecaster.model, forecast)
analysis.set_model(forecaster.model, forecast, df)

# Get component analysis
components = analysis.analyze_components()
print("Trend analysis:", components['trend'])

# Analyze residuals
residuals = analysis.analyze_residuals()
print("Residual statistics:", residuals['residual_stats'])

# Generate comprehensive report
report = analysis.generate_report()
print(report)
```

### Cross-Validation

```python
from PyMin.timeseries import ProphetValidation

# Perform cross-validation
validation = ProphetValidation()
cv_results = validation.cross_validate_model(
    forecaster.model, 
    df,
    initial='365 days',
    period='180 days',
    horizon='30 days'
)

# Calculate metrics
metrics = validation.calculate_cv_metrics(cv_results)
print("Validation metrics:", metrics)

# Plot validation results
validation.plot_cv_results(cv_results, interactive=True)
```

### Data Preprocessing

```python
from PyMin.timeseries import TimeSeriesUtils

# Initialize utilities
utils = TimeSeriesUtils()

# Validate data
is_valid, issues = utils.validate_time_series_data(df)
print(f"Data valid: {is_valid}")

# Clean data
df_clean = utils.clean_time_series_data(
    df,
    handle_missing='interpolate',
    remove_outliers=True,
    outlier_method='iqr'
)

# Create features
df_features = utils.create_lag_features(df_clean, lags=[1, 7, 30])
df_features = utils.create_rolling_features(df_features, windows=[7, 30])
```

## API Reference

### ProphetForecaster

Main forecasting class with comprehensive Prophet functionality.

#### Key Methods:
- `fit(df)`: Fit Prophet model to training data
- `predict(periods, freq)`: Generate forecasts
- `plot_forecast()`: Visualize forecasts and components
- `add_custom_seasonality()`: Add custom seasonal patterns
- `add_regressor()`: Add external regressors
- `get_forecast_summary()`: Get forecast statistics
- `save_model()` / `load_model()`: Model persistence

#### Parameters:
- `growth`: 'linear', 'logistic', or 'flat'
- `seasonality_mode`: 'additive' or 'multiplicative'
- `changepoint_prior_scale`: Trend flexibility
- `seasonality_prior_scale`: Seasonality strength
- `holidays_prior_scale`: Holiday effect strength

### ProphetAnalysis

Advanced analysis and diagnostic capabilities.

#### Key Methods:
- `analyze_components()`: Analyze trend, seasonality, and holiday effects
- `analyze_residuals()`: Statistical residual analysis
- `calculate_accuracy_metrics()`: Performance evaluation
- `analyze_forecast_uncertainty()`: Uncertainty quantification
- `detect_changepoints()`: Changepoint detection
- `plot_component_analysis()`: Comprehensive visualization
- `generate_report()`: Automated reporting

### ProphetValidation

Cross-validation and model comparison tools.

#### Key Methods:
- `cross_validate_model()`: Time series cross-validation
- `calculate_cv_metrics()`: Validation metrics
- `compare_models()`: Model comparison
- `hyperparameter_tuning()`: Automated parameter optimization
- `plot_cv_results()`: Validation visualization
- `plot_model_comparison()`: Model comparison plots
- `generate_validation_report()`: Validation reporting

### TimeSeriesUtils

Data preprocessing and utility functions.

#### Key Methods:
- `validate_time_series_data()`: Data validation
- `clean_time_series_data()`: Data cleaning
- `resample_time_series()`: Frequency conversion
- `create_lag_features()`: Lag feature engineering
- `create_rolling_features()`: Rolling window features
- `decompose_time_series()`: Time series decomposition
- `detect_anomalies()`: Anomaly detection
- `scale_data()`: Data scaling and normalization

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic forecasting workflows
- Advanced model analysis
- Cross-validation procedures
- Data preprocessing pipelines
- Model comparison techniques

## Dependencies

- **prophet**: Facebook Prophet for time series forecasting
- **plotly**: Interactive visualizations
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualizations
- **scikit-learn**: Machine learning utilities
- **scipy**: Scientific computing
- **statsmodels**: Statistical modeling
- **pandas**: Data manipulation
- **numpy**: Numerical computing

## Performance Tips

1. **Data Size**: Prophet works well with datasets from hundreds to millions of observations
2. **Seasonality**: Enable only necessary seasonality components to improve performance
3. **Cross-Validation**: Use appropriate initial/period/horizon parameters for your data
4. **Parallel Processing**: Prophet supports parallel cross-validation for faster results
5. **Memory**: Large datasets may require chunking or sampling for initial exploration

## Troubleshooting

### Common Issues:

1. **Prophet Installation**: 
   ```bash
   # On Windows, you may need:
   conda install -c conda-forge prophet
   
   # Or use pip with specific version:
   pip install prophet==1.1.4
   ```

2. **Memory Issues**: Reduce data size or use sampling for initial analysis

3. **Convergence Issues**: Adjust `changepoint_prior_scale` or `seasonality_prior_scale`

4. **Missing Dependencies**: Install all requirements from `requirements.txt`

### Getting Help:

- Check the example usage in `example_usage.py`
- Review Prophet documentation: https://facebook.github.io/prophet/
- Examine error messages and adjust parameters accordingly

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- New features include appropriate tests
- Documentation is updated for new functionality
- Examples are provided for new features

## License

This module is part of the PyMin project. See the main project license for details.
