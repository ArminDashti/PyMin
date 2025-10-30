"""
Prophet Analysis - Advanced analysis and diagnostic methods for Prophet models

This module provides comprehensive analysis capabilities for Prophet models including
model diagnostics, component analysis, and advanced statistical methods.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import warnings
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from prophet.plot import plot_plotly, plot_components_plotly
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Install with: pip install prophet")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Install with: pip install matplotlib")


class ProphetAnalysis:
    """
    Advanced analysis and diagnostic methods for Prophet models.
    
    This class provides methods for:
    - Model diagnostics and validation
    - Component analysis and interpretation
    - Statistical testing and significance
    - Residual analysis and model checking
    - Performance evaluation and comparison
    - Trend and seasonality analysis
    """
    
    def __init__(self, model: Optional[Prophet] = None, forecast: Optional[pd.DataFrame] = None):
        """
        Initialize ProphetAnalysis with a fitted model and forecast.
        
        Args:
            model (Prophet): Fitted Prophet model
            forecast (pd.DataFrame): Forecast dataframe from the model
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required but not installed. Install with: pip install prophet")
        
        self.model = model
        self.forecast = forecast
        self.training_data = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def set_model(self, model: Prophet, forecast: pd.DataFrame, training_data: pd.DataFrame) -> None:
        """
        Set the model and associated data for analysis.
        
        Args:
            model (Prophet): Fitted Prophet model
            forecast (pd.DataFrame): Forecast dataframe
            training_data (pd.DataFrame): Training data used to fit the model
        """
        self.model = model
        self.forecast = forecast
        self.training_data = training_data
    
    def analyze_components(self) -> Dict[str, Any]:
        """
        Analyze the components of the Prophet model.
        
        Returns:
            Dict[str, Any]: Analysis of trend, seasonality, and holiday components
        """
        if self.model is None or self.forecast is None:
            raise ValueError("Model and forecast must be set before analysis")
        
        analysis = {}
        
        # Trend analysis
        trend_data = self.forecast[['ds', 'trend']].copy()
        trend_data['trend_change'] = trend_data['trend'].diff()
        trend_data['trend_change_pct'] = trend_data['trend_change'] / trend_data['trend'].shift(1) * 100
        
        analysis['trend'] = {
            'start_value': trend_data['trend'].iloc[0],
            'end_value': trend_data['trend'].iloc[-1],
            'total_change': trend_data['trend'].iloc[-1] - trend_data['trend'].iloc[0],
            'total_change_pct': ((trend_data['trend'].iloc[-1] / trend_data['trend'].iloc[0]) - 1) * 100,
            'avg_change_per_period': trend_data['trend_change'].mean(),
            'trend_volatility': trend_data['trend_change'].std(),
            'max_increase': trend_data['trend_change'].max(),
            'max_decrease': trend_data['trend_change'].min()
        }
        
        # Seasonality analysis
        seasonal_components = []
        for col in self.forecast.columns:
            if col.startswith('seasonal_') or col in ['weekly', 'yearly', 'daily']:
                component_data = self.forecast[['ds', col]].copy()
                component_analysis = {
                    'component': col,
                    'amplitude': component_data[col].max() - component_data[col].min(),
                    'mean': component_data[col].mean(),
                    'std': component_data[col].std(),
                    'range': [component_data[col].min(), component_data[col].max()]
                }
                seasonal_components.append(component_analysis)
        
        analysis['seasonality'] = seasonal_components
        
        # Holiday analysis
        holiday_components = []
        for col in self.forecast.columns:
            if col.startswith('holidays_'):
                holiday_data = self.forecast[['ds', col]].copy()
                holiday_analysis = {
                    'holiday': col,
                    'max_effect': holiday_data[col].max(),
                    'min_effect': holiday_data[col].min(),
                    'avg_effect': holiday_data[col].mean(),
                    'effect_days': (holiday_data[col] != 0).sum()
                }
                holiday_components.append(holiday_analysis)
        
        analysis['holidays'] = holiday_components
        
        return analysis
    
    def analyze_residuals(self) -> Dict[str, Any]:
        """
        Analyze the residuals of the Prophet model.
        
        Returns:
            Dict[str, Any]: Residual analysis including statistics and tests
        """
        if self.model is None or self.forecast is None or self.training_data is None:
            raise ValueError("Model, forecast, and training data must be set before analysis")
        
        # Calculate residuals for training period
        training_forecast = self.forecast[self.forecast['ds'].isin(self.training_data['ds'])]
        residuals = self.training_data['y'] - training_forecast['yhat']
        
        # Remove NaN values
        residuals = residuals.dropna()
        
        analysis = {
            'residual_stats': {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'min': residuals.min(),
                'max': residuals.max(),
                'skewness': residuals.skew(),
                'kurtosis': residuals.kurtosis()
            },
            'normality_tests': {
                'shapiro_stat': stats.shapiro(residuals)[0] if len(residuals) <= 5000 else None,
                'shapiro_pvalue': stats.shapiro(residuals)[1] if len(residuals) <= 5000 else None,
                'jarque_bera_stat': stats.jarque_bera(residuals)[0],
                'jarque_bera_pvalue': stats.jarque_bera(residuals)[1],
                'normaltest_stat': stats.normaltest(residuals)[0],
                'normaltest_pvalue': stats.normaltest(residuals)[1]
            },
            'autocorrelation': {
                'ljung_box_stat': stats.acorr_ljungbox(residuals, lags=10, return_df=True)['lb_stat'].iloc[-1],
                'ljung_box_pvalue': stats.acorr_ljungbox(residuals, lags=10, return_df=True)['lb_pvalue'].iloc[-1]
            }
        }
        
        return analysis
    
    def calculate_accuracy_metrics(self, 
                                 actual: Optional[pd.Series] = None,
                                 predicted: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate accuracy metrics for the model.
        
        Args:
            actual (pd.Series): Actual values (if None, uses training data)
            predicted (pd.Series): Predicted values (if None, uses forecast)
            
        Returns:
            Dict[str, float]: Dictionary of accuracy metrics
        """
        if actual is None or predicted is None:
            if self.training_data is None or self.forecast is None:
                raise ValueError("Training data and forecast must be available")
            
            # Use training period for evaluation
            training_forecast = self.forecast[self.forecast['ds'].isin(self.training_data['ds'])]
            actual = self.training_data['y']
            predicted = training_forecast['yhat']
        
        # Remove NaN values
        mask = ~(actual.isna() | predicted.isna())
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return {'mae': np.nan, 'mse': np.nan, 'rmse': np.nan, 'mape': np.nan, 'r2': np.nan}
        
        mae = mean_absolute_error(actual_clean, predicted_clean)
        mse = mean_squared_error(actual_clean, predicted_clean)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
        r2 = r2_score(actual_clean, predicted_clean)
        
        # Additional metrics
        mase = self._calculate_mase(actual_clean, predicted_clean)
        smape = self._calculate_smape(actual_clean, predicted_clean)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'mase': mase,
            'smape': smape
        }
    
    def _calculate_mase(self, actual: pd.Series, predicted: pd.Series) -> float:
        """Calculate Mean Absolute Scaled Error."""
        naive_forecast = actual.shift(1)
        mae_naive = mean_absolute_error(actual[1:], naive_forecast[1:])
        mae_model = mean_absolute_error(actual, predicted)
        return mae_model / mae_naive if mae_naive != 0 else np.nan
    
    def _calculate_smape(self, actual: pd.Series, predicted: pd.Series) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        return np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100
    
    def analyze_forecast_uncertainty(self) -> Dict[str, Any]:
        """
        Analyze the uncertainty in the forecast.
        
        Returns:
            Dict[str, Any]: Uncertainty analysis including confidence intervals and volatility
        """
        if self.forecast is None:
            raise ValueError("Forecast must be available for uncertainty analysis")
        
        # Calculate uncertainty metrics
        forecast_period = self.forecast[~self.forecast['ds'].isin(self.training_data['ds'])] if self.training_data is not None else self.forecast
        
        uncertainty_width = forecast_period['yhat_upper'] - forecast_period['yhat_lower']
        relative_uncertainty = uncertainty_width / forecast_period['yhat']
        
        analysis = {
            'uncertainty_stats': {
                'mean_width': uncertainty_width.mean(),
                'std_width': uncertainty_width.std(),
                'min_width': uncertainty_width.min(),
                'max_width': uncertainty_width.max(),
                'mean_relative_uncertainty': relative_uncertainty.mean(),
                'std_relative_uncertainty': relative_uncertainty.std()
            },
            'uncertainty_trend': {
                'increasing': (uncertainty_width.diff() > 0).sum(),
                'decreasing': (uncertainty_width.diff() < 0).sum(),
                'stable': (uncertainty_width.diff() == 0).sum()
            }
        }
        
        return analysis
    
    def detect_changepoints(self, 
                           changepoint_range: float = 0.8,
                           changepoint_prior_scale: float = 0.05) -> Dict[str, Any]:
        """
        Detect changepoints in the time series.
        
        Args:
            changepoint_range (float): Proportion of history to consider for changepoints
            changepoint_prior_scale (float): Flexibility of changepoint detection
            
        Returns:
            Dict[str, Any]: Changepoint analysis
        """
        if self.training_data is None:
            raise ValueError("Training data must be available for changepoint detection")
        
        # Create a new model with changepoint detection
        model_cp = Prophet(
            changepoint_range=changepoint_range,
            changepoint_prior_scale=changepoint_prior_scale
        )
        model_cp.fit(self.training_data)
        
        # Get changepoints
        changepoints = model_cp.changepoints
        changepoint_scores = model_cp.params['delta'].mean(0)
        
        # Analyze changepoints
        analysis = {
            'num_changepoints': len(changepoints),
            'changepoint_dates': changepoints.tolist(),
            'changepoint_scores': changepoint_scores.tolist(),
            'significant_changepoints': changepoints[np.abs(changepoint_scores) > 0.01].tolist(),
            'max_changepoint_score': np.max(np.abs(changepoint_scores)),
            'avg_changepoint_score': np.mean(np.abs(changepoint_scores))
        }
        
        return analysis
    
    def plot_component_analysis(self, 
                               figsize: Tuple[int, int] = (15, 10),
                               interactive: bool = True) -> Optional[go.Figure]:
        """
        Create comprehensive component analysis plots.
        
        Args:
            figsize (Tuple[int, int]): Figure size for matplotlib plots
            interactive (bool): Whether to use interactive plotly plots
            
        Returns:
            go.Figure or None: Plotly figure if interactive, None for matplotlib
        """
        if self.model is None or self.forecast is None:
            raise ValueError("Model and forecast must be available for plotting")
        
        if interactive and PLOTLY_AVAILABLE:
            # Create subplots for different components
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Trend', 'Weekly Seasonality', 'Yearly Seasonality', 
                              'Holiday Effects', 'Residuals', 'Forecast vs Actual'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Trend plot
            fig.add_trace(
                go.Scatter(x=self.forecast['ds'], y=self.forecast['trend'],
                          mode='lines', name='Trend', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Weekly seasonality
            if 'weekly' in self.forecast.columns:
                fig.add_trace(
                    go.Scatter(x=self.forecast['ds'], y=self.forecast['weekly'],
                              mode='lines', name='Weekly', line=dict(color='green')),
                    row=1, col=2
                )
            
            # Yearly seasonality
            if 'yearly' in self.forecast.columns:
                fig.add_trace(
                    go.Scatter(x=self.forecast['ds'], y=self.forecast['yearly'],
                              mode='lines', name='Yearly', line=dict(color='red')),
                    row=2, col=1
                )
            
            # Holiday effects
            holiday_cols = [col for col in self.forecast.columns if col.startswith('holidays_')]
            if holiday_cols:
                fig.add_trace(
                    go.Scatter(x=self.forecast['ds'], y=self.forecast[holiday_cols[0]],
                              mode='lines', name='Holidays', line=dict(color='orange')),
                    row=2, col=2
                )
            
            # Residuals
            if self.training_data is not None:
                training_forecast = self.forecast[self.forecast['ds'].isin(self.training_data['ds'])]
                residuals = self.training_data['y'] - training_forecast['yhat']
                fig.add_trace(
                    go.Scatter(x=training_forecast['ds'], y=residuals,
                              mode='markers', name='Residuals', marker=dict(color='purple')),
                    row=3, col=1
                )
            
            # Forecast vs Actual
            if self.training_data is not None:
                fig.add_trace(
                    go.Scatter(x=self.training_data['ds'], y=self.training_data['y'],
                              mode='markers', name='Actual', marker=dict(color='black')),
                    row=3, col=2
                )
                fig.add_trace(
                    go.Scatter(x=self.forecast['ds'], y=self.forecast['yhat'],
                              mode='lines', name='Forecast', line=dict(color='blue')),
                    row=3, col=2
                )
            
            fig.update_layout(height=800, showlegend=True, title_text="Prophet Component Analysis")
            return fig
        
        elif MATPLOTLIB_AVAILABLE:
            # Create matplotlib plots
            fig, axes = plt.subplots(3, 2, figsize=figsize)
            fig.suptitle('Prophet Component Analysis', fontsize=16)
            
            # Trend
            axes[0, 0].plot(self.forecast['ds'], self.forecast['trend'], 'b-', linewidth=2)
            axes[0, 0].set_title('Trend')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Weekly seasonality
            if 'weekly' in self.forecast.columns:
                axes[0, 1].plot(self.forecast['ds'], self.forecast['weekly'], 'g-', linewidth=2)
                axes[0, 1].set_title('Weekly Seasonality')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Yearly seasonality
            if 'yearly' in self.forecast.columns:
                axes[1, 0].plot(self.forecast['ds'], self.forecast['yearly'], 'r-', linewidth=2)
                axes[1, 0].set_title('Yearly Seasonality')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Holiday effects
            holiday_cols = [col for col in self.forecast.columns if col.startswith('holidays_')]
            if holiday_cols:
                axes[1, 1].plot(self.forecast['ds'], self.forecast[holiday_cols[0]], 'orange', linewidth=2)
                axes[1, 1].set_title('Holiday Effects')
                axes[1, 1].grid(True, alpha=0.3)
            
            # Residuals
            if self.training_data is not None:
                training_forecast = self.forecast[self.forecast['ds'].isin(self.training_data['ds'])]
                residuals = self.training_data['y'] - training_forecast['yhat']
                axes[2, 0].scatter(training_forecast['ds'], residuals, alpha=0.6, color='purple')
                axes[2, 0].set_title('Residuals')
                axes[2, 0].grid(True, alpha=0.3)
                axes[2, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # Forecast vs Actual
            if self.training_data is not None:
                axes[2, 1].scatter(self.training_data['ds'], self.training_data['y'], 
                                 alpha=0.6, color='black', label='Actual')
                axes[2, 1].plot(self.forecast['ds'], self.forecast['yhat'], 
                               'b-', linewidth=2, label='Forecast')
                axes[2, 1].set_title('Forecast vs Actual')
                axes[2, 1].legend()
                axes[2, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            return None
        
        else:
            self.logger.warning("Neither Plotly nor Matplotlib available for plotting")
            return None
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            str: Formatted analysis report
        """
        if self.model is None or self.forecast is None:
            raise ValueError("Model and forecast must be available for report generation")
        
        report = []
        report.append("=" * 60)
        report.append("PROPHET MODEL ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Model information
        report.append("MODEL INFORMATION:")
        report.append("-" * 20)
        report.append(f"Growth: {self.model.growth}")
        report.append(f"Seasonality Mode: {self.model.seasonality_mode}")
        report.append(f"Daily Seasonality: {self.model.daily_seasonality}")
        report.append(f"Weekly Seasonality: {self.model.weekly_seasonality}")
        report.append(f"Yearly Seasonality: {self.model.yearly_seasonality}")
        report.append("")
        
        # Component analysis
        components = self.analyze_components()
        report.append("COMPONENT ANALYSIS:")
        report.append("-" * 20)
        
        # Trend analysis
        trend = components['trend']
        report.append("Trend:")
        report.append(f"  Start Value: {trend['start_value']:.2f}")
        report.append(f"  End Value: {trend['end_value']:.2f}")
        report.append(f"  Total Change: {trend['total_change']:.2f} ({trend['total_change_pct']:.2f}%)")
        report.append(f"  Average Change per Period: {trend['avg_change_per_period']:.4f}")
        report.append(f"  Trend Volatility: {trend['trend_volatility']:.4f}")
        report.append("")
        
        # Seasonality analysis
        report.append("Seasonality:")
        for seasonal in components['seasonality']:
            report.append(f"  {seasonal['component']}:")
            report.append(f"    Amplitude: {seasonal['amplitude']:.2f}")
            report.append(f"    Mean: {seasonal['mean']:.4f}")
            report.append(f"    Std: {seasonal['std']:.4f}")
        report.append("")
        
        # Accuracy metrics
        metrics = self.calculate_accuracy_metrics()
        report.append("ACCURACY METRICS:")
        report.append("-" * 20)
        for metric, value in metrics.items():
            if not np.isnan(value):
                report.append(f"{metric.upper()}: {value:.4f}")
        report.append("")
        
        # Residual analysis
        residuals = self.analyze_residuals()
        report.append("RESIDUAL ANALYSIS:")
        report.append("-" * 20)
        report.append("Residual Statistics:")
        for stat, value in residuals['residual_stats'].items():
            report.append(f"  {stat}: {value:.4f}")
        report.append("")
        
        report.append("Normality Tests:")
        for test, value in residuals['normality_tests'].items():
            if value is not None:
                report.append(f"  {test}: {value:.4f}")
        report.append("")
        
        # Uncertainty analysis
        uncertainty = self.analyze_forecast_uncertainty()
        report.append("FORECAST UNCERTAINTY:")
        report.append("-" * 20)
        for stat, value in uncertainty['uncertainty_stats'].items():
            report.append(f"{stat}: {value:.4f}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
