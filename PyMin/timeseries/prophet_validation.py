"""
Prophet Validation - Cross-validation and model validation methods

This module provides comprehensive validation capabilities for Prophet models including
cross-validation, performance evaluation, and model comparison methods.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
import warnings
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
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


class ProphetValidation:
    """
    Comprehensive validation methods for Prophet models.
    
    This class provides methods for:
    - Time series cross-validation
    - Performance evaluation and metrics
    - Model comparison and selection
    - Hyperparameter tuning
    - Validation plotting and visualization
    - Statistical significance testing
    """
    
    def __init__(self):
        """Initialize ProphetValidation."""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required but not installed. Install with: pip install prophet")
    
    def cross_validate_model(self, 
                           model: Prophet,
                           df: pd.DataFrame,
                           initial: str = '365 days',
                           period: str = '180 days',
                           horizon: str = '30 days',
                           parallel: str = 'processes',
                           disable_tqdm: bool = False) -> pd.DataFrame:
        """
        Perform time series cross-validation on a Prophet model.
        
        Args:
            model (Prophet): Fitted Prophet model
            df (pd.DataFrame): Training data
            initial (str): Initial training period
            period (str): Period between cutoff dates
            horizon (str): Forecast horizon
            parallel (str): Parallel processing method
            disable_tqdm (bool): Whether to disable progress bar
            
        Returns:
            pd.DataFrame: Cross-validation results
        """
        try:
            cv_results = cross_validation(
                model=model,
                df=df,
                initial=initial,
                period=period,
                horizon=horizon,
                parallel=parallel,
                disable_tqdm=disable_tqdm
            )
            return cv_results
        except Exception as e:
            raise
    
    def calculate_cv_metrics(self, cv_results: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive metrics from cross-validation results.
        
        Args:
            cv_results (pd.DataFrame): Cross-validation results from Prophet
            
        Returns:
            Dict[str, float]: Dictionary of validation metrics
        """
        # Calculate basic metrics
        mae = mean_absolute_error(cv_results['y'], cv_results['yhat'])
        mse = mean_squared_error(cv_results['y'], cv_results['yhat'])
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((cv_results['y'] - cv_results['yhat']) / cv_results['y'])) * 100
        r2 = r2_score(cv_results['y'], cv_results['yhat'])
        
        # Calculate additional metrics
        mase = self._calculate_mase(cv_results['y'], cv_results['yhat'])
        smape = self._calculate_smape(cv_results['y'], cv_results['yhat'])
        
        # Calculate coverage metrics
        coverage = self._calculate_coverage(cv_results)
        
        # Calculate horizon-specific metrics
        horizon_metrics = self._calculate_horizon_metrics(cv_results)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'mase': mase,
            'smape': smape,
            'coverage': coverage,
            **horizon_metrics
        }
        
        return metrics
    
    def _calculate_mase(self, actual: pd.Series, predicted: pd.Series) -> float:
        """Calculate Mean Absolute Scaled Error."""
        naive_forecast = actual.shift(1)
        mae_naive = mean_absolute_error(actual[1:], naive_forecast[1:])
        mae_model = mean_absolute_error(actual, predicted)
        return mae_model / mae_naive if mae_naive != 0 else np.nan
    
    def _calculate_smape(self, actual: pd.Series, predicted: pd.Series) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        return np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100
    
    def _calculate_coverage(self, cv_results: pd.DataFrame, confidence_level: float = 0.8) -> float:
        """Calculate prediction interval coverage."""
        if 'yhat_lower' not in cv_results.columns or 'yhat_upper' not in cv_results.columns:
            return np.nan
        
        in_interval = (cv_results['y'] >= cv_results['yhat_lower']) & (cv_results['y'] <= cv_results['yhat_upper'])
        return in_interval.mean()
    
    def _calculate_horizon_metrics(self, cv_results: pd.DataFrame) -> Dict[str, float]:
        """Calculate metrics for different forecast horizons."""
        if 'horizon' not in cv_results.columns:
            return {}
        
        horizon_metrics = {}
        horizons = sorted(cv_results['horizon'].unique())
        
        for horizon in horizons:
            horizon_data = cv_results[cv_results['horizon'] == horizon]
            if len(horizon_data) > 0:
                mae_h = mean_absolute_error(horizon_data['y'], horizon_data['yhat'])
                rmse_h = np.sqrt(mean_squared_error(horizon_data['y'], horizon_data['yhat']))
                mape_h = np.mean(np.abs((horizon_data['y'] - horizon_data['yhat']) / horizon_data['y'])) * 100
                
                horizon_metrics[f'mae_h{horizon.days}'] = mae_h
                horizon_metrics[f'rmse_h{horizon.days}'] = rmse_h
                horizon_metrics[f'mape_h{horizon.days}'] = mape_h
        
        return horizon_metrics
    
    def plot_cv_results(self, 
                       cv_results: pd.DataFrame,
                       figsize: Tuple[int, int] = (15, 10),
                       interactive: bool = True) -> Optional[go.Figure]:
        """
        Plot cross-validation results.
        
        Args:
            cv_results (pd.DataFrame): Cross-validation results
            figsize (Tuple[int, int]): Figure size for matplotlib plots
            interactive (bool): Whether to use interactive plotly plots
            
        Returns:
            go.Figure or None: Plotly figure if interactive, None for matplotlib
        """
        if interactive and PLOTLY_AVAILABLE:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Predictions vs Actual', 'Residuals', 
                              'Residuals by Horizon', 'Error Distribution'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Predictions vs Actual
            fig.add_trace(
                go.Scatter(x=cv_results['y'], y=cv_results['yhat'],
                          mode='markers', name='Predictions vs Actual',
                          marker=dict(color='blue', opacity=0.6)),
                row=1, col=1
            )
            
            # Add perfect prediction line
            min_val = min(cv_results['y'].min(), cv_results['yhat'].min())
            max_val = max(cv_results['y'].max(), cv_results['yhat'].max())
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                          mode='lines', name='Perfect Prediction',
                          line=dict(color='red', dash='dash')),
                row=1, col=1
            )
            
            # Residuals over time
            residuals = cv_results['y'] - cv_results['yhat']
            fig.add_trace(
                go.Scatter(x=cv_results['ds'], y=residuals,
                          mode='markers', name='Residuals',
                          marker=dict(color='green', opacity=0.6)),
                row=1, col=2
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
            
            # Residuals by horizon
            if 'horizon' in cv_results.columns:
                for horizon in sorted(cv_results['horizon'].unique()):
                    horizon_data = cv_results[cv_results['horizon'] == horizon]
                    horizon_residuals = horizon_data['y'] - horizon_data['yhat']
                    fig.add_trace(
                        go.Scatter(x=horizon_data['ds'], y=horizon_residuals,
                                  mode='markers', name=f'Horizon {horizon.days}',
                                  marker=dict(opacity=0.6)),
                        row=2, col=1
                    )
            
            # Error distribution
            fig.add_trace(
                go.Histogram(x=residuals, name='Residual Distribution',
                           marker=dict(color='purple', opacity=0.7)),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True, 
                            title_text="Cross-Validation Results")
            return fig
        
        elif MATPLOTLIB_AVAILABLE:
            # Create matplotlib plots
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle('Cross-Validation Results', fontsize=16)
            
            # Predictions vs Actual
            axes[0, 0].scatter(cv_results['y'], cv_results['yhat'], alpha=0.6, color='blue')
            min_val = min(cv_results['y'].min(), cv_results['yhat'].min())
            max_val = max(cv_results['y'].max(), cv_results['yhat'].max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
            axes[0, 0].set_xlabel('Actual')
            axes[0, 0].set_ylabel('Predicted')
            axes[0, 0].set_title('Predictions vs Actual')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Residuals over time
            residuals = cv_results['y'] - cv_results['yhat']
            axes[0, 1].scatter(cv_results['ds'], residuals, alpha=0.6, color='green')
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals Over Time')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Residuals by horizon
            if 'horizon' in cv_results.columns:
                for horizon in sorted(cv_results['horizon'].unique()):
                    horizon_data = cv_results[cv_results['horizon'] == horizon]
                    horizon_residuals = horizon_data['y'] - horizon_data['yhat']
                    axes[1, 0].scatter(horizon_data['ds'], horizon_residuals, 
                                     alpha=0.6, label=f'Horizon {horizon.days}')
                axes[1, 0].set_xlabel('Date')
                axes[1, 0].set_ylabel('Residuals')
                axes[1, 0].set_title('Residuals by Horizon')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Error distribution
            axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 1].set_xlabel('Residuals')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Residual Distribution')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            return None
    
    def compare_models(self, 
                      models: Dict[str, Prophet],
                      df: pd.DataFrame,
                      cv_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Compare multiple Prophet models using cross-validation.
        
        Args:
            models (Dict[str, Prophet]): Dictionary of model names and Prophet models
            df (pd.DataFrame): Training data
            cv_params (Dict[str, Any]): Cross-validation parameters
            
        Returns:
            pd.DataFrame: Comparison results
        """
        comparison_results = []
        
        for model_name, model in models.items():
            try:
                # Perform cross-validation
                cv_results = self.cross_validate_model(model, df, **cv_params)
                
                # Calculate metrics
                metrics = self.calculate_cv_metrics(cv_results)
                
                # Add model name
                metrics['model'] = model_name
                comparison_results.append(metrics)
                
            except Exception as e:
                continue
        
        return pd.DataFrame(comparison_results)
    
    def hyperparameter_tuning(self, 
                             df: pd.DataFrame,
                             param_grid: Dict[str, List[Any]],
                             cv_params: Dict[str, Any],
                             metric: str = 'mae') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for Prophet models.
        
        Args:
            df (pd.DataFrame): Training data
            param_grid (Dict[str, List[Any]]): Parameter grid for tuning
            cv_params (Dict[str, Any]): Cross-validation parameters
            metric (str): Metric to optimize ('mae', 'rmse', 'mape', 'r2')
            
        Returns:
            Dict[str, Any]: Best parameters and results
        """
        from itertools import product
        
        best_score = float('inf') if metric in ['mae', 'rmse', 'mape'] else float('-inf')
        best_params = None
        all_results = []
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))
            
            try:
                # Create model with current parameters
                model = Prophet(**params)
                model.fit(df)
                
                # Perform cross-validation
                cv_results = self.cross_validate_model(model, df, **cv_params)
                metrics = self.calculate_cv_metrics(cv_results)
                
                # Store results
                result = params.copy()
                result.update(metrics)
                all_results.append(result)
                
                # Check if this is the best result
                current_score = metrics[metric]
                is_better = (current_score < best_score) if metric in ['mae', 'rmse', 'mape'] else (current_score > best_score)
                
                if is_better:
                    best_score = current_score
                    best_params = params
                    
            except Exception as e:
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': pd.DataFrame(all_results)
        }
    
    def plot_model_comparison(self, 
                            comparison_results: pd.DataFrame,
                            metrics: List[str] = ['mae', 'rmse', 'mape', 'r2'],
                            figsize: Tuple[int, int] = (15, 10),
                            interactive: bool = True) -> Optional[go.Figure]:
        """
        Plot model comparison results.
        
        Args:
            comparison_results (pd.DataFrame): Model comparison results
            metrics (List[str]): Metrics to plot
            figsize (Tuple[int, int]): Figure size for matplotlib plots
            interactive (bool): Whether to use interactive plotly plots
            
        Returns:
            go.Figure or None: Plotly figure if interactive, None for matplotlib
        """
        if interactive and PLOTLY_AVAILABLE:
            # Create subplots for each metric
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=metrics,
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            for i, metric in enumerate(metrics):
                if metric in comparison_results.columns:
                    row = (i // 2) + 1
                    col = (i % 2) + 1
                    
                    fig.add_trace(
                        go.Bar(x=comparison_results['model'], 
                              y=comparison_results[metric],
                              name=metric.upper(),
                              marker_color='lightblue'),
                        row=row, col=col
                    )
            
            fig.update_layout(height=800, showlegend=False, 
                            title_text="Model Comparison")
            return fig
        
        elif MATPLOTLIB_AVAILABLE:
            # Create matplotlib plots
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle('Model Comparison', fontsize=16)
            
            for i, metric in enumerate(metrics):
                if metric in comparison_results.columns:
                    row = i // 2
                    col = i % 2
                    
                    axes[row, col].bar(comparison_results['model'], comparison_results[metric])
                    axes[row, col].set_title(metric.upper())
                    axes[row, col].set_ylabel(metric.upper())
                    axes[row, col].tick_params(axis='x', rotation=45)
                    axes[row, col].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            return None
    
    def generate_validation_report(self, 
                                 cv_results: pd.DataFrame,
                                 metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            cv_results (pd.DataFrame): Cross-validation results
            metrics (Dict[str, float]): Optional pre-calculated metrics
            
        Returns:
            str: Formatted validation report
        """
        if metrics is None:
            metrics = self.calculate_cv_metrics(cv_results)
        
        report = []
        report.append("=" * 60)
        report.append("PROPHET MODEL VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Cross-validation summary
        report.append("CROSS-VALIDATION SUMMARY:")
        report.append("-" * 30)
        report.append(f"Total predictions: {len(cv_results)}")
        report.append(f"Date range: {cv_results['ds'].min()} to {cv_results['ds'].max()}")
        if 'horizon' in cv_results.columns:
            horizons = cv_results['horizon'].unique()
            report.append(f"Forecast horizons: {[h.days for h in horizons]} days")
        report.append("")
        
        # Performance metrics
        report.append("PERFORMANCE METRICS:")
        report.append("-" * 20)
        for metric, value in metrics.items():
            if not np.isnan(value) and metric != 'model':
                report.append(f"{metric.upper()}: {value:.4f}")
        report.append("")
        
        # Horizon-specific performance
        if 'horizon' in cv_results.columns:
            report.append("HORIZON-SPECIFIC PERFORMANCE:")
            report.append("-" * 30)
            horizons = sorted(cv_results['horizon'].unique())
            for horizon in horizons:
                horizon_data = cv_results[cv_results['horizon'] == horizon]
                horizon_mae = mean_absolute_error(horizon_data['y'], horizon_data['yhat'])
                horizon_rmse = np.sqrt(mean_squared_error(horizon_data['y'], horizon_data['yhat']))
                report.append(f"Horizon {horizon.days} days:")
                report.append(f"  MAE: {horizon_mae:.4f}")
                report.append(f"  RMSE: {horizon_rmse:.4f}")
            report.append("")
        
        # Statistical significance
        report.append("STATISTICAL ANALYSIS:")
        report.append("-" * 20)
        residuals = cv_results['y'] - cv_results['yhat']
        report.append(f"Residual mean: {residuals.mean():.4f}")
        report.append(f"Residual std: {residuals.std():.4f}")
        report.append(f"Residual skewness: {residuals.skew():.4f}")
        report.append(f"Residual kurtosis: {residuals.kurtosis():.4f}")
        report.append("")
        
        # Coverage analysis
        if 'yhat_lower' in cv_results.columns and 'yhat_upper' in cv_results.columns:
            coverage = self._calculate_coverage(cv_results)
            report.append(f"Prediction interval coverage: {coverage:.4f}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
