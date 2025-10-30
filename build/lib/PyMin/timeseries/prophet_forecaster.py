"""
Prophet Forecaster - Core forecasting functionality using Facebook Prophet

This module provides the main ProphetForecaster class with comprehensive
forecasting capabilities for time series data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
import warnings
from datetime import datetime, timedelta
import logging

try:
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly
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
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Install with: pip install matplotlib")


class ProphetForecaster:
    """
    A comprehensive Prophet-based time series forecaster with advanced features.
    
    This class provides methods for:
    - Time series forecasting with Prophet
    - Custom seasonality and holiday modeling
    - Uncertainty quantification
    - Model diagnostics and validation
    - Interactive and static plotting
    - Data preprocessing and validation
    """
    
    def __init__(self, 
                 growth: str = 'linear',
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 holidays_prior_scale: float = 10.0,
                 seasonality_mode: str = 'additive',
                 daily_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 yearly_seasonality: bool = True,
                 country_holidays: Optional[str] = None,
                 custom_holidays: Optional[pd.DataFrame] = None,
                 interval_width: float = 0.8,
                 mcmc_samples: int = 0,
                 uncertainty_samples: int = 1000,
                 stan_backend: Optional[str] = None):
        """
        Initialize Prophet forecaster with configuration parameters.
        
        Args:
            growth (str): Type of growth trend ('linear', 'logistic', 'flat')
            changepoint_prior_scale (float): Flexibility of trend changes
            seasonality_prior_scale (float): Strength of seasonality
            holidays_prior_scale (float): Strength of holiday effects
            seasonality_mode (str): 'additive' or 'multiplicative'
            daily_seasonality (bool): Whether to fit daily seasonality
            weekly_seasonality (bool): Whether to fit weekly seasonality
            yearly_seasonality (bool): Whether to fit yearly seasonality
            country_holidays (str): Country code for built-in holidays
            custom_holidays (pd.DataFrame): Custom holidays dataframe
            interval_width (float): Width of uncertainty intervals
            mcmc_samples (int): Number of MCMC samples for uncertainty
            uncertainty_samples (int): Number of samples for uncertainty estimation
            stan_backend (str): Stan backend to use
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required but not installed. Install with: pip install prophet")
        
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.seasonality_mode = seasonality_mode
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.country_holidays = country_holidays
        self.custom_holidays = custom_holidays
        self.interval_width = interval_width
        self.mcmc_samples = mcmc_samples
        self.uncertainty_samples = uncertainty_samples
        self.stan_backend = stan_backend
        
        self.model = None
        self.fitted = False
        self.forecast = None
        self.training_data = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and preprocess input data for Prophet.
        
        Args:
            df (pd.DataFrame): Input dataframe with 'ds' and 'y' columns
            
        Returns:
            pd.DataFrame: Validated and preprocessed dataframe
        """
        if df is None or df.empty:
            raise ValueError("Dataframe cannot be empty")
        
        # Check required columns
        if 'ds' not in df.columns:
            raise ValueError("Dataframe must contain 'ds' column (datetime)")
        if 'y' not in df.columns:
            raise ValueError("Dataframe must contain 'y' column (values)")
        
        # Convert ds to datetime if not already
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Sort by datetime
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Check for duplicates
        if df['ds'].duplicated().any():
            self.logger.warning("Duplicate timestamps found. Aggregating by mean.")
            df = df.groupby('ds')['y'].mean().reset_index()
        
        # Check for missing values
        if df['y'].isna().any():
            self.logger.warning("Missing values found in 'y' column. Forward filling.")
            df['y'] = df['y'].fillna(method='ffill')
        
        return df
    
    def _create_model(self) -> Prophet:
        """Create and configure Prophet model with specified parameters."""
        model = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            interval_width=self.interval_width,
            mcmc_samples=self.mcmc_samples,
            uncertainty_samples=self.uncertainty_samples,
            stan_backend=self.stan_backend
        )
        
        # Add country holidays
        if self.country_holidays:
            model.add_country_holidays(country_name=self.country_holidays)
        
        # Add custom holidays
        if self.custom_holidays is not None:
            model.add_holidays(self.custom_holidays)
        
        return model
    
    def fit(self, df: pd.DataFrame, 
            additional_regressors: Optional[List[str]] = None,
            custom_seasonalities: Optional[Dict[str, Dict[str, Any]]] = None) -> 'ProphetForecaster':
        """
        Fit Prophet model to training data.
        
        Args:
            df (pd.DataFrame): Training data with 'ds' and 'y' columns
            additional_regressors (List[str]): Additional regressor column names
            custom_seasonalities (Dict): Custom seasonality configurations
            
        Returns:
            ProphetForecaster: Self for method chaining
        """
        # Validate and preprocess data
        df = self._validate_data(df)
        self.training_data = df.copy()
        
        # Create and configure model
        self.model = self._create_model()
        
        # Add additional regressors
        if additional_regressors:
            for regressor in additional_regressors:
                if regressor in df.columns:
                    self.model.add_regressor(regressor)
                else:
                    self.logger.warning(f"Regressor '{regressor}' not found in data")
        
        # Add custom seasonalities
        if custom_seasonalities:
            for name, config in custom_seasonalities.items():
                self.model.add_seasonality(
                    name=name,
                    period=config.get('period'),
                    fourier_order=config.get('fourier_order', 3),
                    prior_scale=config.get('prior_scale', 10.0),
                    mode=config.get('mode', 'additive')
                )
        
        # Fit the model
        self.logger.info("Fitting Prophet model...")
        self.model.fit(df)
        self.fitted = True
        self.logger.info("Model fitted successfully")
        
        return self
    
    def predict(self, 
                periods: int = 30,
                freq: str = 'D',
                include_history: bool = True,
                future_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Make predictions using the fitted Prophet model.
        
        Args:
            periods (int): Number of periods to forecast
            freq (str): Frequency of predictions ('D', 'H', 'W', 'M', etc.)
            include_history (bool): Whether to include historical data
            future_data (pd.DataFrame): Custom future dataframe
            
        Returns:
            pd.DataFrame: Forecast dataframe with predictions and uncertainty intervals
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if future_data is not None:
            # Use provided future data
            future = future_data.copy()
            future['ds'] = pd.to_datetime(future['ds'])
        else:
            # Create future dataframe
            last_date = self.training_data['ds'].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=periods,
                freq=freq
            )
            future = pd.DataFrame({'ds': future_dates})
        
        # Make predictions
        self.forecast = self.model.predict(future)
        
        if include_history:
            # Combine with historical data
            historical_forecast = self.model.predict(self.training_data[['ds']])
            self.forecast = pd.concat([historical_forecast, self.forecast], ignore_index=True)
        
        return self.forecast
    
    def plot_forecast(self, 
                     figsize: Tuple[int, int] = (12, 8),
                     plot_components: bool = True,
                     interactive: bool = True) -> Optional[go.Figure]:
        """
        Plot the forecast results.
        
        Args:
            figsize (Tuple[int, int]): Figure size for matplotlib plots
            plot_components (bool): Whether to plot trend and seasonality components
            interactive (bool): Whether to use interactive plotly plots
            
        Returns:
            go.Figure or None: Plotly figure if interactive, None for matplotlib
        """
        if not self.fitted or self.forecast is None:
            raise ValueError("Model must be fitted and predictions made before plotting")
        
        if interactive and PLOTLY_AVAILABLE:
            # Interactive plotly plot
            fig = plot_plotly(self.model, self.forecast)
            
            if plot_components:
                components_fig = plot_components_plotly(self.model, self.forecast)
                return components_fig
            else:
                return fig
        
        elif MATPLOTLIB_AVAILABLE:
            # Static matplotlib plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot historical data
            ax.plot(self.training_data['ds'], self.training_data['y'], 
                   'ko', markersize=3, label='Historical')
            
            # Plot forecast
            ax.plot(self.forecast['ds'], self.forecast['yhat'], 
                   'b-', linewidth=2, label='Forecast')
            
            # Plot uncertainty intervals
            ax.fill_between(self.forecast['ds'], 
                           self.forecast['yhat_lower'], 
                           self.forecast['yhat_upper'],
                           alpha=0.3, color='blue', label='Uncertainty')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.set_title('Prophet Forecast')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            if plot_components:
                self.model.plot_components(self.forecast)
            
            return None
        
        else:
            self.logger.warning("Neither Plotly nor Matplotlib available for plotting")
            return None
    
    def get_forecast_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the forecast results.
        
        Returns:
            Dict[str, Any]: Summary statistics and metrics
        """
        if not self.fitted or self.forecast is None:
            raise ValueError("Model must be fitted and predictions made before getting summary")
        
        # Calculate forecast period (exclude historical data)
        historical_length = len(self.training_data)
        forecast_period = self.forecast.iloc[historical_length:]
        
        summary = {
            'model_type': 'Prophet',
            'growth': self.growth,
            'seasonality_mode': self.seasonality_mode,
            'training_periods': len(self.training_data),
            'forecast_periods': len(forecast_period),
            'forecast_start': forecast_period['ds'].min(),
            'forecast_end': forecast_period['ds'].max(),
            'mean_forecast': forecast_period['yhat'].mean(),
            'std_forecast': forecast_period['yhat'].std(),
            'min_forecast': forecast_period['yhat'].min(),
            'max_forecast': forecast_period['yhat'].max(),
            'uncertainty_width': (forecast_period['yhat_upper'] - forecast_period['yhat_lower']).mean()
        }
        
        return summary
    
    def add_custom_seasonality(self, 
                              name: str,
                              period: float,
                              fourier_order: int = 3,
                              prior_scale: float = 10.0,
                              mode: str = 'additive') -> 'ProphetForecaster':
        """
        Add custom seasonality to the model.
        
        Args:
            name (str): Name of the seasonality
            period (float): Period of the seasonality in days
            fourier_order (int): Number of Fourier components
            prior_scale (float): Prior scale for regularization
            mode (str): 'additive' or 'multiplicative'
            
        Returns:
            ProphetForecaster: Self for method chaining
        """
        if not self.fitted:
            self.model.add_seasonality(
                name=name,
                period=period,
                fourier_order=fourier_order,
                prior_scale=prior_scale,
                mode=mode
            )
        else:
            self.logger.warning("Cannot add seasonality after model is fitted")
        
        return self
    
    def add_regressor(self, name: str, prior_scale: float = 10.0, mode: str = 'additive') -> 'ProphetForecaster':
        """
        Add additional regressor to the model.
        
        Args:
            name (str): Name of the regressor column
            prior_scale (float): Prior scale for regularization
            mode (str): 'additive' or 'multiplicative'
            
        Returns:
            ProphetForecaster: Self for method chaining
        """
        if not self.fitted:
            self.model.add_regressor(name, prior_scale=prior_scale, mode=mode)
        else:
            self.logger.warning("Cannot add regressor after model is fitted")
        
        return self
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get the fitted model parameters.
        
        Returns:
            Dict[str, Any]: Model parameters and hyperparameters
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting parameters")
        
        params = {
            'growth': self.growth,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'holidays_prior_scale': self.holidays_prior_scale,
            'seasonality_mode': self.seasonality_mode,
            'daily_seasonality': self.daily_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'yearly_seasonality': self.yearly_seasonality,
            'interval_width': self.interval_width,
            'mcmc_samples': self.mcmc_samples,
            'uncertainty_samples': self.uncertainty_samples
        }
        
        return params
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before saving")
        
        import joblib
        model_data = {
            'model': self.model,
            'training_data': self.training_data,
            'parameters': self.get_model_parameters()
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'ProphetForecaster':
        """
        Load a fitted model from a file.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            ProphetForecaster: Self for method chaining
        """
        import joblib
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.training_data = model_data['training_data']
        self.fitted = True
        
        # Update parameters
        params = model_data['parameters']
        for key, value in params.items():
            setattr(self, key, value)
        
        self.logger.info(f"Model loaded from {filepath}")
        return self
