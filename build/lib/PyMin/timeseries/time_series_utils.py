"""
Time Series Utilities - Helper functions and data preprocessing tools

This module provides utility functions for time series data preprocessing,
validation, and common operations used across the time series analysis modules.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
import warnings
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class TimeSeriesUtils:
    """
    Utility class for time series data preprocessing and analysis.
    
    This class provides methods for:
    - Data validation and cleaning
    - Missing value handling
    - Outlier detection and treatment
    - Data transformation and scaling
    - Time series decomposition
    - Statistical analysis
    - Data resampling and aggregation
    """
    
    def __init__(self):
        """Initialize TimeSeriesUtils."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def validate_time_series_data(df: pd.DataFrame, 
                                 datetime_col: str = 'ds',
                                 value_col: str = 'y',
                                 min_periods: int = 10) -> Tuple[bool, List[str]]:
        """
        Validate time series data for common issues.
        
        Args:
            df (pd.DataFrame): Input dataframe
            datetime_col (str): Name of datetime column
            value_col (str): Name of value column
            min_periods (int): Minimum number of periods required
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if dataframe is empty
        if df is None or df.empty:
            issues.append("Dataframe is empty")
            return False, issues
        
        # Check required columns
        if datetime_col not in df.columns:
            issues.append(f"Missing datetime column: {datetime_col}")
        if value_col not in df.columns:
            issues.append(f"Missing value column: {value_col}")
        
        if issues:
            return False, issues
        
        # Check minimum periods
        if len(df) < min_periods:
            issues.append(f"Insufficient data points: {len(df)} < {min_periods}")
        
        # Check for missing values in datetime column
        if df[datetime_col].isna().any():
            issues.append("Missing values in datetime column")
        
        # Check for missing values in value column
        missing_values = df[value_col].isna().sum()
        if missing_values > 0:
            issues.append(f"Missing values in value column: {missing_values}")
        
        # Check for duplicate timestamps
        duplicates = df[datetime_col].duplicated().sum()
        if duplicates > 0:
            issues.append(f"Duplicate timestamps: {duplicates}")
        
        # Check if datetime column is actually datetime
        try:
            pd.to_datetime(df[datetime_col])
        except:
            issues.append("Datetime column cannot be converted to datetime")
        
        # Check for constant values
        if df[value_col].nunique() == 1:
            issues.append("All values are identical (no variation)")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def clean_time_series_data(df: pd.DataFrame,
                              datetime_col: str = 'ds',
                              value_col: str = 'y',
                              handle_missing: str = 'forward_fill',
                              handle_duplicates: str = 'mean',
                              remove_outliers: bool = False,
                              outlier_method: str = 'iqr',
                              outlier_threshold: float = 1.5) -> pd.DataFrame:
        """
        Clean time series data by handling common issues.
        
        Args:
            df (pd.DataFrame): Input dataframe
            datetime_col (str): Name of datetime column
            value_col (str): Name of value column
            handle_missing (str): Method to handle missing values ('forward_fill', 'backward_fill', 'interpolate', 'drop')
            handle_duplicates (str): Method to handle duplicates ('mean', 'sum', 'first', 'last', 'drop')
            remove_outliers (bool): Whether to remove outliers
            outlier_method (str): Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
            outlier_threshold (float): Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Convert datetime column
        df_clean[datetime_col] = pd.to_datetime(df_clean[datetime_col])
        
        # Sort by datetime
        df_clean = df_clean.sort_values(datetime_col).reset_index(drop=True)
        
        # Handle duplicates
        if df_clean[datetime_col].duplicated().any():
            if handle_duplicates == 'mean':
                df_clean = df_clean.groupby(datetime_col)[value_col].mean().reset_index()
            elif handle_duplicates == 'sum':
                df_clean = df_clean.groupby(datetime_col)[value_col].sum().reset_index()
            elif handle_duplicates == 'first':
                df_clean = df_clean.drop_duplicates(subset=[datetime_col], keep='first')
            elif handle_duplicates == 'last':
                df_clean = df_clean.drop_duplicates(subset=[datetime_col], keep='last')
            elif handle_duplicates == 'drop':
                df_clean = df_clean.drop_duplicates(subset=[datetime_col])
        
        # Handle missing values
        if df_clean[value_col].isna().any():
            if handle_missing == 'forward_fill':
                df_clean[value_col] = df_clean[value_col].fillna(method='ffill')
            elif handle_missing == 'backward_fill':
                df_clean[value_col] = df_clean[value_col].fillna(method='bfill')
            elif handle_missing == 'interpolate':
                df_clean[value_col] = df_clean[value_col].interpolate()
            elif handle_missing == 'drop':
                df_clean = df_clean.dropna(subset=[value_col])
        
        # Remove outliers
        if remove_outliers:
            df_clean = TimeSeriesUtils._remove_outliers(
                df_clean, value_col, outlier_method, outlier_threshold
            )
        
        return df_clean
    
    @staticmethod
    def _remove_outliers(df: pd.DataFrame, 
                        value_col: str,
                        method: str = 'iqr',
                        threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from time series data."""
        values = df[value_col].values
        
        if method == 'iqr':
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (values >= lower_bound) & (values <= upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(values))
            mask = z_scores < threshold
        
        elif method == 'modified_zscore':
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            modified_z_scores = 0.6745 * (values - median) / mad
            mask = np.abs(modified_z_scores) < threshold
        
        else:
            raise ValueError(f"Unknown outlier method: {method}")
        
        return df[mask].reset_index(drop=True)
    
    @staticmethod
    def resample_time_series(df: pd.DataFrame,
                           datetime_col: str = 'ds',
                           value_col: str = 'y',
                           freq: str = 'D',
                           agg_method: str = 'mean') -> pd.DataFrame:
        """
        Resample time series data to a different frequency.
        
        Args:
            df (pd.DataFrame): Input dataframe
            datetime_col (str): Name of datetime column
            value_col (str): Name of value column
            freq (str): Target frequency ('D', 'W', 'M', 'Q', 'Y', etc.)
            agg_method (str): Aggregation method ('mean', 'sum', 'min', 'max', 'first', 'last')
            
        Returns:
            pd.DataFrame: Resampled dataframe
        """
        df_resampled = df.copy()
        df_resampled[datetime_col] = pd.to_datetime(df_resampled[datetime_col])
        df_resampled = df_resampled.set_index(datetime_col)
        
        # Resample
        if agg_method == 'mean':
            df_resampled = df_resampled.resample(freq).mean()
        elif agg_method == 'sum':
            df_resampled = df_resampled.resample(freq).sum()
        elif agg_method == 'min':
            df_resampled = df_resampled.resample(freq).min()
        elif agg_method == 'max':
            df_resampled = df_resampled.resample(freq).max()
        elif agg_method == 'first':
            df_resampled = df_resampled.resample(freq).first()
        elif agg_method == 'last':
            df_resampled = df_resampled.resample(freq).last()
        else:
            raise ValueError(f"Unknown aggregation method: {agg_method}")
        
        # Reset index and clean up
        df_resampled = df_resampled.reset_index()
        df_resampled = df_resampled.dropna().reset_index(drop=True)
        
        return df_resampled
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame,
                          value_col: str = 'y',
                          lags: List[int] = [1, 7, 30],
                          prefix: str = 'lag_') -> pd.DataFrame:
        """
        Create lag features for time series data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            value_col (str): Name of value column
            lags (List[int]): List of lag periods
            prefix (str): Prefix for lag column names
            
        Returns:
            pd.DataFrame: Dataframe with lag features
        """
        df_lagged = df.copy()
        
        for lag in lags:
            df_lagged[f'{prefix}{lag}'] = df_lagged[value_col].shift(lag)
        
        return df_lagged
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame,
                              value_col: str = 'y',
                              windows: List[int] = [7, 30, 90],
                              functions: List[str] = ['mean', 'std', 'min', 'max'],
                              prefix: str = 'rolling_') -> pd.DataFrame:
        """
        Create rolling window features for time series data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            value_col (str): Name of value column
            windows (List[int]): List of window sizes
            functions (List[str]): List of functions to apply ('mean', 'std', 'min', 'max', 'sum')
            prefix (str): Prefix for rolling column names
            
        Returns:
            pd.DataFrame: Dataframe with rolling features
        """
        df_rolling = df.copy()
        
        for window in windows:
            for func in functions:
                if func == 'mean':
                    df_rolling[f'{prefix}{window}_{func}'] = df_rolling[value_col].rolling(window).mean()
                elif func == 'std':
                    df_rolling[f'{prefix}{window}_{func}'] = df_rolling[value_col].rolling(window).std()
                elif func == 'min':
                    df_rolling[f'{prefix}{window}_{func}'] = df_rolling[value_col].rolling(window).min()
                elif func == 'max':
                    df_rolling[f'{prefix}{window}_{func}'] = df_rolling[value_col].rolling(window).max()
                elif func == 'sum':
                    df_rolling[f'{prefix}{window}_{func}'] = df_rolling[value_col].rolling(window).sum()
        
        return df_rolling
    
    @staticmethod
    def decompose_time_series(df: pd.DataFrame,
                            value_col: str = 'y',
                            model: str = 'additive',
                            period: Optional[int] = None) -> Dict[str, pd.Series]:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            df (pd.DataFrame): Input dataframe
            value_col (str): Name of value column
            model (str): Decomposition model ('additive' or 'multiplicative')
            period (int): Period for seasonal decomposition
            
        Returns:
            Dict[str, pd.Series]: Dictionary with trend, seasonal, and residual components
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Set datetime index
        df_decomp = df.copy()
        df_decomp['ds'] = pd.to_datetime(df_decomp['ds'])
        df_decomp = df_decomp.set_index('ds')
        
        # Perform decomposition
        decomposition = seasonal_decompose(
            df_decomp[value_col], 
            model=model, 
            period=period
        )
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'observed': decomposition.observed
        }
    
    @staticmethod
    def calculate_statistics(df: pd.DataFrame, value_col: str = 'y') -> Dict[str, float]:
        """
        Calculate comprehensive statistics for time series data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            value_col (str): Name of value column
            
        Returns:
            Dict[str, float]: Dictionary of statistical measures
        """
        values = df[value_col].dropna()
        
        stats_dict = {
            'count': len(values),
            'mean': values.mean(),
            'median': values.median(),
            'std': values.std(),
            'var': values.var(),
            'min': values.min(),
            'max': values.max(),
            'range': values.max() - values.min(),
            'skewness': values.skew(),
            'kurtosis': values.kurtosis(),
            'q25': values.quantile(0.25),
            'q75': values.quantile(0.75),
            'iqr': values.quantile(0.75) - values.quantile(0.25)
        }
        
        return stats_dict
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame,
                        value_col: str = 'y',
                        method: str = 'iqr',
                        threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect anomalies in time series data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            value_col (str): Name of value column
            method (str): Detection method ('iqr', 'zscore', 'modified_zscore', 'isolation_forest')
            threshold (float): Threshold for anomaly detection
            
        Returns:
            pd.DataFrame: Dataframe with anomaly flags
        """
        df_anomaly = df.copy()
        values = df[value_col].values
        
        if method == 'iqr':
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_anomaly['is_anomaly'] = (values < lower_bound) | (values > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(values))
            df_anomaly['is_anomaly'] = z_scores > threshold
        
        elif method == 'modified_zscore':
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            modified_z_scores = 0.6745 * (values - median) / mad
            df_anomaly['is_anomaly'] = np.abs(modified_z_scores) > threshold
        
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=threshold)
            anomaly_labels = iso_forest.fit_predict(values.reshape(-1, 1))
            df_anomaly['is_anomaly'] = anomaly_labels == -1
        
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        
        return df_anomaly
    
    @staticmethod
    def scale_data(df: pd.DataFrame,
                  value_col: str = 'y',
                  method: str = 'standard',
                  return_scaler: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Any]]:
        """
        Scale time series data using various methods.
        
        Args:
            df (pd.DataFrame): Input dataframe
            value_col (str): Name of value column
            method (str): Scaling method ('standard', 'minmax', 'robust')
            return_scaler (bool): Whether to return the scaler object
            
        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, Any]]: Scaled dataframe and optionally scaler
        """
        df_scaled = df.copy()
        values = df[value_col].values.reshape(-1, 1)
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        scaled_values = scaler.fit_transform(values)
        df_scaled[f'{value_col}_scaled'] = scaled_values.flatten()
        
        if return_scaler:
            return df_scaled, scaler
        else:
            return df_scaled
    
    @staticmethod
    def calculate_forecast_metrics(actual: np.ndarray, 
                                 predicted: np.ndarray) -> Dict[str, float]:
        """
        Calculate forecast accuracy metrics.
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            
        Returns:
            Dict[str, float]: Dictionary of accuracy metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return {'mae': np.nan, 'mse': np.nan, 'rmse': np.nan, 'mape': np.nan, 'r2': np.nan}
        
        mae = mean_absolute_error(actual_clean, predicted_clean)
        mse = mean_squared_error(actual_clean, predicted_clean)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
        r2 = r2_score(actual_clean, predicted_clean)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
