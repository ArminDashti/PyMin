import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from scipy import stats


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numerical_cols:
        return pd.DataFrame()
    
    stats_dict = {}
    
    for col in numerical_cols:
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        stats_dict[col] = {
            'count': len(col_data),
            'mean': col_data.mean(),
            'std': col_data.std(),
            'harmonic_mean': stats.hmean(col_data) if (col_data > 0).all() else np.nan,
            'median': col_data.median(),
            'min': col_data.min(),
            'max': col_data.max(),
            'q25': col_data.quantile(0.25),
            'q75': col_data.quantile(0.75),
            'skewness': col_data.skew(),
            'kurtosis': col_data.kurtosis(),
            'variance': col_data.var(),
            'range': col_data.max() - col_data.min(),
            'iqr': col_data.quantile(0.75) - col_data.quantile(0.25),
            'missing_count': df[col].isna().sum(),
            'missing_pct': (df[col].isna().sum() / len(df)) * 100
        }
    
    return pd.DataFrame(stats_dict).T


def compute_cov_corr(df: pd.DataFrame) -> Dict[str, Any]:
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) < 2:
        return {
            'covariance': pd.DataFrame(),
            'correlation': pd.DataFrame(),
            'high_corr_pairs': [],
            'statistics': {}
        }
    
    df_numerical = df[numerical_cols].dropna()
    
    cov_matrix = df_numerical.cov()
    corr_matrix = df_numerical.corr()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.loc[col1, col2]
            
            if abs(corr_value) > 0.7:
                high_corr_pairs.append({
                    'column1': col1,
                    'column2': col2,
                    'correlation': corr_value
                })
    
    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    cov_stats = {
        'total_variables': len(numerical_cols),
        'covariance_mean': cov_matrix.values[np.triu_indices_from(cov_matrix.values, k=1)].mean(),
        'covariance_std': cov_matrix.values[np.triu_indices_from(cov_matrix.values, k=1)].std(),
        'correlation_mean': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
        'correlation_std': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].std(),
        'high_corr_count': len(high_corr_pairs)
    }
    
    return {
        'covariance': cov_matrix,
        'correlation': corr_matrix,
        'high_corr_pairs': high_corr_pairs,
        'statistics': cov_stats
    }

