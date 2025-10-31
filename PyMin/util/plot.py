import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Install with: pip install matplotlib")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray, 
                           figsize: Tuple[int, int] = (15, 5),
                           interactive: bool = False,
                           title: Optional[str] = None) -> Optional[go.Figure]:
    if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
        raise ImportError("Matplotlib or Plotly required for plotting")
    
    if interactive and PLOTLY_AVAILABLE:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Actual vs Predicted', 'Residuals', 'Residuals Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions',
                      marker=dict(color='blue', opacity=0.6)),
            row=1, col=1
        )
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Perfect', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals',
                      marker=dict(color='green', opacity=0.6)),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        fig.add_trace(
            go.Histogram(x=residuals, name='Distribution', marker_color='orange'),
            row=1, col=3
        )
        
        fig.update_xaxes(title_text="Actual", row=1, col=1)
        fig.update_yaxes(title_text="Predicted", row=1, col=1)
        fig.update_xaxes(title_text="Predicted", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        fig.update_xaxes(title_text="Residuals", row=1, col=3)
        
        if title:
            fig.update_layout(title=title, height=500)
        else:
            fig.update_layout(height=500)
        
        return fig
    
    elif MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title('Actual vs Predicted')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals Plot')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].hist(residuals, bins=30, color='orange', alpha=0.7)
        axes[2].set_xlabel('Residuals')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Residuals Distribution')
        axes[2].grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        plt.show()
        return None


def plot_classification_results(y_true: np.ndarray, y_pred: np.ndarray,
                               y_proba: Optional[np.ndarray] = None,
                               class_names: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (15, 5),
                               interactive: bool = False,
                               title: Optional[str] = None) -> Optional[go.Figure]:
    if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
        raise ImportError("Matplotlib or Plotly required for plotting")
    
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    if interactive and PLOTLY_AVAILABLE:
        n_plots = 2 if y_proba is not None else 1
        fig = make_subplots(
            rows=1, cols=n_plots,
            subplot_titles=('Confusion Matrix', 'ROC Curve') if n_plots == 2 else ('Confusion Matrix',)
        )
        
        cm = confusion_matrix(y_true, y_pred)
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
        
        fig.add_trace(
            go.Heatmap(z=cm, x=class_names, y=class_names,
                      colorscale='Blues', showscale=True),
            row=1, col=1
        )
        fig.update_xaxes(title_text="Predicted", row=1, col=1)
        fig.update_yaxes(title_text="Actual", row=1, col=1)
        
        if y_proba is not None:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, mode='lines',
                              name=f'ROC (AUC = {roc_auc:.2f})',
                              line=dict(color='blue')),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                              name='Random', line=dict(color='red', dash='dash')),
                    row=1, col=2
                )
                fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
                fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
        
        if title:
            fig.update_layout(title=title, height=500)
        else:
            fig.update_layout(height=500)
        
        return fig
    
    elif MATPLOTLIB_AVAILABLE:
        n_plots = 2 if y_proba is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        
        cm = confusion_matrix(y_true, y_pred)
        if SEABORN_AVAILABLE:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                       xticklabels=class_names, yticklabels=class_names)
        else:
            axes[0].imshow(cm, cmap='Blues')
            for i in range(len(cm)):
                for j in range(len(cm)):
                    axes[0].text(j, i, cm[i, j], ha='center', va='center')
            if class_names:
                axes[0].set_xticks(range(len(class_names)))
                axes[0].set_xticklabels(class_names)
                axes[0].set_yticks(range(len(class_names)))
                axes[0].set_yticklabels(class_names)
        
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        axes[0].set_title('Confusion Matrix')
        
        if y_proba is not None and n_plots > 1:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                axes[1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
                axes[1].plot([0, 1], [0, 1], 'r--', label='Random')
                axes[1].set_xlabel('False Positive Rate')
                axes[1].set_ylabel('True Positive Rate')
                axes[1].set_title('ROC Curve')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        plt.show()
        return None


def plot_feature_importance(feature_importance: Dict[str, float],
                           top_n: Optional[int] = None,
                           figsize: Tuple[int, int] = (10, 6),
                           interactive: bool = False,
                           title: Optional[str] = None) -> Optional[go.Figure]:
    if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
        raise ImportError("Matplotlib or Plotly required for plotting")
    
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    if top_n:
        sorted_features = sorted_features[:top_n]
    
    features = [f[0] for f in sorted_features]
    importance = [f[1] for f in sorted_features]
    
    if interactive and PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=importance, y=features, orientation='h',
                  marker=dict(color='steelblue'))
        )
        fig.update_layout(
            xaxis_title='Importance',
            yaxis_title='Features',
            title=title if title else 'Feature Importance',
            height=max(400, len(features) * 30)
        )
        return fig
    
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(features, importance, color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Features')
        ax.set_title(title if title else 'Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
        return None


def plot_time_series(df: pd.DataFrame,
                    date_col: str,
                    value_col: str,
                    figsize: Tuple[int, int] = (12, 6),
                    interactive: bool = False,
                    title: Optional[str] = None) -> Optional[go.Figure]:
    if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
        raise ImportError("Matplotlib or Plotly required for plotting")
    
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Columns '{date_col}' or '{value_col}' not found in DataFrame")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    if interactive and PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df[date_col], y=df[value_col],
                      mode='lines', name=value_col,
                      line=dict(color='blue'))
        )
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title=value_col,
            title=title if title else f'Time Series: {value_col}',
            hovermode='x unified'
        )
        return fig
    
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df[date_col], df[value_col], 'b-', linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel(value_col)
        ax.set_title(title if title else f'Time Series: {value_col}')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return None


def plot_correlation_matrix(df: pd.DataFrame,
                           figsize: Tuple[int, int] = (10, 8),
                           interactive: bool = False,
                           title: Optional[str] = None) -> Optional[go.Figure]:
    if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
        raise ImportError("Matplotlib or Plotly required for plotting")
    
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("DataFrame must contain numeric columns")
    
    corr_matrix = numeric_df.corr()
    
    if interactive and PLOTLY_AVAILABLE:
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig.update_layout(
            title=title if title else 'Correlation Matrix',
            width=800,
            height=800
        )
        return fig
    
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=figsize)
        if SEABORN_AVAILABLE:
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, ax=ax)
        else:
            im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.index)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_matrix.index)
            for i in range(len(corr_matrix.index)):
                for j in range(len(corr_matrix.columns)):
                    ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha='center', va='center', color='black')
            plt.colorbar(im, ax=ax)
        ax.set_title(title if title else 'Correlation Matrix')
        plt.tight_layout()
        plt.show()
        return None


def plot_distribution(data: Union[np.ndarray, pd.Series],
                     bins: int = 30,
                     figsize: Tuple[int, int] = (10, 6),
                     interactive: bool = False,
                     title: Optional[str] = None) -> Optional[go.Figure]:
    if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
        raise ImportError("Matplotlib or Plotly required for plotting")
    
    if isinstance(data, pd.Series):
        data = data.values
    
    if interactive and PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(x=data, nbinsx=bins, marker_color='steelblue',
                        opacity=0.7, name='Distribution')
        )
        fig.update_layout(
            xaxis_title='Value',
            yaxis_title='Frequency',
            title=title if title else 'Distribution',
            bargap=0.1
        )
        return fig
    
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(data, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(title if title else 'Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
        return None


def plot_model_comparison(results: List[Dict[str, Any]],
                         metric: str = 'r2',
                         figsize: Tuple[int, int] = (10, 6),
                         interactive: bool = False,
                         title: Optional[str] = None) -> Optional[go.Figure]:
    if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
        raise ImportError("Matplotlib or Plotly required for plotting")
    
    algorithms = [r.get('algorithm', 'Unknown') for r in results if 'status' in r and r['status'] == 'success']
    metrics = [r.get(metric, 0) for r in results if 'status' in r and r['status'] == 'success']
    
    if not algorithms:
        raise ValueError("No successful results found in results list")
    
    if interactive and PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=algorithms, y=metrics,
                  marker=dict(color='steelblue'))
        )
        fig.update_layout(
            xaxis_title='Algorithm',
            yaxis_title=metric.upper(),
            title=title if title else f'Model Comparison: {metric.upper()}',
            xaxis_tickangle=-45
        )
        return fig
    
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(algorithms, metrics, color='steelblue')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel(metric.upper())
        ax.set_title(title if title else f'Model Comparison: {metric.upper()}')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        return None

