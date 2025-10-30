"""
PyMin - A Python toolkit for various data science and AI tasks

This package provides a comprehensive set of tools for:
- Machine Learning (classification, regression, ensemble models)
- Time Series Analysis (forecasting with Prophet and other tools)
- Database Operations (MSSQL connectivity and data operations)
- Network Utilities (DNS, IP, and WebRTC utilities)
- Image Processing (image conversion and manipulation)
- API Integration (OpenRouter API client)
"""

__version__ = "1.0.0"
__author__ = "PyMin Team"
__email__ = "pymin@example.com"

# Import main modules for easy access
from . import api
from . import classification
from . import regression
from . import timeseries
from . import network
from . import db
from . import util
from . import simple_ml

# Import simple ML functions for direct access
from .simple_ml import (
    simple_regression,
    simple_classification,
    quick_regression,
    quick_classification,
    get_available_algorithms
)

__all__ = [
    "api",
    "classification", 
    "regression",
    "timeseries",
    "network",
    "db",
    "util",
    "simple_ml",
    # Simple ML functions
    "simple_regression",
    "simple_classification", 
    "quick_regression",
    "quick_classification",
    "get_available_algorithms"
]
