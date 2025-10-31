__version__ = "1.0.0"
__author__ = "PyMin Team"
__email__ = "pymin@example.com"

from . import api
from . import classification
from . import regression
from . import timeseries
from . import network
from . import db
from . import util
from . import simple_ml

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
    "simple_regression",
    "simple_classification", 
    "quick_regression",
    "quick_classification",
    "get_available_algorithms"
]
