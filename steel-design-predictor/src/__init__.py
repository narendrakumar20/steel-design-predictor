"""
Steel Property Predictor - Source Package
AI-Powered Materials Discovery Platform
"""

__version__ = "1.0.0"
__author__ = "Steel Property Predictor Team"

from .utils import ELEMENT_COLS, TARGET_COLS
from .data_loader import SteelDataLoader
from .feature_engineering import SteelFeatureEngineering
from .pcnn_model import SteelPropertyPredictor, PhysicsConstrainedNN
from .inverse_design import InverseDesignEngine
from .uncertainty import UncertaintyEstimator

__all__ = [
    'ELEMENT_COLS',
    'TARGET_COLS',
    'SteelDataLoader',
    'SteelFeatureEngineering',
    'SteelPropertyPredictor',
    'PhysicsConstrainedNN',
    'InverseDesignEngine',
    'UncertaintyEstimator',
]
