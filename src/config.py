"""
Configuration management for SteelML.
Centralized configuration for paths, model parameters, and application settings.
"""
from pathlib import Path
from typing import Dict, List, Optional
import os

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File paths
DEFAULT_DATA_PATH = DATA_DIR / "steel_data.csv"
PCNN_MODEL_PATH = MODELS_DIR / "pcnn_model.pt"
FEATURE_ENGINEER_PATH = MODELS_DIR / "feature_engineer.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.pkl"

# Element columns in dataset
ELEMENT_COLS: List[str] = [
    'C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'Cu', 'V', 'W', 'Co', 'Al', 'Ti', 'Nb'
]

# Target columns
TARGET_COLS: List[str] = [
    'yield_strength_MPa',
    'ultimate_tensile_strength_MPa',
    'elongation_percent'
]

# Model hyperparameters
class ModelConfig:
    """Model training and architecture configuration."""
    
    # Neural network architecture
    HIDDEN_DIMS: List[int] = [128, 64, 32]
    DROPOUT_RATE: float = 0.3
    
    # Training parameters
    EPOCHS: int = 200
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    PATIENCE: int = 30
    
    # Physics constraint
    CONSTRAINT_WEIGHT: float = 10.0
    
    # Uncertainty estimation
    N_ENSEMBLE_MODELS: int = 5
    ENSEMBLE_EPOCHS: int = 150
    
    # Device
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


class DataConfig:
    """Data processing configuration."""
    
    # Train/test split
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.15
    RANDOM_STATE: int = 42
    
    # Feature engineering
    TOP_K_FEATURES: int = 50
    FEATURE_SELECTION_METHOD: str = "correlation"
    
    # Composition validation
    MIN_TOTAL_COMPOSITION: float = 95.0
    MAX_TOTAL_COMPOSITION: float = 100.0
    MAX_CARBON_CONTENT: float = 2.0


class InverseDesignConfig:
    """Inverse design optimization configuration."""
    
    # Genetic algorithm parameters
    POPULATION_SIZE: int = 100
    GENERATIONS: int = 50
    MUTATION_RATE: float = 0.3
    CROSSOVER_RATE: float = 0.7
    ELITE_SIZE: int = 10
    TOP_K_RESULTS: int = 5
    
    # Composition constraints
    ELEMENT_RANGES: Dict[str, tuple] = {
        'C': (0.0, 2.0),
        'Mn': (0.0, 2.5),
        'Si': (0.0, 1.5),
        'Cr': (0.0, 25.0),
        'Ni': (0.0, 20.0),
        'Mo': (0.0, 5.0),
        'Cu': (0.0, 1.0),
        'V': (0.0, 2.0),
        'W': (0.0, 10.0),
        'Co': (0.0, 5.0),
        'Al': (0.0, 0.5),
        'Ti': (0.0, 0.5),
        'Nb': (0.0, 0.5),
    }


class LoggingConfig:
    """Logging configuration."""
    
    LOG_FILE: Path = LOGS_DIR / "steelml.log"
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    # Console logging
    CONSOLE_LOG_LEVEL: str = "INFO"
    
    # File logging
    FILE_LOG_LEVEL: str = "DEBUG"
    MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT: int = 5


class AppConfig:
    """Streamlit application configuration."""
    
    PAGE_TITLE: str = "Steel Property Predictor"
    PAGE_ICON: str = "🔬"
    LAYOUT: str = "wide"
    
    # GitHub repository
    GITHUB_URL: str = "https://github.com/vamsi-op/SteelML"
    
    # Preset compositions
    PRESETS: Dict[str, Dict[str, float]] = {
        "Low Carbon Steel": {
            'C': 0.08, 'Mn': 0.45, 'Si': 0.18, 'Cr': 0.0, 'Ni': 0.0,
            'Mo': 0.0, 'Cu': 0.0, 'V': 0.0, 'W': 0.0, 'Co': 0.0,
            'Al': 0.02, 'Ti': 0.0, 'Nb': 0.0
        },
        "Medium Carbon Steel": {
            'C': 0.42, 'Mn': 0.75, 'Si': 0.30, 'Cr': 1.00, 'Ni': 0.0,
            'Mo': 0.20, 'Cu': 0.0, 'V': 0.0, 'W': 0.0, 'Co': 0.0,
            'Al': 0.0, 'Ti': 0.0, 'Nb': 0.0
        },
        "Stainless Steel 304": {
            'C': 0.08, 'Mn': 2.0, 'Si': 0.75, 'Cr': 18.0, 'Ni': 8.0,
            'Mo': 0.0, 'Cu': 0.0, 'V': 0.0, 'W': 0.0, 'Co': 0.0,
            'Al': 0.0, 'Ti': 0.0, 'Nb': 0.0
        },
        "Tool Steel": {
            'C': 0.95, 'Mn': 1.2, 'Si': 0.35, 'Cr': 5.0, 'Ni': 0.0,
            'Mo': 1.1, 'V': 1.0, 'W': 6.5, 'Co': 0.0, 'Al': 0.0,
            'Ti': 0.0, 'Nb': 0.0
        }
    }


# Export configurations
__all__ = [
    'PROJECT_ROOT',
    'DATA_DIR',
    'MODELS_DIR',
    'PLOTS_DIR',
    'LOGS_DIR',
    'ELEMENT_COLS',
    'TARGET_COLS',
    'ModelConfig',
    'DataConfig',
    'InverseDesignConfig',
    'LoggingConfig',
    'AppConfig',
]
