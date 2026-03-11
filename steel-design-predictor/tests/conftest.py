"""
Pytest configuration and shared fixtures for SteelML tests.
"""
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.config import ELEMENT_COLS, TARGET_COLS


@pytest.fixture
def sample_composition():
    """Sample steel composition for testing."""
    return {
        'C': 0.42,
        'Mn': 0.75,
        'Si': 0.30,
        'Cr': 1.00,
        'Ni': 0.0,
        'Mo': 0.20,
        'Cu': 0.0,
        'V': 0.0,
        'W': 0.0,
        'Co': 0.0,
        'Al': 0.0,
        'Ti': 0.0,
        'Nb': 0.0
    }


@pytest.fixture
def sample_compositions_df():
    """DataFrame with multiple sample compositions."""
    compositions = [
        {'C': 0.08, 'Mn': 0.45, 'Si': 0.18, 'Cr': 0.0, 'Ni': 0.0, 'Mo': 0.0,
         'Cu': 0.0, 'V': 0.0, 'W': 0.0, 'Co': 0.0, 'Al': 0.02, 'Ti': 0.0, 'Nb': 0.0},
        {'C': 0.42, 'Mn': 0.75, 'Si': 0.30, 'Cr': 1.00, 'Ni': 0.0, 'Mo': 0.20,
         'Cu': 0.0, 'V': 0.0, 'W': 0.0, 'Co': 0.0, 'Al': 0.0, 'Ti': 0.0, 'Nb': 0.0},
        {'C': 0.08, 'Mn': 2.0, 'Si': 0.75, 'Cr': 18.0, 'Ni': 8.0, 'Mo': 0.0,
         'Cu': 0.0, 'V': 0.0, 'W': 0.0, 'Co': 0.0, 'Al': 0.0, 'Ti': 0.0, 'Nb': 0.0},
    ]
    return pd.DataFrame(compositions)


@pytest.fixture
def sample_properties():
    """Sample mechanical properties for testing."""
    return {
        'yield_strength_MPa': 500.0,
        'ultimate_tensile_strength_MPa': 650.0,
        'elongation_percent': 20.0
    }


@pytest.fixture
def sample_dataset():
    """Small dataset for testing."""
    np.random.seed(42)
    n_samples = 50
    
    # Generate random compositions
    compositions = pd.DataFrame({
        elem: np.random.uniform(0, 2 if elem == 'C' else 5, n_samples)
        for elem in ELEMENT_COLS
    })
    
    # Generate correlated properties
    properties = pd.DataFrame({
        'yield_strength_MPa': 300 + compositions['C'] * 200 + compositions['Mn'] * 50 + np.random.normal(0, 20, n_samples),
        'ultimate_tensile_strength_MPa': 450 + compositions['C'] * 250 + compositions['Mn'] * 60 + np.random.normal(0, 30, n_samples),
        'elongation_percent': 25 - compositions['C'] * 5 + np.random.normal(0, 2, n_samples)
    })
    
    # Ensure UTS > YS
    properties['ultimate_tensile_strength_MPa'] = properties[['yield_strength_MPa', 'ultimate_tensile_strength_MPa']].max(axis=1) + 50
    
    return pd.concat([compositions, properties], axis=1)


@pytest.fixture
def temp_model_path(tmp_path):
    """Temporary path for saving/loading models."""
    return tmp_path / "test_model.pt"


@pytest.fixture
def mock_trained_predictor(sample_dataset):
    """Mock trained predictor for testing (not actually trained)."""
    from src.pcnn_model import SteelPropertyPredictor
    
    predictor = SteelPropertyPredictor(input_dim=13)
    # Note: Not actually trained, just initialized
    return predictor


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Session-scoped temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
