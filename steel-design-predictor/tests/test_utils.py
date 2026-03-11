"""
Unit tests for utility functions.
"""
import pytest
import pandas as pd
import numpy as np

from src.utils import (
    ELEMENT_COLS,
    TARGET_COLS,
    identify_steel_family,
    get_steel_family,
    validate_composition,
    calculate_carbon_equivalent,
    format_composition_string,
    calculate_cost_estimate
)


class TestSteelFamilyIdentification:
    """Tests for steel family identification functions."""
    
    def test_identify_steel_family_carbon_steel(self):
        """Test identification of carbon steel."""
        df = pd.DataFrame([{
            'C': 0.4, 'Mn': 0.8, 'Si': 0.3, 'Cr': 0.2, 'Ni': 0.0,
            'Mo': 0.0, 'Cu': 0.0, 'V': 0.0, 'W': 0.0, 'Co': 0.0,
            'Al': 0.0, 'Ti': 0.0, 'Nb': 0.0
        }])
        families = identify_steel_family(df)
        assert families.iloc[0] == 'Carbon Steel'
    
    def test_identify_steel_family_stainless(self):
        """Test identification of stainless steel."""
        df = pd.DataFrame([{
            'C': 0.08, 'Mn': 2.0, 'Si': 0.75, 'Cr': 18.0, 'Ni': 8.0,
            'Mo': 0.0, 'Cu': 0.0, 'V': 0.0, 'W': 0.0, 'Co': 0.0,
            'Al': 0.0, 'Ti': 0.0, 'Nb': 0.0
        }])
        families = identify_steel_family(df)
        assert families.iloc[0] == 'Stainless Steel'
    
    def test_get_steel_family_single_composition(self, sample_composition):
        """Test getting steel family for single composition."""
        family = get_steel_family(sample_composition)
        assert isinstance(family, str)
        assert family in ['Carbon Steel', 'Low Alloy Steel', 'Stainless Steel',
                         'High Strength Low Alloy', 'Tool Steel', 'Unknown']


class TestCompositionValidation:
    """Tests for composition validation."""
    
    def test_validate_valid_composition(self, sample_composition):
        """Test validation of valid composition."""
        is_valid, message = validate_composition(sample_composition)
        assert is_valid is True
        assert message == "Valid composition"
    
    def test_validate_composition_exceeds_100(self):
        """Test validation fails when total exceeds 100%."""
        composition = {elem: 10.0 for elem in ELEMENT_COLS}
        is_valid, message = validate_composition(composition)
        assert is_valid is False
        assert "exceeds 100%" in message
    
    def test_validate_composition_too_low(self):
        """Test validation fails when total is too low."""
        composition = {elem: 0.1 for elem in ELEMENT_COLS}
        is_valid, message = validate_composition(composition)
        assert is_valid is False
        assert "too low" in message
    
    def test_validate_negative_value(self):
        """Test validation fails with negative values."""
        composition = sample_composition.copy()
        composition['C'] = -0.5
        is_valid, message = validate_composition(composition)
        assert is_valid is False
        assert "cannot be negative" in message
    
    def test_validate_carbon_too_high(self):
        """Test validation fails when carbon is too high."""
        composition = {elem: 0.1 for elem in ELEMENT_COLS}
        composition['C'] = 2.5
        is_valid, message = validate_composition(composition)
        assert is_valid is False
        assert "Carbon content too high" in message


class TestCarbonEquivalent:
    """Tests for carbon equivalent calculation."""
    
    def test_calculate_carbon_equivalent(self):
        """Test carbon equivalent calculation."""
        df = pd.DataFrame([{
            'C': 0.4, 'Mn': 0.8, 'Si': 0.3, 'Cr': 1.0, 'Ni': 0.0,
            'Mo': 0.2, 'Cu': 0.0, 'V': 0.0, 'W': 0.0, 'Co': 0.0,
            'Al': 0.0, 'Ti': 0.0, 'Nb': 0.0
        }])
        ce = calculate_carbon_equivalent(df)
        assert isinstance(ce, pd.Series)
        assert ce.iloc[0] > 0
        # CE = C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15
        expected = 0.4 + 0.8/6 + (1.0+0.2)/5
        assert abs(ce.iloc[0] - expected) < 0.01


class TestCompositionFormatting:
    """Tests for composition formatting."""
    
    def test_format_composition_string(self, sample_composition):
        """Test formatting composition as string."""
        formatted = format_composition_string(sample_composition)
        assert isinstance(formatted, str)
        assert 'C:' in formatted
        assert 'Mn:' in formatted
        # Should be sorted by value
        assert formatted.index('Cr:') < formatted.index('Mo:')
    
    def test_format_composition_filters_small_values(self):
        """Test that small values are filtered out."""
        composition = {elem: 0.001 for elem in ELEMENT_COLS}
        composition['C'] = 0.5
        formatted = format_composition_string(composition)
        # Only C should appear (others < 0.01)
        assert 'C:' in formatted
        assert 'Mn:' not in formatted


class TestCostEstimation:
    """Tests for cost estimation."""
    
    def test_calculate_cost_estimate(self, sample_composition):
        """Test cost estimation."""
        cost = calculate_cost_estimate(sample_composition)
        assert isinstance(cost, float)
        assert cost >= 1.0  # At least base iron cost
    
    def test_cost_increases_with_expensive_elements(self):
        """Test that cost increases with expensive alloying elements."""
        cheap_composition = {elem: 0.0 for elem in ELEMENT_COLS}
        cheap_composition['C'] = 0.5
        cheap_composition['Mn'] = 0.5
        
        expensive_composition = cheap_composition.copy()
        expensive_composition['Mo'] = 2.0  # Expensive element
        expensive_composition['W'] = 1.0   # Very expensive
        
        cheap_cost = calculate_cost_estimate(cheap_composition)
        expensive_cost = calculate_cost_estimate(expensive_composition)
        
        assert expensive_cost > cheap_cost


class TestConstants:
    """Tests for module constants."""
    
    def test_element_cols_defined(self):
        """Test that element columns are defined."""
        assert isinstance(ELEMENT_COLS, list)
        assert len(ELEMENT_COLS) == 13
        assert 'C' in ELEMENT_COLS
        assert 'Fe' not in ELEMENT_COLS  # Iron is balance
    
    def test_target_cols_defined(self):
        """Test that target columns are defined."""
        assert isinstance(TARGET_COLS, list)
        assert len(TARGET_COLS) == 3
        assert 'yield_strength_MPa' in TARGET_COLS
        assert 'ultimate_tensile_strength_MPa' in TARGET_COLS
        assert 'elongation_percent' in TARGET_COLS
