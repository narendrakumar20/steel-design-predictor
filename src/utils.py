"""
Utility functions for the Steel Materials Co-Pilot
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# Element columns in dataset
ELEMENT_COLS = ['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'Cu', 'V', 'W', 'Co', 'Al', 'Ti', 'Nb']

# Target columns
TARGET_COLS = ['yield_strength_MPa', 'ultimate_tensile_strength_MPa', 'elongation_percent']

# Steel grade families based on composition
STEEL_FAMILIES = {
    'Carbon Steel': lambda df: (df['C'] > 0.25) & (df['Mn'] < 1.5) & (df['Cr'] < 0.5),
    'Low Alloy Steel': lambda df: (df['C'] < 0.3) & ((df['Cr'] + df['Ni'] + df['Mo']) < 5),
    'Stainless Steel': lambda df: (df['Cr'] > 10.5),
    'High Strength Low Alloy': lambda df: (df['C'] < 0.25) & (df['Mn'] > 1.0) & ((df['V'] + df['Nb'] + df['Ti']) > 0.05),
    'Tool Steel': lambda df: (df['C'] > 0.6) & ((df['Cr'] + df['W'] + df['Mo'] + df['V']) > 2),
}

def identify_steel_family(df: pd.DataFrame) -> pd.Series:
    """Identify steel family for each sample"""
    families = pd.Series(['Unknown'] * len(df), index=df.index)
    
    for family_name, condition in STEEL_FAMILIES.items():
        mask = condition(df)
        families[mask] = family_name
    
    return families

def validate_composition(composition: Dict[str, float]) -> Tuple[bool, str]:
    """Validate if composition is physically feasible"""
    total = sum(composition.values())
    
    if total > 100:
        return False, f"Total composition exceeds 100% ({total:.2f}%)"
    
    if total < 95:
        return False, f"Total composition too low ({total:.2f}%). Iron should make up the balance."
    
    # Check for negative values
    for element, value in composition.items():
        if value < 0:
            return False, f"{element} cannot be negative"
        if value > 100:
            return False, f"{element} cannot exceed 100%"
    
    # Check critical element limits
    if composition.get('C', 0) > 2.0:
        return False, "Carbon content too high (>2%) - this would be cast iron"
    
    return True, "Valid composition"

def calculate_carbon_equivalent(df: pd.DataFrame) -> pd.Series:
    """Calculate carbon equivalent for weldability assessment"""
    return df['C'] + df['Mn']/6 + (df['Cr'] + df['Mo'] + df['V'])/5 + (df['Ni'] + df['Cu'])/15

def get_element_groups() -> Dict[str, List[str]]:
    """Return element groups by metallurgical function"""
    return {
        'Strengthening': ['C', 'Mn', 'Mo', 'V', 'Nb', 'Ti'],
        'Hardening': ['C', 'Cr', 'Mo', 'W', 'V'],
        'Corrosion Resistance': ['Cr', 'Ni', 'Mo', 'Cu'],
        'Deoxidizers': ['Si', 'Al', 'Ti'],
        'Grain Refiners': ['V', 'Nb', 'Ti', 'Al'],
    }

def format_composition_string(composition: Dict[str, float]) -> str:
    """Format composition as readable string"""
    sorted_elements = sorted(composition.items(), key=lambda x: x[1], reverse=True)
    main_elements = [f"{elem}: {val:.3f}%" for elem, val in sorted_elements if val > 0.01]
    return ", ".join(main_elements)

def calculate_cost_estimate(composition: Dict[str, float]) -> float:
    """Estimate relative cost based on alloying element prices (2026 USD/kg)"""
    # Relative prices (normalized to Fe = 1)
    element_costs = {
        'C': 1.5, 'Mn': 2.0, 'Si': 2.5, 'Cr': 8.0, 'Ni': 20.0,
        'Mo': 60.0, 'Cu': 10.0, 'V': 80.0, 'W': 100.0, 'Co': 70.0,
        'Al': 3.0, 'Ti': 15.0, 'Nb': 90.0
    }
    
    total_cost = 1.0  # Base iron cost
    for element, percentage in composition.items():
        if element in element_costs:
            total_cost += (percentage / 100) * element_costs[element]
    
    return total_cost
