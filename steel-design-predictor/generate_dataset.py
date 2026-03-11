"""
Generate realistic synthetic steel dataset for demonstration
Based on materials science relationships
"""
import numpy as np
import pandas as pd
from typing import Tuple

def generate_steel_dataset(n_samples: int = 400, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic steel composition-property dataset
    
    Properties based on empirical relationships from materials science:
    - YS increases with: C, Mn, Mo, V, Nb, Ti (strengthening elements)
    - UTS follows YS with offset
    - Elongation decreases with strength (trade-off)
    """
    np.random.seed(random_state)
    
    print("Generating synthetic steel dataset...")
    
    # Define steel grade families with typical compositions
    steel_grades = [
        {
            'name': 'Carbon Steel',
            'n_samples': 100,
            'composition': {
                'C': (0.30, 0.60), 'Mn': (0.60, 1.50), 'Si': (0.15, 0.40),
                'Cr': (0.0, 0.30), 'Ni': (0.0, 0.20), 'Mo': (0.0, 0.10),
                'Cu': (0.0, 0.30), 'V': (0.0, 0.05), 'W': (0.0, 0.0),
                'Co': (0.0, 0.0), 'Al': (0.01, 0.05), 'Ti': (0.0, 0.02), 'Nb': (0.0, 0.02)
            }
        },
        {
            'name': 'Low Alloy Steel',
            'n_samples': 120,
            'composition': {
                'C': (0.15, 0.30), 'Mn': (1.00, 1.80), 'Si': (0.20, 0.50),
                'Cr': (0.30, 1.50), 'Ni': (0.20, 1.00), 'Mo': (0.10, 0.50),
                'Cu': (0.10, 0.40), 'V': (0.02, 0.15), 'W': (0.0, 0.20),
                'Co': (0.0, 0.20), 'Al': (0.01, 0.08), 'Ti': (0.0, 0.05), 'Nb': (0.01, 0.08)
            }
        },
        {
            'name': 'Stainless Steel',
            'n_samples': 80,
            'composition': {
                'C': (0.03, 0.15), 'Mn': (0.50, 2.00), 'Si': (0.30, 1.00),
                'Cr': (11.0, 18.0), 'Ni': (8.0, 12.0), 'Mo': (0.0, 2.50),
                'Cu': (0.0, 0.50), 'V': (0.0, 0.10), 'W': (0.0, 0.0),
                'Co': (0.0, 0.30), 'Al': (0.0, 0.10), 'Ti': (0.0, 0.50), 'Nb': (0.0, 0.30)
            }
        },
        {
            'name': 'High Strength Low Alloy',
            'n_samples': 60,
            'composition': {
                'C': (0.05, 0.20), 'Mn': (1.20, 1.80), 'Si': (0.15, 0.50),
                'Cr': (0.20, 0.80), 'Ni': (0.20, 0.60), 'Mo': (0.05, 0.30),
                'Cu': (0.20, 0.60), 'V': (0.05, 0.15), 'W': (0.0, 0.0),
                'Co': (0.0, 0.0), 'Al': (0.02, 0.06), 'Ti': (0.01, 0.05), 'Nb': (0.03, 0.10)
            }
        },
        {
            'name': 'Tool Steel',
            'n_samples': 40,
            'composition': {
                'C': (0.70, 1.50), 'Mn': (0.20, 0.60), 'Si': (0.15, 0.40),
                'Cr': (3.00, 5.50), 'Ni': (0.0, 0.30), 'Mo': (0.50, 3.00),
                'Cu': (0.0, 0.20), 'V': (0.15, 0.50), 'W': (0.50, 2.00),
                'Co': (0.0, 5.00), 'Al': (0.0, 0.05), 'Ti': (0.0, 0.10), 'Nb': (0.0, 0.05)
            }
        }
    ]
    
    all_data = []
    
    for grade in steel_grades:
        for _ in range(grade['n_samples']):
            # Generate composition
            composition = {}
            for element, (min_val, max_val) in grade['composition'].items():
                composition[element] = np.random.uniform(min_val, max_val)
            
            # Calculate properties based on composition
            # These are simplified empirical relationships
            
            # Yield Strength (MPa): Base + contributions from each element
            ys = 200  # Base iron strength
            ys += composition['C'] * 450  # Carbon is strongest strengthener
            ys += composition['Mn'] * 50
            ys += composition['Cr'] * 30
            ys += composition['Ni'] * 20
            ys += composition['Mo'] * 120
            ys += composition['V'] * 200
            ys += composition['Nb'] * 250
            ys += composition['Ti'] * 180
            ys += composition['Si'] * 40
            
            # Add interaction effects
            ys += composition['C'] * composition['Mn'] * 80
            ys += composition['Cr'] * composition['Ni'] * 15
            
            # Add noise
            ys += np.random.normal(0, 30)
            ys = max(250, min(1200, ys))  # Physical limits
            
            # Ultimate Tensile Strength: Always > YS
            # UTS/YS ratio typically 1.1 to 1.5
            uts_ratio = 1.15 + 0.25 * (1 - composition['C'])  # Lower C = higher ratio
            uts = ys * uts_ratio
            uts += np.random.normal(0, 40)
            uts = max(ys + 50, uts)  # Ensure UTS > YS
            
            # Elongation (%): Trade-off with strength
            # Higher strength = lower elongation
            elong = 40 - (ys - 300) / 30  # Base relationship
            elong += composition['Ni'] * 2  # Nickel improves ductility
            elong -= composition['C'] * 15  # Carbon reduces ductility
            elong += composition['Mn'] * 1.5
            elong += np.random.normal(0, 2)
            elong = max(5, min(40, elong))  # Physical limits
            
            # Combine into row
            row = composition.copy()
            row['yield_strength_MPa'] = ys
            row['ultimate_tensile_strength_MPa'] = uts
            row['elongation_percent'] = elong
            row['steel_family'] = grade['name']
            
            all_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"✓ Generated {len(df)} steel samples")
    print(f"\nGrade distribution:")
    print(df['steel_family'].value_counts())
    
    print(f"\nProperty ranges:")
    print(f"  Yield Strength: {df['yield_strength_MPa'].min():.0f} - {df['yield_strength_MPa'].max():.0f} MPa")
    print(f"  UTS: {df['ultimate_tensile_strength_MPa'].min():.0f} - {df['ultimate_tensile_strength_MPa'].max():.0f} MPa")
    print(f"  Elongation: {df['elongation_percent'].min():.1f} - {df['elongation_percent'].max():.1f} %")
    
    # Verify constraint
    violations = (df['ultimate_tensile_strength_MPa'] <= df['yield_strength_MPa']).sum()
    print(f"\nConstraint check: {violations} violations (should be 0)")
    
    return df

if __name__ == "__main__":
    import os
    
    # Generate dataset
    df = generate_steel_dataset(n_samples=400)
    
    # Save to CSV
    os.makedirs("data", exist_ok=True)
    output_path = "data/steel_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Dataset saved to {output_path}")
    print(f"\nReady for analysis! Run:")
    print(f"  python train_model.py")
    print(f"  streamlit run app.py")
