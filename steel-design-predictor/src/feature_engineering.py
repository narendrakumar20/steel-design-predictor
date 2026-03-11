"""
Domain-informed feature engineering for steel materials
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
from src.utils import ELEMENT_COLS, calculate_carbon_equivalent

class SteelFeatureEngineering:
    """Create metallurgical features from chemical composition"""
    
    def __init__(self):
        self.feature_names = []
        self.interaction_pairs = [
            ('C', 'Mn'), ('C', 'Cr'), ('C', 'Mo'), ('C', 'Ni'),
            ('Cr', 'Ni'), ('Cr', 'Mo'), ('Ni', 'Mo'),
            ('V', 'Nb'), ('V', 'Ti'), ('Nb', 'Ti'),
            ('Si', 'Al')
        ]
        
    def create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        X_enhanced = X.copy()
        
        # 1. Two-way interactions (synergistic effects)
        for elem1, elem2 in self.interaction_pairs:
            if elem1 in X.columns and elem2 in X.columns:
                feature_name = f'{elem1}x{elem2}'
                X_enhanced[feature_name] = X[elem1] * X[elem2]
                self.feature_names.append(feature_name)
        
        # 2. Element group sums (functional groups)
        X_enhanced['total_strengthening'] = (
            X['C'] + X['Mn'] + X['Mo'] + X['V'] + X['Nb'] + X['Ti']
        )
        self.feature_names.append('total_strengthening')
        
        X_enhanced['total_hardening'] = (
            X['C'] + X['Cr'] + X['Mo'] + X['W'] + X['V']
        )
        self.feature_names.append('total_hardening')
        
        X_enhanced['total_corrosion_resistance'] = (
            X['Cr'] + X['Ni'] + X['Mo'] + X['Cu']
        )
        self.feature_names.append('total_corrosion_resistance')
        
        X_enhanced['total_grain_refiners'] = (
            X['V'] + X['Nb'] + X['Ti'] + X['Al']
        )
        self.feature_names.append('total_grain_refiners')
        
        # 3. Carbon equivalent (critical for weldability and strength)
        X_enhanced['carbon_equivalent'] = calculate_carbon_equivalent(X)
        self.feature_names.append('carbon_equivalent')
        
        # 4. Alloy intensity
        X_enhanced['alloy_content'] = X[ELEMENT_COLS].sum(axis=1)
        self.feature_names.append('alloy_content')
        
        # 5. Ratio features
        X_enhanced['Cr_Ni_ratio'] = X['Cr'] / (X['Ni'] + 0.001)  # Avoid division by zero
        X_enhanced['C_Mn_ratio'] = X['C'] / (X['Mn'] + 0.001)
        self.feature_names.extend(['Cr_Ni_ratio', 'C_Mn_ratio'])
        
        # 6. Quadratic terms for key elements (non-linear effects)
        for elem in ['C', 'Mn', 'Cr', 'Ni', 'Mo']:
            feature_name = f'{elem}_squared'
            X_enhanced[feature_name] = X[elem] ** 2
            self.feature_names.append(feature_name)
        
        # 7. Exponential terms for carbon (severe non-linearity)
        X_enhanced['C_exp'] = np.exp(X['C'])
        self.feature_names.append('C_exp')
        
        # 8. Log transforms for low-concentration elements
        for elem in ['V', 'Nb', 'Ti']:
            feature_name = f'{elem}_log'
            X_enhanced[feature_name] = np.log1p(X[elem])  # log(1 + x) to handle zeros
            self.feature_names.append(feature_name)
        
        print(f"✓ Created {len(self.feature_names)} engineered features")
        print(f"  - {len(self.interaction_pairs)} interaction terms")
        print(f"  - 4 functional group sums")
        print(f"  - Carbon equivalent")
        print(f"  - 5 quadratic terms")
        print(f"  - Additional ratio and transform features")
        
        return X_enhanced
    
    def get_feature_importance_groups(self) -> dict:
        """Return feature groups for interpretation"""
        return {
            'Base Elements': ELEMENT_COLS,
            'Interactions': [f'{e1}x{e2}' for e1, e2 in self.interaction_pairs],
            'Functional Groups': [
                'total_strengthening', 'total_hardening', 
                'total_corrosion_resistance', 'total_grain_refiners'
            ],
            'Derived Features': [
                'carbon_equivalent', 'alloy_content', 
                'Cr_Ni_ratio', 'C_Mn_ratio', 'C_exp'
            ]
        }
    
    def select_features(self, X: pd.DataFrame, y: pd.DataFrame, 
                       method: str = 'correlation', top_k: int = 50) -> List[str]:
        """Select most important features"""
        from sklearn.feature_selection import mutual_info_regression
        
        if method == 'correlation':
            # Use average absolute correlation across all targets
            correlations = []
            for target in y.columns:
                corr = X.corrwith(y[target]).abs()
                correlations.append(corr)
            
            avg_correlation = pd.concat(correlations, axis=1).mean(axis=1)
            selected = avg_correlation.nlargest(top_k).index.tolist()
            
        elif method == 'mutual_info':
            # Use mutual information (captures non-linear relationships)
            mi_scores = []
            for target in y.columns:
                mi = mutual_info_regression(X, y[target], random_state=42)
                mi_scores.append(mi)
            
            avg_mi = np.mean(mi_scores, axis=0)
            selected_idx = np.argsort(avg_mi)[-top_k:]
            selected = X.columns[selected_idx].tolist()
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"✓ Selected top {len(selected)} features using {method}")
        return selected
    
    def normalize_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Normalize features using training statistics"""
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        print(f"✓ Features normalized (mean=0, std=1)")
        
        return X_train_scaled, X_test_scaled, scaler
