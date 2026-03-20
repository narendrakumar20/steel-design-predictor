"""
Data loading and exploratory data analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
import os

from src.utils import ELEMENT_COLS, TARGET_COLS, identify_steel_family, calculate_carbon_equivalent

class SteelDataLoader:
    """Load and analyze steel composition-property data"""
    
    def __init__(self, data_path: str = "data/steel_data.csv"):
        self.data_path = data_path
        self.df = None
        self.train_df = None
        self.test_df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load steel dataset"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}. Run generate_dataset.py first.")
        
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} steel samples")
        print(f"✓ Features: {len(ELEMENT_COLS)} elements")
        print(f"✓ Targets: {len(TARGET_COLS)} properties")
        
        return self.df
    
    def validate_physical_constraints(self) -> pd.DataFrame:
        """Validate that UTS > YS for all samples"""
        violations = self.df[self.df['ultimate_tensile_strength_MPa'] <= self.df['yield_strength_MPa']]
        
        if len(violations) > 0:
            print(f"⚠ WARNING: {len(violations)} samples violate UTS > YS constraint")
            return violations
        else:
            print("✓ All samples satisfy physical constraint UTS > YS")
            return pd.DataFrame()
    
    def exploratory_analysis(self, save_plots: bool = True) -> None:
        """Perform comprehensive EDA"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print("\n1. TARGET PROPERTIES STATISTICS:")
        print(self.df[TARGET_COLS].describe())
        
        # Steel family distribution
        self.df['steel_family'] = identify_steel_family(self.df[ELEMENT_COLS])
        print("\n2. STEEL FAMILY DISTRIBUTION:")
        print(self.df['steel_family'].value_counts())
        
        # Carbon equivalent
        self.df['carbon_equivalent'] = calculate_carbon_equivalent(self.df[ELEMENT_COLS])
        print(f"\n3. CARBON EQUIVALENT (weldability):")
        print(f"   Mean: {self.df['carbon_equivalent'].mean():.3f}")
        print(f"   Range: {self.df['carbon_equivalent'].min():.3f} - {self.df['carbon_equivalent'].max():.3f}")
        
        # Element correlations
        print("\n4. TOP ELEMENT CORRELATIONS WITH PROPERTIES:")
        for target in TARGET_COLS:
            correlations = self.df[ELEMENT_COLS].corrwith(self.df[target]).abs().sort_values(ascending=False)
            print(f"\n   {target}:")
            for element, corr in correlations.head(5).items():
                print(f"      {element}: {corr:.3f}")
        
        if save_plots:
            self._generate_plots()
    
    def _generate_plots(self) -> None:
        """Generate visualization plots"""
        os.makedirs("plots", exist_ok=True)
        
        # Plot 1: Target distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for idx, target in enumerate(TARGET_COLS):
            axes[idx].hist(self.df[target], bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(target.replace('_', ' ').title())
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {target.split("_")[0].upper()}')
        plt.tight_layout()
        plt.savefig("plots/target_distributions.png", dpi=150)
        plt.close()
        
        # Plot 2: Element correlations heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df[ELEMENT_COLS + TARGET_COLS].corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                    vmin=-1, vmax=1, square=True, linewidths=0.5)
        plt.title('Element and Property Correlations', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("plots/correlation_heatmap.png", dpi=150)
        plt.close()
        
        # Plot 3: UTS vs YS scatter (physical constraint)
        plt.figure(figsize=(8, 8))
        plt.scatter(self.df['yield_strength_MPa'], 
                   self.df['ultimate_tensile_strength_MPa'],
                   alpha=0.6, s=50, c=self.df['elongation_percent'], cmap='viridis')
        plt.plot([200, 1200], [200, 1200], 'r--', label='UTS = YS (constraint boundary)', linewidth=2)
        plt.xlabel('Yield Strength (MPa)', fontsize=12)
        plt.ylabel('Ultimate Tensile Strength (MPa)', fontsize=12)
        plt.title('Physical Constraint: UTS > YS', fontsize=14, fontweight='bold')
        plt.colorbar(label='Elongation %')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/physical_constraint.png", dpi=150)
        plt.close()
        
        # Plot 4: Steel family properties
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        families = self.df['steel_family'].unique()
        
        for idx, target in enumerate(TARGET_COLS):
            data_by_family = [self.df[self.df['steel_family'] == family][target] 
                             for family in families]
            axes[idx].boxplot(data_by_family, labels=families)
            axes[idx].set_ylabel(target.replace('_', ' ').title())
            axes[idx].set_xlabel('Steel Family')
            axes[idx].set_title(f'{target.split("_")[0].upper()} by Family')
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig("plots/family_properties.png", dpi=150)
        plt.close()
        
        print("\n✓ Plots saved to plots/ directory")
    
    def stratified_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data by steel families for better generalization"""
        from sklearn.model_selection import train_test_split
        
        # Ensure steel_family exists
        if 'steel_family' not in self.df.columns:
            self.df['steel_family'] = identify_steel_family(self.df[ELEMENT_COLS])
        
        # Stratified split
        self.train_df, self.test_df = train_test_split(
            self.df, 
            test_size=test_size, 
            stratify=self.df['steel_family'],
            random_state=random_state
        )
        
        print(f"\n✓ Train set: {len(self.train_df)} samples")
        print(f"✓ Test set: {len(self.test_df)} samples")
        print("\nFamily distribution in train/test:")
        
        train_dist = self.train_df['steel_family'].value_counts()
        test_dist = self.test_df['steel_family'].value_counts()
        
        for family in train_dist.index:
            print(f"  {family}: {train_dist[family]}/{test_dist[family]} (train/test)")
        
        return self.train_df, self.test_df
    
    def get_features_targets(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract feature and target columns"""
        if df is None:
            df = self.df
        
        X = df[ELEMENT_COLS].copy()
        y = df[TARGET_COLS].copy()
        
        return X, y
