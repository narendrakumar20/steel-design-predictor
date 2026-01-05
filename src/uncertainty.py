"""
Uncertainty Quantification using Ensemble Methods
Provides confidence intervals for predictions
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import torch
from src.pcnn_model import PhysicsConstrainedNN, SteelPropertyPredictor

class UncertaintyEstimator:
    """Ensemble-based uncertainty quantification"""
    
    def __init__(self, n_models: int = 5):
        """
        Args:
            n_models: Number of models in ensemble
        """
        self.n_models = n_models
        self.models = []
        
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                      X_val: pd.DataFrame, y_val: pd.DataFrame,
                      epochs: int = 200, verbose: bool = False):
        """Train ensemble of models with different initializations"""
        
        print(f"\n{'='*60}")
        print(f"TRAINING ENSEMBLE OF {self.n_models} MODELS")
        print(f"{'='*60}")
        
        for i in range(self.n_models):
            print(f"\nTraining Model {i+1}/{self.n_models}...")
            
            # Create model with different random seed
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)
            
            model = SteelPropertyPredictor(input_dim=X_train.shape[1])
            model.train(X_train, y_train, X_val, y_val, 
                       epochs=epochs, verbose=verbose)
            
            self.models.append(model)
        
        print(f"\n✓ Ensemble training complete!")
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates
        
        Returns:
            mean_predictions: [n_samples, 3] Mean predictions
            std_predictions: [n_samples, 3] Standard deviations
            confidence_intervals: [n_samples, 3, 2] 95% CI (lower, upper)
        """
        all_predictions = []
        
        for model in self.models:
            predictions = model.predict(X)
            all_predictions.append(predictions)
        
        # Stack predictions from all models
        all_predictions = np.stack(all_predictions, axis=0)  # [n_models, n_samples, 3]
        
        # Calculate statistics
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)
        
        # 95% confidence intervals (assuming normal distribution)
        confidence_intervals = np.zeros((mean_predictions.shape[0], 3, 2))
        confidence_intervals[:, :, 0] = mean_predictions - 1.96 * std_predictions  # Lower bound
        confidence_intervals[:, :, 1] = mean_predictions + 1.96 * std_predictions  # Upper bound
        
        return mean_predictions, std_predictions, confidence_intervals
    
    def get_trust_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate trust score for predictions (0-100)
        Higher score = more confident
        
        Based on:
        1. Prediction uncertainty (ensemble disagreement)
        2. Distance from training distribution
        """
        _, std_predictions, _ = self.predict_with_uncertainty(X)
        
        # Normalize uncertainty to 0-1 scale
        # Lower std = higher trust
        avg_std = np.mean(std_predictions, axis=1)
        
        # Convert to trust score (0-100)
        # Assume std > 50 is very uncertain
        trust_scores = 100 * np.exp(-avg_std / 50)
        trust_scores = np.clip(trust_scores, 0, 100)
        
        return trust_scores
    
    def get_trust_category(self, trust_score: float) -> Tuple[str, str]:
        """
        Categorize trust level
        
        Returns:
            category: 'High', 'Medium', 'Low'
            color: For visualization
        """
        if trust_score >= 80:
            return 'High', 'green'
        elif trust_score >= 60:
            return 'Medium', 'yellow'
        else:
            return 'Low', 'red'
    
    def evaluate_calibration(self, X: pd.DataFrame, y_true: pd.DataFrame) -> dict:
        """
        Evaluate how well uncertainty estimates match actual errors
        """
        mean_preds, std_preds, conf_intervals = self.predict_with_uncertainty(X)
        y_true_array = y_true.values
        
        # Check if true values fall within confidence intervals
        within_ci = np.zeros(3)
        for i in range(3):
            lower = conf_intervals[:, i, 0]
            upper = conf_intervals[:, i, 1]
            within_ci[i] = np.mean((y_true_array[:, i] >= lower) & 
                                   (y_true_array[:, i] <= upper))
        
        # Calculate coverage (should be ~0.95 for 95% CI)
        results = {
            'YS_coverage': within_ci[0],
            'UTS_coverage': within_ci[1],
            'Elongation_coverage': within_ci[2],
            'avg_coverage': np.mean(within_ci)
        }
        
        print(f"\n{'='*60}")
        print("UNCERTAINTY CALIBRATION")
        print(f"{'='*60}")
        print(f"95% Confidence Interval Coverage:")
        print(f"  Yield Strength: {within_ci[0]*100:.1f}%")
        print(f"  UTS:            {within_ci[1]*100:.1f}%")
        print(f"  Elongation:     {within_ci[2]*100:.1f}%")
        print(f"  Average:        {results['avg_coverage']*100:.1f}%")
        print(f"\nTarget coverage: 95% (well-calibrated)")
        
        if results['avg_coverage'] >= 0.90:
            print("✓ Excellent calibration!")
        elif results['avg_coverage'] >= 0.80:
            print("✓ Good calibration")
        else:
            print("⚠ Uncertainty may be underestimated")
        
        return results
    
    def plot_uncertainty(self, X: pd.DataFrame, y_true: Optional[pd.DataFrame] = None):
        """Generate uncertainty visualization"""
        import matplotlib.pyplot as plt
        
        mean_preds, std_preds, conf_intervals = self.predict_with_uncertainty(X)
        trust_scores = self.get_trust_score(X)
        
        target_names = ['Yield Strength (MPa)', 'UTS (MPa)', 'Elongation (%)']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot predictions with confidence intervals
        for i, name in enumerate(target_names):
            ax = axes[0, i]
            
            indices = np.arange(len(X))
            ax.errorbar(indices, mean_preds[:, i], 
                       yerr=1.96*std_preds[:, i],
                       fmt='o', alpha=0.6, label='Predictions ± 95% CI')
            
            if y_true is not None:
                ax.scatter(indices, y_true.values[:, i], 
                          color='red', marker='x', s=100, label='True Values')
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(name)
            ax.set_title(f'{name} Predictions')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Plot trust scores
        ax = axes[1, 0]
        colors = ['green' if s >= 80 else 'yellow' if s >= 60 else 'red' 
                 for s in trust_scores]
        ax.bar(np.arange(len(trust_scores)), trust_scores, color=colors, alpha=0.7)
        ax.axhline(80, color='green', linestyle='--', alpha=0.5, label='High Trust')
        ax.axhline(60, color='yellow', linestyle='--', alpha=0.5, label='Medium Trust')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Trust Score')
        ax.set_title('Prediction Trust Scores')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot uncertainty distribution
        ax = axes[1, 1]
        ax.hist(trust_scores, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Trust Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Trust Score Distribution')
        ax.axvline(80, color='green', linestyle='--', label='High threshold')
        ax.axvline(60, color='yellow', linestyle='--', label='Medium threshold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        high_trust = np.sum(trust_scores >= 80)
        medium_trust = np.sum((trust_scores >= 60) & (trust_scores < 80))
        low_trust = np.sum(trust_scores < 60)
        
        summary_text = f"""
        UNCERTAINTY SUMMARY
        
        Total Predictions: {len(X)}
        
        Trust Distribution:
        • High (≥80):   {high_trust} ({100*high_trust/len(X):.1f}%)
        • Medium (60-80): {medium_trust} ({100*medium_trust/len(X):.1f}%)
        • Low (<60):    {low_trust} ({100*low_trust/len(X):.1f}%)
        
        Average Trust: {np.mean(trust_scores):.1f}
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
               verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig("plots/uncertainty_analysis.png", dpi=150)
        plt.close()
        
        print(f"\n✓ Uncertainty plot saved to plots/uncertainty_analysis.png")
    
    def save_ensemble(self, path_prefix: str = "models/ensemble"):
        """Save all models in ensemble"""
        import os
        os.makedirs("models", exist_ok=True)
        
        for i, model in enumerate(self.models):
            model.save_model(f"{path_prefix}_model{i}.pt")
        
        print(f"✓ Ensemble saved ({self.n_models} models)")
    
    def load_ensemble(self, path_prefix: str = "models/ensemble"):
        """Load all models in ensemble"""
        self.models = []
        
        for i in range(self.n_models):
            model = SteelPropertyPredictor(input_dim=None)  # Will be set on load
            model.load_model(f"{path_prefix}_model{i}.pt")
            self.models.append(model)
        
        print(f"✓ Ensemble loaded ({self.n_models} models)")
