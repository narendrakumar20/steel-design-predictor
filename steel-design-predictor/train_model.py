"""
Complete training pipeline for Steel Materials Co-Pilot
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import SteelDataLoader
from src.feature_engineering import SteelFeatureEngineering
from src.pcnn_model import SteelPropertyPredictor
from src.uncertainty import UncertaintyEstimator

def main():
    print("="*70)
    print(" STEEL PROPERTY PREDICTOR - TRAINING PIPELINE")
    print("="*70)
    
    # 1. Load data
    print("\n[1/6] Loading dataset...")
    loader = SteelDataLoader("data/steel_data.csv")
    df = loader.load_data()
    
    # 2. Exploratory analysis
    print("\n[2/6] Performing exploratory data analysis...")
    loader.validate_physical_constraints()
    loader.exploratory_analysis(save_plots=True)
    
    # 3. Split data stratified by steel family
    print("\n[3/6] Splitting data...")
    train_df, test_df = loader.stratified_split(test_size=0.2, random_state=42)
    
    # Further split train into train/val
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(train_df, test_size=0.15, 
                                        stratify=train_df['steel_family'],
                                        random_state=42)
    
    print(f"Final split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    # 4. Feature engineering
    print("\n[4/6] Engineering features...")
    fe = SteelFeatureEngineering()
    
    X_train, y_train = loader.get_features_targets(train_df)
    X_val, y_val = loader.get_features_targets(val_df)
    X_test, y_test = loader.get_features_targets(test_df)
    
    # Create features
    X_train_enhanced = fe.create_features(X_train)
    X_val_enhanced = fe.create_features(X_val)
    X_test_enhanced = fe.create_features(X_test)
    
    # Feature selection (optional - keep top features)
    selected_features = fe.select_features(X_train_enhanced, y_train, 
                                          method='correlation', top_k=50)
    
    X_train_selected = X_train_enhanced[selected_features]
    X_val_selected = X_val_enhanced[selected_features]
    X_test_selected = X_test_enhanced[selected_features]
    
    # Normalize
    X_train_norm, X_val_norm, scaler = fe.normalize_features(X_train_selected, X_val_selected)
    X_test_norm = pd.DataFrame(
        scaler.transform(X_test_selected),
        columns=X_test_selected.columns,
        index=X_test_selected.index
    )
    
    # 5. Train Physics-Constrained Neural Network
    print("\n[5/6] Training Physics-Constrained Neural Network...")
    predictor = SteelPropertyPredictor(input_dim=X_train_norm.shape[1])
    predictor.train(X_train_norm, y_train, X_val_norm, y_val, 
                   epochs=200, batch_size=32, lr=0.001, patience=30, verbose=True)
    
    # Evaluate
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    test_metrics = predictor.evaluate(X_test_norm, y_test)
    
    # Save model
    predictor.save_model("models/pcnn_model.pt")
    
    # 6. Train Uncertainty Ensemble (optional - takes longer)
    train_ensemble = input("\nTrain uncertainty ensemble? (5 models, ~5min) [y/N]: ").lower() == 'y'
    
    if train_ensemble:
        print("\n[6/6] Training uncertainty ensemble...")
        uncertainty_estimator = UncertaintyEstimator(n_models=5)
        uncertainty_estimator.train_ensemble(X_train_norm, y_train, X_val_norm, y_val,
                                            epochs=150, verbose=False)
        
        # Evaluate calibration
        uncertainty_estimator.evaluate_calibration(X_test_norm, y_test)
        
        # Generate uncertainty plots
        uncertainty_estimator.plot_uncertainty(X_test_norm, y_test)
        
        # Save ensemble
        uncertainty_estimator.save_ensemble("models/ensemble")
    else:
        print("\n[6/6] Skipping ensemble training (use single model for uncertainty)")
    
    # Save artifacts for inference
    import joblib
    joblib.dump(fe, "models/feature_engineer.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(selected_features, "models/selected_features.pkl")
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETE!")
    print("="*70)
    print("\nSaved models:")
    print("  ✓ models/pcnn_model.pt")
    print("  ✓ models/feature_engineer.pkl")
    print("  ✓ models/scaler.pkl")
    if train_ensemble:
        print("  ✓ models/ensemble_*.pt (5 models)")
    
    print("\nGenerated plots:")
    print("  ✓ plots/target_distributions.png")
    print("  ✓ plots/correlation_heatmap.png")
    print("  ✓ plots/physical_constraint.png")
    print("  ✓ plots/family_properties.png")
    if train_ensemble:
        print("  ✓ plots/uncertainty_analysis.png")
    
    print("\n" + "="*70)
    print("Next steps:")
    print("  1. Launch dashboard: streamlit run app.py")
    print("  2. Try inverse design: Input target properties → Get composition")
    print("  3. Make predictions with confidence intervals")
    print("="*70)

if __name__ == "__main__":
    main()
