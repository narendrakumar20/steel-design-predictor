"""
Quick test script to verify installation and models
"""
import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    required_packages = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('sklearn', None),
        ('torch', None),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('streamlit', 'st'),
        ('plotly', None),
        ('scipy', None),
        ('joblib', None),
    ]
    
    failed = []
    for package_info in required_packages:
        package = package_info[0]
        alias = package_info[1]
        
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ✗ {package} - {e}")
            failed.append(package)
    
    return len(failed) == 0, failed

def test_data():
    """Check if dataset exists"""
    print("\nChecking dataset...")
    if os.path.exists("data/steel_data.csv"):
        import pandas as pd
        df = pd.read_csv("data/steel_data.csv")
        print(f"  ✓ Dataset found: {len(df)} samples")
        return True
    else:
        print("  ✗ Dataset not found. Run: python generate_dataset.py")
        return False

def test_models():
    """Check if models are trained"""
    print("\nChecking trained models...")
    required_files = [
        "models/pcnn_model.pt",
        "models/feature_engineer.pkl",
        "models/scaler.pkl",
        "models/selected_features.pkl"
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} not found")
            all_exist = False
    
    if not all_exist:
        print("  → Run: python train_model.py")
    
    return all_exist

def test_quick_prediction():
    """Test a quick prediction"""
    print("\nTesting prediction pipeline...")
    try:
        import pandas as pd
        import joblib
        from src.pcnn_model import SteelPropertyPredictor
        from src.utils import ELEMENT_COLS
        
        # Load models
        feature_engineer = joblib.load("models/feature_engineer.pkl")
        scaler = joblib.load("models/scaler.pkl")
        selected_features = joblib.load("models/selected_features.pkl")
        
        predictor = SteelPropertyPredictor(input_dim=len(selected_features))
        predictor.load_model("models/pcnn_model.pt")
        
        # Test composition (simple carbon steel)
        composition = {
            'C': 0.4, 'Mn': 0.8, 'Si': 0.3, 'Cr': 0.2, 'Ni': 0.1,
            'Mo': 0.05, 'Cu': 0.1, 'V': 0.02, 'W': 0.0, 'Co': 0.0,
            'Al': 0.03, 'Ti': 0.01, 'Nb': 0.01
        }
        
        comp_df = pd.DataFrame([composition])
        comp_features = feature_engineer.create_features(comp_df)
        comp_selected = comp_features[selected_features]
        comp_scaled = pd.DataFrame(
            scaler.transform(comp_selected),
            columns=comp_selected.columns
        )
        
        predictions = predictor.predict(comp_scaled)[0]
        ys, uts, elong = predictions
        
        print(f"  Test composition: 0.4% C, 0.8% Mn, 0.3% Si, 0.2% Cr")
        print(f"  Predicted YS:  {ys:.0f} MPa")
        print(f"  Predicted UTS: {uts:.0f} MPa")
        print(f"  Predicted Elong: {elong:.1f} %")
        
        if uts > ys:
            print(f"  ✓ Physics constraint satisfied (UTS > YS)")
        else:
            print(f"  ✗ Physics constraint violated!")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Prediction test failed: {e}")
        return False

def main():
    print("="*60)
    print(" STEEL MATERIALS CO-PILOT - SYSTEM CHECK")
    print("="*60)
    
    # Test imports
    imports_ok, failed = test_imports()
    
    if not imports_ok:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return
    
    # Test data
    data_ok = test_data()
    
    # Test models
    models_ok = test_models()
    
    # Quick prediction test
    if models_ok:
        prediction_ok = test_quick_prediction()
    else:
        prediction_ok = False
    
    # Summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    
    if imports_ok and data_ok and models_ok and prediction_ok:
        print("✅ All systems operational!")
        print("\nNext steps:")
        print("  1. Launch dashboard: streamlit run app.py")
        print("  2. Open browser and test all features")
        print("  3. Review PRESENTATION_GUIDE.md for pitch prep")
    else:
        print("⚠️ Setup incomplete. Follow these steps:\n")
        if not data_ok:
            print("  1. Generate dataset: python generate_dataset.py")
        if not models_ok:
            print("  2. Train models: python train_model.py")
        if not prediction_ok and models_ok:
            print("  3. Debug model loading issues")
    
    print("="*60)

if __name__ == "__main__":
    main()
