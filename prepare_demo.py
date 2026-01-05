"""
Demo helper script - Run before presentation to verify everything works
"""
import os
import sys

def check_files():
    """Verify all necessary files exist"""
    print("Checking project files...")
    
    required_files = {
        "Code Files": [
            "app.py",
            "generate_dataset.py", 
            "train_model.py",
            "test_setup.py",
            "src/data_loader.py",
            "src/feature_engineering.py",
            "src/pcnn_model.py",
            "src/inverse_design.py",
            "src/uncertainty.py",
            "src/utils.py"
        ],
        "Data Files": [
            "data/steel_data.csv"
        ],
        "Model Files": [
            "models/pcnn_model.pt",
            "models/feature_engineer.pkl",
            "models/scaler.pkl",
            "models/selected_features.pkl"
        ],
        "Documentation": [
            "README.md",
            "QUICKSTART.md",
            "PRESENTATION_GUIDE.md"
        ]
    }
    
    all_ok = True
    for category, files in required_files.items():
        print(f"\n{category}:")
        for file in files:
            exists = os.path.exists(file)
            status = "✓" if exists else "✗"
            print(f"  {status} {file}")
            if not exists:
                all_ok = False
    
    return all_ok

def test_dashboard():
    """Test if dashboard can be imported"""
    print("\nTesting dashboard import...")
    try:
        import app
        print("  ✓ Dashboard can be imported")
        return True
    except Exception as e:
        print(f"  ✗ Dashboard import failed: {e}")
        return False

def test_quick_prediction():
    """Run a quick prediction"""
    print("\nTesting prediction...")
    try:
        import pandas as pd
        import joblib
        from src.pcnn_model import SteelPropertyPredictor
        
        feature_engineer = joblib.load("models/feature_engineer.pkl")
        scaler = joblib.load("models/scaler.pkl")
        selected_features = joblib.load("models/selected_features.pkl")
        
        predictor = SteelPropertyPredictor(input_dim=len(selected_features))
        predictor.load_model("models/pcnn_model.pt")
        
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
        
        print(f"  ✓ Prediction successful")
        print(f"    YS: {ys:.0f} MPa, UTS: {uts:.0f} MPa, Elongation: {elong:.1f}%")
        print(f"    Constraint check: {'✓ PASS' if uts > ys else '✗ FAIL'}")
        
        return uts > ys
        
    except Exception as e:
        print(f"  ✗ Prediction failed: {e}")
        return False

def create_demo_compositions():
    """Create sample compositions for demo"""
    print("\nCreating demo composition files...")
    
    # Demo 1: Carbon steel
    demo1 = {
        'name': 'Carbon Steel (AISI 1045)',
        'composition': {
            'C': 0.45, 'Mn': 0.75, 'Si': 0.25, 'Cr': 0.1, 'Ni': 0.05,
            'Mo': 0.02, 'Cu': 0.05, 'V': 0.01, 'W': 0.0, 'Co': 0.0,
            'Al': 0.03, 'Ti': 0.005, 'Nb': 0.005
        }
    }
    
    # Demo 2: Stainless steel
    demo2 = {
        'name': 'Stainless Steel (304)',
        'composition': {
            'C': 0.08, 'Mn': 2.0, 'Si': 0.75, 'Cr': 18.0, 'Ni': 8.0,
            'Mo': 0.1, 'Cu': 0.2, 'V': 0.05, 'W': 0.0, 'Co': 0.1,
            'Al': 0.05, 'Ti': 0.02, 'Nb': 0.02
        }
    }
    
    # Demo 3: High strength
    demo3 = {
        'name': 'High Strength Steel',
        'composition': {
            'C': 0.15, 'Mn': 1.5, 'Si': 0.4, 'Cr': 0.6, 'Ni': 0.4,
            'Mo': 0.25, 'Cu': 0.3, 'V': 0.1, 'W': 0.05, 'Co': 0.02,
            'Al': 0.04, 'Ti': 0.03, 'Nb': 0.05
        }
    }
    
    # Save to file
    os.makedirs("demo", exist_ok=True)
    
    with open("demo/demo_compositions.txt", "w") as f:
        for demo in [demo1, demo2, demo3]:
            f.write(f"\n{demo['name']}:\n")
            for element, value in demo['composition'].items():
                f.write(f"  {element}: {value}%\n")
    
    print("  ✓ Demo compositions saved to demo/demo_compositions.txt")
    return True

def create_presentation_checklist():
    """Create pre-presentation checklist"""
    checklist = """
# 🎯 PRE-PRESENTATION CHECKLIST

## Technical Setup
- [ ] All models trained and loaded successfully
- [ ] Dashboard launches without errors (test: streamlit run app.py)
- [ ] All plots generated in plots/ folder
- [ ] Test prediction works (python test_setup.py)
- [ ] Test inverse design works (click through dashboard)
- [ ] Internet connection stable (for demo)

## Demo Preparation
- [ ] Browser tabs ready:
  - [ ] Dashboard (localhost:8501)
  - [ ] plots/ folder open
  - [ ] demo/demo_compositions.txt open
- [ ] Demo compositions memorized or on note card
- [ ] Backup screenshots saved (in case demo fails)
- [ ] Timer set for 5-minute pitch

## Presentation Materials
- [ ] Slide deck finalized (10 slides max)
- [ ] Talking points memorized (see PRESENTATION_GUIDE.md)
- [ ] Question answers prepared
- [ ] Business model slides ready
- [ ] Technical architecture diagram ready

## Story Flow
- [ ] Opening hook practiced (75K experiment vs 3 seconds)
- [ ] Innovation #1 demo smooth (PCNN constraint)
- [ ] Innovation #2 demo smooth (inverse design)
- [ ] Innovation #3 demo smooth (trust scores)
- [ ] Business case slides flow well
- [ ] Closing pitch confident

## Team Coordination
- [ ] Roles defined (who presents what)
- [ ] Transitions practiced
- [ ] Backup presenter identified
- [ ] Q&A strategy agreed

## Contingency Plans
- [ ] Laptop fully charged + charger ready
- [ ] Backup laptop available
- [ ] Screenshots saved (if live demo fails)
- [ ] Code snippets ready to show
- [ ] GitHub repo link ready to share

## Final Checks (30 min before)
- [ ] Full run-through completed (< 5 minutes)
- [ ] Dashboard working perfectly
- [ ] All team members present
- [ ] Water bottles ready
- [ ] Deep breath taken 😊

---

**Remember: You're not pitching a project. You're pitching a COMPANY.**

**You've got this! 🚀**
"""
    
    with open("demo/CHECKLIST.md", "w", encoding='utf-8') as f:
        f.write(checklist)
    
    print("  ✓ Checklist saved to demo/CHECKLIST.md")

def main():
    print("="*70)
    print(" DEMO PREPARATION CHECK")
    print("="*70)
    
    # Check files
    files_ok = check_files()
    
    # Test dashboard
    dashboard_ok = test_dashboard()
    
    # Test prediction
    prediction_ok = test_quick_prediction()
    
    # Create demo files
    demo_ok = create_demo_compositions()
    
    # Create checklist
    create_presentation_checklist()
    
    # Summary
    print("\n" + "="*70)
    print(" DEMO READINESS SUMMARY")
    print("="*70)
    
    if files_ok and dashboard_ok and prediction_ok:
        print("\n✅ SYSTEM READY FOR DEMO!")
        print("\nNext steps:")
        print("  1. Review: demo/CHECKLIST.md")
        print("  2. Practice: Read PRESENTATION_GUIDE.md")
        print("  3. Launch: streamlit run app.py")
        print("  4. Test: Try all 3 modes (prediction, inverse, batch)")
        print("  5. Rehearse: 5-minute pitch 3+ times")
        print("\n🎯 Demo compositions available in: demo/demo_compositions.txt")
    else:
        print("\n⚠️ ISSUES DETECTED!")
        if not files_ok:
            print("  - Missing files (check above)")
        if not dashboard_ok:
            print("  - Dashboard import issues")
        if not prediction_ok:
            print("  - Prediction pipeline issues")
        print("\nFix these before demo!")
    
    print("="*70)

if __name__ == "__main__":
    main()
