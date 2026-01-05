# 🏆 Steel Materials Co-Pilot
## AI-Powered Materials Discovery Platform

### Revolutionary Features
1. **Physics-Constrained Neural Network** - Guarantees UTS > YS always
2. **Inverse Design Engine** - Dream Steel Builder: Input properties → Get composition
3. **Uncertainty Quantification** - Know when to trust predictions

### Quick Start
```bash
pip install -r requirements.txt
python generate_dataset.py  # Create demo dataset
streamlit run app.py        # Launch dashboard
```

### Project Structure
```
├── data/                   # Dataset storage
├── models/                 # Trained models
├── src/
│   ├── data_loader.py     # Data loading & EDA
│   ├── feature_engineering.py  # Domain features
│   ├── pcnn_model.py      # Physics-constrained model
│   ├── inverse_design.py  # Composition optimizer
│   ├── uncertainty.py     # Confidence estimation
│   └── utils.py           # Helper functions
├── app.py                 # Streamlit dashboard
├── generate_dataset.py    # Demo data generator
└── train_model.py         # Training pipeline
```

### Innovation Highlights
- **No Physics Violations**: Custom loss function enforces metallurgical constraints
- **Real-time Inverse Design**: Find optimal compositions in seconds
- **Production-Ready**: Confidence intervals for every prediction
