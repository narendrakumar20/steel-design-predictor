"""
Streamlit Dashboard for Steel Materials Co-Pilot
Interactive UI with all three innovative features
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import torch
import joblib

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pcnn_model import SteelPropertyPredictor
from src.inverse_design import InverseDesignEngine
from src.uncertainty import UncertaintyEstimator
from src.utils import ELEMENT_COLS, TARGET_COLS, format_composition_string, calculate_cost_estimate

# Page config
st.set_page_config(
    page_title="Steel Property Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/narendrakumar20/steel-design-predictor',
        'Report a bug': 'https://github.com/narendrakumar20/steel-design-predictor/issues',
        'About': '# Steel Property Predictor\nAI-powered materials discovery platform'
    }
)

# Custom CSS for extreme-level animated dark theme
st.markdown("""
<style>
    /* ============================================
       KEYFRAME ANIMATIONS
       ============================================ */
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(5deg); }
    }
    
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.4); }
        50% { box-shadow: 0 0 40px rgba(59, 130, 246, 0.8); }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.8);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    @keyframes countUp {
        from { opacity: 0; transform: scale(0.5); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes particle-float {
        0%, 100% {
            transform: translate(0, 0) scale(1);
            opacity: 0.6;
        }
        25% {
            transform: translate(10px, -10px) scale(1.1);
            opacity: 0.8;
        }
        50% {
            transform: translate(-5px, -20px) scale(0.9);
            opacity: 1;
        }
        75% {
            transform: translate(-15px, -10px) scale(1.05);
            opacity: 0.7;
        }
    }
    
    @keyframes glow-pulse {
        0%, 100% {
            filter: drop-shadow(0 0 5px rgba(96, 165, 250, 0.5));
        }
        50% {
            filter: drop-shadow(0 0 20px rgba(96, 165, 250, 1));
        }
    }
    
    /* ============================================
       BACKGROUND & LAYOUT
       ============================================ */
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #1e3a8a 100%);
        background-size: 400% 400%;
        animation: gradient-shift 15s ease infinite;
        position: relative;
        overflow-x: hidden;
    }
    
    .main {
        background: transparent;
        animation: fadeIn 0.8s ease-in;
        position: relative;
        z-index: 1;
    }
    
    [data-testid="stAppViewContainer"] {
        background: transparent;
        position: relative;
        z-index: 1;
    }
    
    /* Animated particle background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 30%, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(124, 58, 237, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, rgba(37, 99, 235, 0.05) 0%, transparent 50%);
        animation: float 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    /* Floating orbs */
    .stApp::after {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background-image:
            radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.15) 1px, transparent 1px),
            radial-gradient(circle at 80% 80%, rgba(124, 58, 237, 0.15) 1px, transparent 1px),
            radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.15) 1px, transparent 1px),
            radial-gradient(circle at 30% 70%, rgba(37, 99, 235, 0.15) 1px, transparent 1px),
            radial-gradient(circle at 70% 30%, rgba(59, 130, 246, 0.15) 1px, transparent 1px);
        background-size: 200px 200px, 300px 300px, 250px 250px, 350px 350px, 180px 180px;
        animation: particle-float 30s linear infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    /* ============================================
       TYPOGRAPHY WITH ANIMATIONS
       ============================================ */
    
    h1 {
        color: #60a5fa !important;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5), 0 0 30px rgba(59, 130, 246, 0.3);
        animation: fadeInUp 0.6s ease-out, glow-pulse 3s ease-in-out infinite;
    }
    
    h2, h3 {
        color: #93c5fd !important;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .stMarkdown, p, label, span {
        color: #e2e8f0 !important;
    }
    
    /* ============================================
       GLASSMORPHISM CARDS
       ============================================ */
    
    .stExpander {
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.1);
        margin-bottom: 15px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: scaleIn 0.5s ease-out;
    }
    
    .stExpander:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(59, 130, 246, 0.2), 0 0 0 1px rgba(96, 165, 250, 0.3);
        border-color: rgba(96, 165, 250, 0.5);
    }
    
    /* ============================================
       ANIMATED METRICS
       ============================================ */
    
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: bold;
        color: #60a5fa !important;
        animation: countUp 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        text-shadow: 0 0 20px rgba(96, 165, 250, 0.4);
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        animation: fadeIn 0.4s ease-in;
    }
    
    [data-testid="stMetricDelta"] {
        color: #cbd5e1 !important;
        animation: fadeInUp 0.5s ease-out 0.2s backwards;
    }
    
    /* ============================================
       PREMIUM BUTTONS
       ============================================ */
    
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.7rem 2.5rem;
        border: none;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6), 0 0 30px rgba(59, 130, 246, 0.3);
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(0.98);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.5);
    }
    
    /* Primary button pulse effect */
    .stButton>button[kind="primary"] {
        animation: fadeInUp 0.5s ease-out, pulse-glow 2s ease-in-out infinite;
    }
    
    /* ============================================
       TRUST SCORE CARDS WITH GLOW
       ============================================ */
    
    .trust-high {
        background: linear-gradient(135deg, #166534 0%, #15803d 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #22c55e;
        box-shadow: 0 4px 20px rgba(34, 197, 94, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        color: #86efac !important;
        animation: fadeInUp 0.6s ease-out, pulse-glow 3s ease-in-out infinite;
        transition: all 0.3s ease;
    }
    
    .trust-high:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 30px rgba(34, 197, 94, 0.5);
    }
    
    .trust-medium {
        background: linear-gradient(135deg, #854d0e 0%, #a16207 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #eab308;
        box-shadow: 0 4px 20px rgba(234, 179, 8, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        color: #fde047 !important;
        animation: fadeInUp 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .trust-medium:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 30px rgba(234, 179, 8, 0.5);
    }
    
    .trust-low {
        background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #ef4444;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        color: #fca5a5 !important;
        animation: fadeInUp 0.6s ease-out, shake 0.5s ease-in-out;
        transition: all 0.3s ease;
    }
    
    .trust-low:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 30px rgba(239, 68, 68, 0.5);
    }
    
    /* ============================================
       INFO BOXES WITH SLIDE-IN
       ============================================ */
    
    .info-box {
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.8) 0%, rgba(30, 64, 175, 0.6) 100%);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 15px 0;
        color: #93c5fd !important;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.2);
        animation: slideInLeft 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .info-box:hover {
        transform: translateX(5px);
        border-left-width: 6px;
        box-shadow: 0 6px 30px rgba(59, 130, 246, 0.3);
    }
    
    .info-box strong {
        color: #60a5fa !important;
        text-shadow: 0 0 10px rgba(96, 165, 250, 0.5);
    }
    
    /* ============================================
       FAMILY BADGE WITH ANIMATION
       ============================================ */
    
    .family-badge {
        background: linear-gradient(90deg, #7c3aed 0%, #6366f1 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        display: inline-block;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.5);
        margin: 15px 0;
        animation: scaleIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1), float 3s ease-in-out infinite;
        transition: all 0.3s ease;
    }
    
    .family-badge:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 25px rgba(124, 58, 237, 0.7);
    }
    
    /* ============================================
       SIDEBAR WITH SLIDE-IN
       ============================================ */
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%) !important;
        animation: slideInLeft 0.5s ease-out;
        box-shadow: 5px 0 30px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
        font-size: 16px;
        padding: 12px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(5px);
    }
    
    /* ============================================
       ANIMATED INPUT FIELDS
       ============================================ */
    
    .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #334155;
        background-color: rgba(30, 41, 59, 0.8) !important;
        backdrop-filter: blur(5px);
        color: #e2e8f0 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stNumberInput>div>div>input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2), 0 0 20px rgba(59, 130, 246, 0.3);
        background-color: rgba(30, 41, 59, 0.95) !important;
        transform: scale(1.02);
    }
    
    .stNumberInput>div>div>input:hover {
        border-color: #475569;
        background-color: rgba(30, 41, 59, 0.9) !important;
    }
    
    /* ============================================
       SELECT BOXES WITH ANIMATION
       ============================================ */
    
    .stSelectbox>div>div {
        background-color: rgba(30, 41, 59, 0.8) !important;
        backdrop-filter: blur(5px);
        color: #e2e8f0 !important;
        border: 2px solid #334155;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox>div>div:hover {
        border-color: #475569;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* ============================================
       FILE UPLOADER WITH ANIMATION
       ============================================ */
    
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(51, 65, 85, 0.4) 100%);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 30px;
        border: 2px dashed #475569;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
        border-style: solid;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.2);
        transform: translateY(-3px);
    }
    
    /* ============================================
       DATAFRAMES WITH HOVER EFFECTS
       ============================================ */
    
    .dataframe {
        background-color: rgba(30, 41, 59, 0.8) !important;
        backdrop-filter: blur(10px);
        color: #e2e8f0 !important;
        border-radius: 10px;
        overflow: hidden;
        animation: fadeIn 0.5s ease-in;
    }
    
    .dataframe tbody tr {
        transition: all 0.2s ease;
    }
    
    .dataframe tbody tr:hover {
        background-color: rgba(59, 130, 246, 0.1) !important;
        transform: scale(1.01);
    }
    
    /* ============================================
       SUCCESS/INFO/WARNING/ERROR WITH ANIMATIONS
       ============================================ */
    
    .stSuccess {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%) !important;
        color: #6ee7b7 !important;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        animation: slideInRight 0.5s ease-out;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 100%) !important;
        color: #93c5fd !important;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        animation: slideInRight 0.5s ease-out;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%) !important;
        color: #fcd34d !important;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        animation: slideInRight 0.5s ease-out, bounce 1s ease-in-out;
    }
    
    .stError {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%) !important;
        color: #fca5a5 !important;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        animation: slideInRight 0.5s ease-out, shake 0.5s ease-in-out;
    }
    
    /* ============================================
       LOADING SPINNERS
       ============================================ */
    
    .stSpinner > div {
        border-color: #3b82f6 !important;
        animation: rotate 1s linear infinite;
    }
    
    /* ============================================
       COLUMN ANIMATIONS
       ============================================ */
    
    [data-testid="column"] {
        animation: fadeInUp 0.6s ease-out;
    }
    
    [data-testid="column"]:nth-child(1) {
        animation-delay: 0.1s;
    }
    
    [data-testid="column"]:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    [data-testid="column"]:nth-child(3) {
        animation-delay: 0.3s;
    }
    
    /* ============================================
       SCROLLBAR STYLING
       ============================================ */
    
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%);
        border-radius: 10px;
        border: 2px solid #1e293b;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* ============================================
       PERFORMANCE OPTIMIZATIONS
       ============================================ */
    
    * {
        will-change: auto;
    }
    
    .stButton>button,
    .stExpander,
    [data-testid="stMetricValue"],
    .info-box,
    .family-badge {
        will-change: transform, box-shadow;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models and artifacts"""
    try:
        # Load feature engineering pipeline
        feature_engineer = joblib.load("models/feature_engineer.pkl")
        scaler = joblib.load("models/scaler.pkl")
        selected_features = joblib.load("models/selected_features.pkl")
        
        # Load main predictor
        predictor = SteelPropertyPredictor(input_dim=len(selected_features))
        predictor.load_model("models/pcnn_model.pt")
        
        # Try to load ensemble for uncertainty
        uncertainty_estimator = None
        if os.path.exists("models/ensemble_model0.pt"):
            uncertainty_estimator = UncertaintyEstimator(n_models=5)
            uncertainty_estimator.load_ensemble("models/ensemble")
        
        return predictor, feature_engineer, scaler, selected_features, uncertainty_estimator
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run 'python train_model.py' first to train the models.")
        return None, None, None, None, None

def main():
    # Header with enhanced styling
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔬 Steel Property Predictor")
        st.markdown("### AI-Powered Materials Discovery Platform")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-View_Code-blue?logo=github)](https://github.com/narendrakumar20/steel-design-predictor)")
    
    st.markdown("---")
    
    # Load models with progress indicator
    with st.spinner("🔄 Loading AI models..."):
        predictor, feature_engineer, scaler, selected_features, uncertainty_estimator = load_models()
    
    if predictor is None:
        st.stop()
    
    # Success message
    st.sidebar.success("✅ Models loaded successfully!")
    
    # Sidebar navigation with enhanced styling
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    mode = st.sidebar.radio(
        "Select Mode:",
        ["🎯 Property Prediction", "💡 Inverse Design", "📊 Batch Analysis", "ℹ️ About"],
        help="Choose the analysis mode you want to use"
    )
    
    # Add helpful info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Quick Tips")
    
    if mode == "🎯 Property Prediction":
        st.sidebar.info("""
        **How to use:**
        1. Enter element percentages
        2. Click 'Predict Properties'
        3. Review predictions with confidence intervals
        """)
    elif mode == "💡 Inverse Design":
        st.sidebar.info("""
        **How to use:**
        1. Set target properties
        2. Adjust cost sensitivity
        3. Generate optimal compositions
        """)
    elif mode == "📊 Batch Analysis":
        st.sidebar.info("""
        **How to use:**
        1. Upload CSV with compositions
        2. Click 'Predict All'
        3. Download results
        """)
    
    if mode == "🎯 Property Prediction":
        property_prediction_mode(predictor, feature_engineer, scaler, selected_features, uncertainty_estimator)
    
    elif mode == "💡 Inverse Design":
        inverse_design_mode(predictor, feature_engineer, scaler, selected_features)
    
    elif mode == "📊 Batch Analysis":
        batch_analysis_mode(predictor, feature_engineer, scaler, selected_features, uncertainty_estimator)
    
    elif mode == "ℹ️ About":
        about_page()

def property_prediction_mode(predictor, feature_engineer, scaler, selected_features, uncertainty_estimator):
    """Mode 1: Predict properties from composition"""
    st.header("🎯 Property Prediction from Composition")
    
    # Info banner
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ How it works:</strong> Enter the chemical composition of your steel alloy (weight percentages), 
        and our AI will predict its mechanical properties with confidence intervals and physics validation.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("📝 Input Composition (wt%)")
        st.markdown("<small>Enter weight percentage for each element</small>", unsafe_allow_html=True)
        
        # Preset examples
        st.markdown("**Quick Presets:**")
        preset = st.selectbox(
            "Load example composition",
            ["Custom", "Low Carbon Steel", "Medium Carbon Steel", "Stainless Steel 304", "Tool Steel"],
            help="Select a preset to auto-fill common steel compositions"
        )
        
        # Preset values
        presets = {
            "Custom": {},
            "Low Carbon Steel": {'C': 0.08, 'Mn': 0.45, 'Si': 0.18, 'Cr': 0.0, 'Ni': 0.0, 'Mo': 0.0, 'Cu': 0.0, 'V': 0.0, 'W': 0.0, 'Co': 0.0, 'Al': 0.02, 'Ti': 0.0, 'Nb': 0.0},
            "Medium Carbon Steel": {'C': 0.42, 'Mn': 0.75, 'Si': 0.30, 'Cr': 1.00, 'Ni': 0.0, 'Mo': 0.20, 'Cu': 0.0, 'V': 0.0, 'W': 0.0, 'Co': 0.0, 'Al': 0.0, 'Ti': 0.0, 'Nb': 0.0},
            "Stainless Steel 304": {'C': 0.08, 'Mn': 2.0, 'Si': 0.75, 'Cr': 18.0, 'Ni': 8.0, 'Mo': 0.0, 'Cu': 0.0, 'V': 0.0, 'W': 0.0, 'Co': 0.0, 'Al': 0.0, 'Ti': 0.0, 'Nb': 0.0},
            "Tool Steel": {'C': 0.95, 'Mn': 1.2, 'Si': 0.35, 'Cr': 5.0, 'Ni': 0.0, 'Mo': 1.1, 'V': 1.0, 'W': 6.5, 'Co': 0.0, 'Al': 0.0, 'Ti': 0.0, 'Nb': 0.0}
        }
        
        default_composition = presets.get(preset, {})
        st.markdown("---")
        
        # Input fields for elements
        composition = {}
        
        # Major elements with tooltips
        st.markdown("**Major Elements:**")
        element_help = {
            'C': 'Carbon: Increases hardness and strength',
            'Mn': 'Manganese: Improves hardenability',
            'Si': 'Silicon: Deoxidizer, improves strength',
            'Cr': 'Chromium: Increases corrosion resistance',
            'Ni': 'Nickel: Improves toughness and ductility',
            'Mo': 'Molybdenum: Improves high-temperature strength'
        }
        cols = st.columns(3)
        for idx, element in enumerate(['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo']):
            with cols[idx % 3]:
                default_value = default_composition.get(element, 0.2 if element == 'C' else 0.5)
                composition[element] = st.number_input(
                    f"{element} (%)", min_value=0.0, max_value=20.0, 
                    value=default_value, step=0.01,
                    help=element_help[element]
                )
        
        # Minor elements with tooltips
        st.markdown("**Minor/Trace Elements:**")
        minor_help = {
            'Cu': 'Copper: Improves corrosion resistance',
            'V': 'Vanadium: Grain refinement, high strength',
            'W': 'Tungsten: High-temperature hardness',
            'Co': 'Cobalt: High-temperature strength',
            'Al': 'Aluminum: Deoxidizer, grain refiner',
            'Ti': 'Titanium: Grain refiner, carbide former',
            'Nb': 'Niobium: Grain refiner, precipitation hardening'
        }
        with st.expander("⚙️ Show Minor/Trace Elements (Optional)", expanded=False):
            cols = st.columns(3)
            for idx, element in enumerate(['Cu', 'V', 'W', 'Co', 'Al', 'Ti', 'Nb']):
                with cols[idx % 3]:
                    default_value = default_composition.get(element, 0.0)
                    composition[element] = st.number_input(
                        f"{element} (%)", min_value=0.0, max_value=10.0,
                        value=default_value, step=0.01, key=f"minor_{element}",
                        help=minor_help[element]
                    )
        
        # Calculate totals with visual feedback
        st.markdown("---")
        total_alloy = sum(composition.values())
        iron_content = 100 - total_alloy
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Alloying Elements", f"{total_alloy:.2f}%", 
                     delta="High" if total_alloy > 10 else "Normal")
        with col_b:
            st.metric("Iron (Balance)", f"{iron_content:.2f}%")
        
        # Validation messages
        if total_alloy > 50:
            st.error("⚠️ Total alloying elements exceed 50%. This is unusual for steel.")
        elif total_alloy > 30:
            st.warning("⚠️ High alloy content. Ensure this is intentional.")
        elif iron_content < 50:
            st.warning("⚠️ Iron content below 50%. This may not be classified as steel.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Predict button with better styling
        predict_disabled = total_alloy > 50 or iron_content < 0
        if st.button("🔮 Predict Properties", type="primary", disabled=predict_disabled, use_container_width=True):
            with col2:
                with st.spinner("🧠 AI is analyzing your composition..."):
                    predict_properties(composition, predictor, feature_engineer, scaler, 
                                     selected_features, uncertainty_estimator)

def predict_properties(composition, predictor, feature_engineer, scaler, selected_features, uncertainty_estimator):
    """Make prediction with uncertainty"""
    st.subheader("📊 Predicted Properties")
    st.markdown("<small>Based on AI model trained on 400+ steel alloys</small>", unsafe_allow_html=True)
    st.markdown("---")
    
    try:
        # Prepare input
        comp_df = pd.DataFrame([composition])
        comp_features = feature_engineer.create_features(comp_df)
        comp_selected = comp_features[selected_features]
        comp_scaled = pd.DataFrame(
            scaler.transform(comp_selected),
            columns=comp_selected.columns
        )
        
        # Make prediction
        if uncertainty_estimator:
            mean_pred, std_pred, conf_intervals = uncertainty_estimator.predict_with_uncertainty(comp_scaled)
            trust_score = uncertainty_estimator.get_trust_score(comp_scaled)[0]
            trust_category, trust_color = uncertainty_estimator.get_trust_category(trust_score)
            
            ys, uts, elong = mean_pred[0]
            ys_std, uts_std, elong_std = std_pred[0]
            ys_ci, uts_ci, elong_ci = conf_intervals[0]
        else:
            predictions = predictor.predict(comp_scaled)[0]
            ys, uts, elong = predictions
            trust_score = 75  # Default
            trust_category = "Medium"
            trust_color = "yellow"
        
        # Determine steel family
        from src.utils import get_steel_family
        steel_family = get_steel_family(composition)
        
        # Display steel family
        st.markdown(f'<div class="family-badge">🔬 Steel Family: {steel_family}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Display predictions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Yield Strength", f"{ys:.0f} MPa")
            if uncertainty_estimator:
                st.caption(f"95% CI: [{ys_ci[0]:.0f}, {ys_ci[1]:.0f}] MPa")
        
        with col2:
            st.metric("Ultimate Tensile Strength", f"{uts:.0f} MPa")
            if uncertainty_estimator:
                st.caption(f"95% CI: [{uts_ci[0]:.0f}, {uts_ci[1]:.0f}] MPa")
        
        with col3:
            st.metric("Elongation", f"{elong:.1f} %")
            if uncertainty_estimator:
                st.caption(f"95% CI: [{elong_ci[0]:.1f}, {elong_ci[1]:.1f}] %")
        
        # Trust score
        st.markdown("---")
        st.subheader("🎯 Prediction Trust Score")
        
        if trust_category == "High":
            st.markdown(f'<div class="trust-high">✅ <b>High Trust ({trust_score:.0f}/100)</b><br>This prediction is reliable. Composition is similar to training data.</div>', unsafe_allow_html=True)
        elif trust_category == "Medium":
            st.markdown(f'<div class="trust-medium">⚠️ <b>Medium Trust ({trust_score:.0f}/100)</b><br>Prediction is reasonable but verify with experiments for critical applications.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="trust-low">🚫 <b>Low Trust ({trust_score:.0f}/100)</b><br>Composition is outside training distribution. Experimental validation strongly recommended.</div>', unsafe_allow_html=True)
        
        # Physics constraint check
        st.markdown("---")
        st.subheader("⚗️ Physics Validation")
        
        if uts > ys:
            st.success(f"✅ Physical constraint satisfied: UTS ({uts:.0f} MPa) > YS ({ys:.0f} MPa)")
        else:
            st.error(f"❌ Physics violation: UTS ({uts:.0f} MPa) ≤ YS ({ys:.0f} MPa)")
        
        # Cost estimate
        cost = calculate_cost_estimate(composition)
        st.info(f"💰 Estimated relative cost: **{cost:.2f}x** base steel price")
        
        # Visualization
        st.markdown("---")
        st.subheader("📊 Composition Breakdown")
        
        # Pie chart
        comp_data = pd.DataFrame({
            'Element': list(composition.keys()) + ['Fe'],
            'Percentage': list(composition.values()) + [100 - sum(composition.values())]
        })
        comp_data = comp_data[comp_data['Percentage'] > 0.01].sort_values('Percentage', ascending=False)
        
        fig = px.pie(comp_data, values='Percentage', names='Element', 
                    title='Composition by Element')
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Prediction error: {e}")

def inverse_design_mode(predictor, feature_engineer, scaler, selected_features):
    """Mode 2: Dream Steel Builder - Inverse design"""
    st.header("💡 Inverse Design: Dream Steel Builder")
    
    # Info banner
    st.markdown("""
    <div class="info-box">
        <strong>🚀 How it works:</strong> Specify the mechanical properties you need, and our genetic algorithm 
        will search through thousands of possible compositions to find the optimal steel alloy that meets your requirements.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 Target Properties")
        st.markdown("<small>Specify your desired mechanical properties</small>", unsafe_allow_html=True)
        
        target_ys = st.number_input(
            "Yield Strength (MPa)",
            min_value=200, max_value=1200, value=600, step=10,
            help="The stress at which permanent deformation begins. Typical range: 200-1200 MPa"
        )
        
        target_uts = st.number_input(
            "Ultimate Tensile Strength (MPa)",
            min_value=300, max_value=1500, value=750, step=10,
            help="Maximum stress before fracture. Must be greater than Yield Strength. Typical range: 300-1500 MPa"
        )
        
        target_elong = st.number_input(
            "Elongation (%)",
            min_value=5.0, max_value=40.0, value=20.0, step=0.5,
            help="Ductility measure - percentage increase in length before fracture. Higher is more ductile."
        )
        
        # Validation with better feedback
        st.markdown("---")
        if target_uts <= target_ys:
            st.error("❌ **Invalid:** UTS must be greater than YS (physical constraint)")
            valid_input = False
        else:
            st.success("✅ Valid property targets")
            valid_input = True
        
        st.markdown("---")
        st.subheader("⚙️ Optimization Settings")
        st.markdown("<small>Fine-tune the optimization process</small>", unsafe_allow_html=True)
        
        cost_penalty = st.slider(
            "Cost Sensitivity",
            min_value=0.0, max_value=1.0, value=0.2, step=0.1,
            help="0 = Ignore cost (optimize only for properties), 1 = Minimize cost (may compromise properties)"
        )
        
        n_results = st.slider(
            "Number of solutions",
            min_value=1, max_value=10, value=5,
            help="How many optimized compositions to generate (top candidates)"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Generate Solutions", type="primary", disabled=not valid_input, use_container_width=True):
            with col2:
                st.info("🧬 Running genetic algorithm (100 population × 50 generations)...")
                generate_solutions(
                    target_ys, target_uts, target_elong,
                    predictor, feature_engineer, scaler, selected_features,
                    cost_penalty, n_results
                )

def generate_solutions(target_ys, target_uts, target_elong, predictor, 
                      feature_engineer, scaler, selected_features, cost_penalty, n_results):
    """Run inverse design optimization"""
    st.subheader("🎯 Optimized Steel Compositions")
    st.markdown("<small>Ranked by fitness score (lower is better)</small>", unsafe_allow_html=True)
    st.markdown("---")
    
    try:
        # Create inverse design engine
        engine = InverseDesignEngine(predictor, feature_engineer, scaler, selected_features)
        
        # Run optimization
        results = engine.optimize(
            target_ys=target_ys,
            target_uts=target_uts,
            target_elong=target_elong,
            population_size=100,
            generations=50,
            top_k=n_results,
            cost_penalty=cost_penalty,
            verbose=False
        )
        
        # Display results
        for i, result in enumerate(results):
            with st.expander(f"🥇 Solution #{i+1} - Fitness: {result['fitness']:.4f}", expanded=(i==0)):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Predicted Properties:**")
                    st.metric("Yield Strength", f"{result['predicted_ys']:.0f} MPa", 
                             delta=f"{result['ys_error_percent']:.1f}% error")
                    st.metric("UTS", f"{result['predicted_uts']:.0f} MPa",
                             delta=f"{result['uts_error_percent']:.1f}% error")
                    st.metric("Elongation", f"{result['predicted_elongation']:.1f} %",
                             delta=f"{result['elongation_error_percent']:.1f}% error")
                    
                    st.markdown(f"**Relative Cost:** {result['relative_cost']:.2f}x")
                
                with col2:
                    st.markdown("**Composition (wt%):**")
                    comp_df = pd.DataFrame({
                        'Element': result['composition'].keys(),
                        'Percentage': result['composition'].values()
                    })
                    comp_df = comp_df[comp_df['Percentage'] > 0.01].sort_values('Percentage', ascending=False)
                    
                    for _, row in comp_df.head(10).iterrows():
                        st.text(f"{row['Element']:>3s}: {row['Percentage']:>6.3f}%")
                    
                    if len(comp_df) > 10:
                        st.caption(f"... and {len(comp_df)-10} more elements")
        
        st.success("✅ Optimization complete!")
        
    except Exception as e:
        st.error(f"Optimization error: {e}")
        import traceback
        st.code(traceback.format_exc())

def batch_analysis_mode(predictor, feature_engineer, scaler, selected_features, uncertainty_estimator):
    """Mode 3: Batch analysis from CSV"""
    st.header("📊 Batch Analysis")
    
    # Info banner
    st.markdown("""
    <div class="info-box">
        <strong>📁 How it works:</strong> Upload a CSV file containing multiple steel compositions, 
        and get predictions for all samples at once. Perfect for analyzing large datasets or comparing multiple alloys.
    </div>
    """, unsafe_allow_html=True)
    
    # File format info
    with st.expander("📝 CSV Format Requirements", expanded=False):
        st.markdown("""
        **Required columns (weight %):**
        - C, Mn, Si, Cr, Ni, Mo, Cu, V, W, Co, Al, Ti, Nb
        
        **Example:**
        ```
        C,Mn,Si,Cr,Ni,Mo,Cu,V,W,Co,Al,Ti,Nb
        0.42,0.75,0.30,1.00,0.00,0.20,0.00,0.00,0.00,0.00,0.00,0.00,0.00
        0.38,0.65,0.28,0.90,0.10,0.15,0.05,0.00,0.00,0.00,0.00,0.00,0.00
        ```
        
        [Download sample CSV template](https://github.com/narendrakumar20/steel-design-predictor/blob/master/data/sample_template.csv)
        """)
    
    uploaded_file = st.file_uploader(
        "📤 Upload CSV file with composition data", 
        type=['csv'],
        help="CSV file must contain columns for all 13 elements (C, Mn, Si, Cr, Ni, Mo, Cu, V, W, Co, Al, Ti, Nb)"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} samples")
            
            # Check if required columns exist
            missing_cols = [col for col in ELEMENT_COLS if col not in df.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                st.stop()
            
            st.dataframe(df.head())
            
            if st.button("🔮 Predict All", type="primary"):
                with st.spinner("Making predictions..."):
                    # Prepare features
                    X = df[ELEMENT_COLS]
                    X_features = feature_engineer.create_features(X)
                    X_selected = X_features[selected_features]
                    X_scaled = pd.DataFrame(
                        scaler.transform(X_selected),
                        columns=X_selected.columns
                    )
                    
                    # Predict
                    if uncertainty_estimator:
                        mean_pred, std_pred, conf_intervals = uncertainty_estimator.predict_with_uncertainty(X_scaled)
                        trust_scores = uncertainty_estimator.get_trust_score(X_scaled)
                        
                        df['predicted_ys'] = mean_pred[:, 0]
                        df['predicted_uts'] = mean_pred[:, 1]
                        df['predicted_elongation'] = mean_pred[:, 2]
                        df['trust_score'] = trust_scores
                    else:
                        predictions = predictor.predict(X_scaled)
                        df['predicted_ys'] = predictions[:, 0]
                        df['predicted_uts'] = predictions[:, 1]
                        df['predicted_elongation'] = predictions[:, 2]
                    
                    st.success("✅ Predictions complete!")
                    
                    # Display results
                    st.subheader("Results")
                    st.dataframe(df)
                    
                    # Download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "steel_predictions.csv",
                        "text/csv"
                    )
                    
                    # Visualizations
                    st.subheader("Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.scatter(df, x='predicted_ys', y='predicted_uts',
                                       hover_data=ELEMENT_COLS[:3],
                                       title='Predicted Strength Properties')
                        fig.add_trace(go.Scatter(x=[df['predicted_ys'].min(), df['predicted_ys'].max()],
                                                y=[df['predicted_ys'].min(), df['predicted_ys'].max()],
                                                mode='lines', name='UTS=YS boundary',
                                                line=dict(dash='dash', color='red')))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.histogram(df, x='predicted_ys', 
                                         title='Yield Strength Distribution',
                                         nbins=30)
                        st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error: {e}")

def about_page():
    """About page with project info"""
    st.header("ℹ️ About Steel Materials Co-Pilot")
    
    st.markdown("""
    ## 🎯 Mission
    Accelerate materials discovery and reduce experimentation costs by 100x using AI-powered prediction and inverse design.
    
    ## 🚀 Key Innovations
    
    ### 1. Physics-Constrained Neural Network (PCNN)
    - **Why it matters:** Traditional ML models can predict physically impossible properties
    - **Our solution:** Custom loss function enforces UTS > YS constraint in the network architecture
    - **Impact:** 100% physics compliance, building trust with engineers
    
    ### 2. Inverse Design Engine (Dream Steel Builder)
    - **Why it matters:** Engineers know what properties they need, not what composition to use
    - **Our solution:** Genetic algorithm explores 10,000+ compositions to find optimal candidates
    - **Impact:** Design new materials in seconds vs. months of trial-and-error
    
    ### 3. Uncertainty Quantification
    - **Why it matters:** Engineers won't trust "black box" predictions
    - **Our solution:** Ensemble methods provide confidence intervals and trust scores
    - **Impact:** Know WHEN to trust predictions and when to run experiments
    
    ## 💼 Business Model
    - **SaaS Platform:** $2K-50K/month subscriptions
    - **Transaction Model:** $50 per prediction
    - **Custom Consulting:** $200K+ for proprietary models
    
    ## 📈 Market Opportunity
    - **TAM:** $150M (3,000 global steel manufacturers)
    - **Early Adopter Focus:** Advanced manufacturers in NA/EU
    - **Expansion:** Aluminum, titanium, concrete, battery materials
    
    ## 🏆 Competitive Advantages
    1. **Data Flywheel:** More customers → Better models → More customers
    2. **Domain Expertise:** Physics-informed approach is defensible
    3. **First Mover:** No direct competitors with all three features
    
    ## 👥 Team
    Built by materials science and ML experts passionate about accelerating innovation.
    
    ## 📞 Contact
    Ready to revolutionize your materials development? Let's talk!
    """)
    
    st.markdown("---")
    st.markdown("### 🔬 Technical Stack")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **ML Framework**
        - PyTorch
        - Scikit-learn
        - NumPy/Pandas
        """)
    
    with col2:
        st.markdown("""
        **Features**
        - Domain features
        - Interaction terms
        - Multi-target learning
        """)
    
    with col3:
        st.markdown("""
        **Deployment**
        - Streamlit UI
        - REST API ready
        - Cloud scalable
        """)

if __name__ == "__main__":
    main()
