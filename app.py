"""
Home Dashboard - Fertilizer Recommendation Agent
B.Tech Data Science Mini Project
"""

import os
import json
import streamlit as st
import pandas as pd

from src.ui_helpers import render_header, render_sidebar, apply_custom_css
from src.database import get_all_farm_records, get_all_recommendations, get_all_schedules

# Page configuration
st.set_page_config(
    page_title="Home Dashboard | Fertilizer Recommendation Agent",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

render_sidebar()
render_header(
    title="Intelligent Fertilizer Recommendation Agent",
    subtitle="A Machine Learning & Agricultural Decision Support System for Sustainable Crop Production",
    icon="🌾"
)

# Load metadata
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

meta = {}
meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
if os.path.exists(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)

# Fetch database metrics
try:
    df_farms = get_all_farm_records()
    total_farms = len(df_farms)
except Exception:
    total_farms = 0
    df_farms = pd.DataFrame()

try:
    df_recs = get_all_recommendations()
    total_recs = len(df_recs)
except Exception:
    total_recs = 0
    df_recs = pd.DataFrame()

try:
    df_scheds = get_all_schedules()
    total_tasks = len(df_scheds)
    completed_tasks = len(df_scheds[df_scheds["status"] == "Completed"]) if not df_scheds.empty else 0
except Exception:
    total_tasks = 0
    completed_tasks = 0
    df_scheds = pd.DataFrame()

# -------------------------------------------------------------
# Top Metric Cards
# -------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Best ML Model</div>
        <div class="metric-value">{meta.get('best_model_name', 'XGBoost')}</div>
        <div class="metric-subtitle">Accuracy: {meta.get('best_accuracy', 1.0)*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Recommendations</div>
        <div class="metric-value">{total_recs}</div>
        <div class="metric-subtitle">Saved In SQLite DB</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Active Farm Parcels</div>
        <div class="metric-value">{total_farms}</div>
        <div class="metric-subtitle">Tracked Plots</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Schedule Tasks</div>
        <div class="metric-value">{completed_tasks} / {total_tasks}</div>
        <div class="metric-subtitle">Completed Agronomic Stages</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Project Introduction & Architecture
# -------------------------------------------------------------
intro_col, act_col = st.columns([1.7, 1.3])

with intro_col:
    st.markdown("""
    <div class="agri-card">
        <h3 style="color: #1B5E20; margin-top: 0;">📌 Project Overview</h3>
        <p style="color: #4A5568; line-height: 1.6;">
            Excessive and imprecise fertilizer application leads to severe soil degradation, groundwater pollution, 
            and escalated farming costs. The <b>Intelligent Fertilizer Recommendation Agent</b> addresses this critical 
            agricultural challenge by utilizing supervised machine learning models trained on Kaggle agricultural data 
            to predict optimal fertilizer formulations tailored to specific soil and climatic conditions.
        </p>
        <div style="background-color: #F1F8E9; border-left: 4px solid #4CAF50; padding: 12px 16px; border-radius: 4px; margin: 15px 0;">
            <b>Core System Highlights:</b>
            <ul style="margin: 6px 0 0 0; padding-left: 20px; color: #2E7D32;">
                <li><b>Phase 1 (Machine Learning):</b> Rigorous 5-model comparison (Random Forest, XGBoost, SVM, KNN, MLP) with 0% data leakage.</li>
                <li><b>Phase 2 (Agricultural Agent):</b> 5 integrated modules covering prediction, stage-wise scheduling, farm parcel CRUD, offline AI assistance, and historical tracking.</li>
            </ul>
        </div>
        <p style="color: #718096; font-size: 0.9rem; margin-bottom: 0;">
            Benchmark Dataset: <i>gdabhishek/fertilizer-prediction</i> (99 samples, 8 input features, 7 target fertilizers).
        </p>
    </div>
    """, unsafe_allow_html=True)

with act_col:
    st.markdown("""
    <div class="agri-card">
        <h3 style="color: #1B5E20; margin-top: 0;">⚡ Quick Navigation</h3>
    """, unsafe_allow_html=True)

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("🌱 Get Recommendation", use_container_width=True):
            st.switch_page("pages/1_🌱_Fertilizer_Recommendation.py")
        if st.button("🚜 Farm Management", use_container_width=True):
            st.switch_page("pages/3_🚜_Farm_Management.py")
        if st.button("📊 Model Performance", use_container_width=True):
            st.switch_page("pages/5_📊_ML_Model_Performance.py")

    with c_btn2:
        if st.button("📅 View Schedule", use_container_width=True):
            st.switch_page("pages/2_📅_Application_Schedule.py")
        if st.button("💬 AI Assistant", use_container_width=True):
            st.switch_page("pages/4_💬_AI_Agricultural_Assistant.py")
        if st.button("📜 History Log", use_container_width=True):
            st.switch_page("pages/6_📜_Recommendation_History.py")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Recent Recommendations & Activity
# -------------------------------------------------------------
st.markdown("### 🕒 Recent Activity")

if not df_recs.empty:
    recent_recs = df_recs.head(5)[["farmer_name", "crop_type", "soil_type", "recommended_fertilizer", "confidence", "created_at"]]
    recent_recs.columns = ["Farmer", "Crop", "Soil", "Recommended Fertilizer", "Confidence", "Date"]
    # Format confidence
    recent_recs["Confidence"] = recent_recs["Confidence"].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
    st.dataframe(recent_recs, use_container_width=True, hide_index=True)
else:
    st.info("No fertilizer recommendations generated yet. Head over to the **Fertilizer Recommendation** page to run your first prediction!")

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Fertilizer Recommendation Agent | B.Tech Data Science Mini Project | Built with Python, Scikit-learn, XGBoost & Streamlit")
