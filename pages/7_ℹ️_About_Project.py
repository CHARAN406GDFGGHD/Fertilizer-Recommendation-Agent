"""
Page 7: About Project
Complete project documentation, academic problem statement, system architecture,
and technological framework.
"""

import streamlit as st
from src.ui_helpers import render_header, render_sidebar, apply_custom_css

st.set_page_config(
    page_title="About Project | AgriAgent",
    page_icon="ℹ️",
    layout="wide"
)

render_sidebar()
render_header(
    title="About This Project",
    subtitle="Intelligent Fertilizer Recommendation Agent Using Machine Learning — B.Tech Data Science Mini Project",
    icon="ℹ️"
)

st.markdown("""
<div class="agri-card">
    <h2 style="color: #1B5E20; margin-top: 0;">🎓 Academic Project Synopsis</h2>
    <p style="color: #2D3748; line-height: 1.7; font-size: 1.02rem;">
        <b>Project Title:</b> Intelligent Fertilizer Recommendation Agent Using Machine Learning<br>
        <b>Domain:</b> Agricultural Artificial Intelligence & Data Science<br>
        <b>Target Level:</b> B.Tech / Undergraduate Data Science & Machine Learning Capstone
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="agri-card">
        <h3 style="color: #1B5E20; margin-top: 0;">🎯 Problem Statement</h3>
        <p style="color: #4A5568; line-height: 1.6;">
            Excessive and unbalanced chemical fertilizer usage is one of the leading causes of topsoil acidification, 
            nutrient runoff into freshwater ecosystems, and escalated cultivation costs for smallholder farmers. 
            Traditional agricultural advice is often generic, ignoring parcel-specific soil nutrient levels (N-P-K), 
            local temperature, moisture, and crop-specific nutrient uptake curves. 
            There is an acute need for accessible, intelligent, and offline-capable decision support agents that provide 
            precise fertilizer recommendations and stage-wise application schedules.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="agri-card">
        <h3 style="color: #1B5E20; margin-top: 0;">📚 Project Objectives</h3>
        <ol style="color: #4A5568; line-height: 1.6; padding-left: 20px;">
            <li>Develop and benchmark <b>five supervised machine learning classifiers</b> on agricultural dataset benchmarks.</li>
            <li>Eliminate data leakage using disciplined <code>ColumnTransformer</code> preprocessing fitted exclusively on training sets.</li>
            <li>Construct an intelligent agent offering tailored fertilizer recommendations with confidence scoring and agronomic rationales.</li>
            <li>Generate stage-wise split-application schedules with reminders and safety guidelines.</li>
            <li>Deliver an offline AI assistant that explains agricultural science without expensive third-party cloud API costs.</li>
            <li>Provide persistent SQLite data management for farm parcels and recommendation audit trails.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="agri-card">
        <h3 style="color: #1B5E20; margin-top: 0;">🔬 Machine Learning Algorithms</h3>
        <ul style="color: #4A5568; line-height: 1.6; padding-left: 20px;">
            <li><b>1. Random Forest Classifier:</b> Ensemble of decision trees with bagging; handles feature non-linearities smoothly.</li>
            <li><b>2. XGBoost Classifier:</b> Gradient-boosted decision trees; optimizes classification loss with regularized objective.</li>
            <li><b>3. Support Vector Machine (SVM):</b> RBF kernel with calibrated probability estimates; maps features into high-dimensional hyperplane.</li>
            <li><b>4. K-Nearest Neighbors (KNN):</b> Instance-based non-parametric classifier using distance-weighted nearest neighbors.</li>
            <li><b>5. Multi-Layer Perceptron (MLP):</b> Feed-forward artificial neural network with backpropagation and Adam optimizer.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="agri-card">
        <h3 style="color: #1B5E20; margin-top: 0;">🛠️ Technology Stack</h3>
        <p style="color: #4A5568; line-height: 1.6;">
            • <b>Language:</b> Python 3.10+<br>
            • <b>Data Processing:</b> Pandas, NumPy<br>
            • <b>Machine Learning:</b> Scikit-learn, XGBoost, Joblib<br>
            • <b>Visualization:</b> Matplotlib, Seaborn<br>
            • <b>Web Dashboard:</b> Streamlit<br>
            • <b>Database:</b> SQLite3 (Zero-setup local persistence)<br>
            • <b>Dataset Benchmark:</b> Kaggle <code>gdabhishek/fertilizer-prediction</code>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="agri-card">
    <h3 style="color: #1B5E20; margin-top: 0;">🏗️ System Architecture</h3>
    <div style="background-color: #F8FAF9; padding: 16px; border-radius: 8px; font-family: monospace; font-size: 0.88rem; line-height: 1.7; color: #1A202C;">
        [Kaggle Dataset: Fertilizer Prediction.csv]<br>
        &nbsp;&nbsp;&nbsp;&nbsp;│<br>
        &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
        [Data Preprocessing (StandardScaler + OneHotEncoder fitted ONLY on Train)]<br>
        &nbsp;&nbsp;&nbsp;&nbsp;│<br>
        &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
        [5-Model Benchmark: RF, XGBoost, SVM, KNN, MLP (Stratified 5-Fold CV)]<br>
        &nbsp;&nbsp;&nbsp;&nbsp;│<br>
        &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
        [Model Artifacts (Joblib) & Evaluation Metrics (JSON / CSV / PNG)]<br>
        &nbsp;&nbsp;&nbsp;&nbsp;│<br>
        &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
        ┌────────────────────────────────────────────────────────┐<br>
        │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;STREAMLIT DASHBOARD SUITE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
        ├──────────────────────┬─────────────────────────────────┤<br>
        │ 1. Home Dashboard&nbsp;&nbsp;&nbsp;│ 5. AI Agricultural Assistant&nbsp;&nbsp;&nbsp;&nbsp;│<br>
        │ 2. Fertilizer Rec.&nbsp;&nbsp;&nbsp;│ 6. Model Performance Benchmark&nbsp;&nbsp;│<br>
        │ 3. App. Schedule&nbsp;&nbsp;&nbsp;&nbsp;│ 7. Recommendation History&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
        │ 4. Farm Parcel DB&nbsp;&nbsp;&nbsp;│ 8. SQLite Persistent Storage&nbsp;&nbsp;&nbsp;&nbsp;│<br>
        └──────────────────────┴─────────────────────────────────┘
    </div>
</div>
""", unsafe_allow_html=True)
