"""
Page 5: Machine Learning Model Performance
Phase 1 Evaluation Dashboard
Presents dataset exploration, model comparison metrics across all 5 algorithms,
visual charts, confusion matrices, and detailed classification reports.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np

from src.ui_helpers import render_header, render_sidebar, apply_custom_css

st.set_page_config(
    page_title="Model Performance | AgriAgent",
    page_icon="📊",
    layout="wide"
)

render_sidebar()
render_header(
    title="Machine Learning Model Evaluation & Benchmark",
    subtitle="Rigorous empirical comparison of 5 classification algorithms on the Kaggle Fertilizer Prediction dataset.",
    icon="📊"
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load evaluation data
summary_path = os.path.join(RESULTS_DIR, "dataset_summary.json")
comp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
eval_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
chart_path = os.path.join(RESULTS_DIR, "model_comparison.png")
best_cm_path = os.path.join(RESULTS_DIR, "best_confusion_matrix.png")
all_cm_path = os.path.join(RESULTS_DIR, "all_confusion_matrices.png")

if not os.path.exists(comp_path) or not os.path.exists(eval_path):
    st.error("Model evaluation results not found. Please run `python -m src.train_models` first.")
    st.stop()

with open(summary_path, "r") as f:
    dataset_summary = json.load(f)

with open(eval_path, "r") as f:
    eval_results = json.load(f)

df_comp = pd.read_csv(comp_path)

tab_comp, tab_dataset, tab_matrices, tab_reports = st.tabs([
    "🏆 Model Comparison", "📁 Dataset Processing & EDA", "🎯 Confusion Matrices", "📄 Detailed Classification Reports"
])

# -------------------------------------------------------------
# TAB 1: Model Comparison
# -------------------------------------------------------------
with tab_comp:
    st.markdown("### 🏆 5-Model Performance Comparison")
    st.markdown("""
    All models were evaluated on an identical **Stratified Test Set (20% split, 20 samples)** 
    with preprocessing (StandardScaler and OneHotEncoder) fitted strictly on the training partition 
    to guarantee **zero data leakage**. Cross-Validation was performed using **5-Fold Stratified K-Fold**.
    """)

    # Format table for display
    display_comp = df_comp.copy()
    display_comp["Accuracy"] = display_comp["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
    display_comp["Precision (Weighted)"] = display_comp["Precision (Weighted)"].apply(lambda x: f"{x*100:.2f}%")
    display_comp["Recall (Weighted)"] = display_comp["Recall (Weighted)"].apply(lambda x: f"{x*100:.2f}%")
    display_comp["F1-Score (Weighted)"] = display_comp["F1-Score (Weighted)"].apply(lambda x: f"{x*100:.2f}%")
    display_comp["CV Accuracy (Mean)"] = display_comp["CV Accuracy (Mean)"].apply(lambda x: f"{x*100:.2f}%")
    display_comp["CV Accuracy (Std)"] = display_comp["CV Accuracy (Std)"].apply(lambda x: f"±{x*100:.2f}%")

    st.dataframe(display_comp, use_container_width=True, hide_index=True)

    # Best Model Callout
    best_row = df_comp.iloc[0]
    st.markdown(f"""
    <div style="background-color: #E8F5E9; border-left: 5px solid #2E7D32; padding: 16px 20px; border-radius: 8px; margin: 20px 0;">
        <h4 style="margin: 0; color: #1B5E20;">🌟 Top Ranked Algorithm: {best_row['Model']}</h4>
        <p style="margin: 6px 0 0 0; color: #2E7D32;">
            Selected as the production model due to superior test performance 
            (<b>Accuracy: {best_row['Accuracy']*100:.2f}%</b>, <b>F1-Score: {best_row['F1-Score (Weighted)']*100:.2f}%</b>) 
            and highest cross-validation stability (<b>CV Mean: {best_row['CV Accuracy (Mean)']*100:.2f}%</b>).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Chart
    if os.path.exists(chart_path):
        st.markdown("#### 📊 Comparative Accuracy & F1-Score Chart")
        st.image(chart_path, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: Dataset Processing & EDA
# -------------------------------------------------------------
with tab_dataset:
    st.markdown("### 📁 Dataset Overview & Exploratory Data Analysis")
    st.caption("Benchmark: Kaggle `gdabhishek/fertilizer-prediction`")

    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    col_d1.metric("Total Records", dataset_summary["rows"])
    col_d2.metric("Total Features", dataset_summary["columns"])
    col_d3.metric("Missing Values", sum(dataset_summary["missing_values"].values()))
    col_d4.metric("Duplicates", dataset_summary["duplicates"])

    st.markdown("#### Feature Schema & Types")
    dtypes_df = pd.DataFrame({
        "Column Name": dataset_summary["column_names"],
        "Data Type": [dataset_summary["data_types"][c] for c in dataset_summary["column_names"]],
        "Missing Values": [dataset_summary["missing_values"][c] for c in dataset_summary["column_names"]]
    })
    st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    col_tgt1, col_tgt2 = st.columns(2)
    with col_tgt1:
        st.markdown("#### Target Class Distribution (`Fertilizer Name`)")
        target_df = pd.DataFrame(
            list(dataset_summary["target_distribution"].items()),
            columns=["Fertilizer", "Count"]
        )
        st.dataframe(target_df, use_container_width=True, hide_index=True)

    with col_tgt2:
        st.markdown("#### Supported Categories")
        st.markdown(f"**Soil Types ({len(dataset_summary['unique_soil_types'])}):** " + ", ".join(dataset_summary["unique_soil_types"]))
        st.markdown(f"**Crop Types ({len(dataset_summary['unique_crop_types'])}):** " + ", ".join(dataset_summary["unique_crop_types"]))

# -------------------------------------------------------------
# TAB 3: Confusion Matrices
# -------------------------------------------------------------
with tab_matrices:
    st.markdown("### 🎯 Confusion Matrix Visualizations")
    st.markdown("Evaluates exact class-by-class true positives and misclassifications across test samples.")

    if os.path.exists(best_cm_path):
        st.markdown("#### Best Model Matrix")
        st.image(best_cm_path, width=650)

    st.markdown("---")
    if os.path.exists(all_cm_path):
        st.markdown("#### All 5 Models Comparative Matrices")
        st.image(all_cm_path, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: Detailed Classification Reports
# -------------------------------------------------------------
with tab_reports:
    st.markdown("### 📄 Classification Reports (Per Class Metrics)")
    selected_rep_model = st.selectbox("Select Model to Inspect", options=list(eval_results.keys()))

    if selected_rep_model in eval_results:
        rep_text = eval_results[selected_rep_model]["classification_report_text"]
        st.code(rep_text, language="text")

        st.markdown("##### Performance Breakdown Table")
        rep_dict = eval_results[selected_rep_model]["classification_report"]
        rows = []
        for k, v in rep_dict.items():
            if isinstance(v, dict):
                rows.append({
                    "Class / Average": k,
                    "Precision": f"{v.get('precision', 0)*100:.1f}%",
                    "Recall": f"{v.get('recall', 0)*100:.1f}%",
                    "F1-Score": f"{v.get('f1-score', 0)*100:.1f}%",
                    "Support": int(v.get('support', 0))
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
