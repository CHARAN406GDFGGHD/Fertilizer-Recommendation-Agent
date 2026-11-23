"""
Page 6: Recommendation History
Module 5 of Phase 2
Stores, searches, reviews, and deletes historical fertilizer recommendations.
Backed by SQLite persistent storage.
"""

import streamlit as st
import pandas as pd

from src.ui_helpers import render_header, render_sidebar, apply_custom_css
from src.database import (
    get_all_recommendations,
    delete_recommendation_history,
    clear_all_recommendations
)

st.set_page_config(
    page_title="Recommendation History | AgriAgent",
    page_icon="📜",
    layout="wide"
)

render_sidebar()
render_header(
    title="Recommendation History & Audit Log",
    subtitle="Review past machine learning fertilizer predictions, input parameters, and confidence scores.",
    icon="📜"
)

df_recs = get_all_recommendations()

if df_recs.empty:
    st.info("No recommendation history found. Generate recommendations from the 'Fertilizer Recommendation' page to log entries.")
else:
    # Summary Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Recommendations", len(df_recs))
    top_fert = df_recs["recommended_fertilizer"].mode()[0] if not df_recs.empty else "N/A"
    col2.metric("Most Frequent Fertilizer", top_fert)
    top_crop = df_recs["crop_type"].mode()[0] if not df_recs.empty else "N/A"
    col3.metric("Most Frequent Crop", top_crop)

    st.markdown("---")

    # Search & Filter controls
    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        crop_options = ["All Crops"] + sorted(df_recs["crop_type"].dropna().unique().tolist())
        sel_crop = st.selectbox("Filter by Crop", options=crop_options)
    with col_s2:
        fert_options = ["All Fertilizers"] + sorted(df_recs["recommended_fertilizer"].dropna().unique().tolist())
        sel_fert = st.selectbox("Filter by Fertilizer", options=fert_options)

    filtered_df = df_recs.copy()
    if sel_crop != "All Crops":
        filtered_df = filtered_df[filtered_df["crop_type"] == sel_crop]
    if sel_fert != "All Fertilizers":
        filtered_df = filtered_df[filtered_df["recommended_fertilizer"] == sel_fert]

    # Display Table
    table_view = filtered_df[[
        "id", "farmer_name", "crop_type", "soil_type",
        "temperature", "humidity", "moisture",
        "nitrogen", "potassium", "phosphorous",
        "recommended_fertilizer", "confidence", "created_at"
    ]].copy()
    table_view.columns = [
        "ID", "Farmer", "Crop", "Soil",
        "Temp (°C)", "Humidity (%)", "Moisture (%)",
        "N", "K", "P",
        "Recommended Fertilizer", "Confidence", "Date Logged"
    ]
    table_view["Confidence"] = table_view["Confidence"].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")

    st.dataframe(table_view, use_container_width=True, hide_index=True)

    # Detailed Inspection Cards
    st.markdown("### 🔍 Detailed Inspection & Action")
    rec_options = [
        f"Record #{r['id']} - {r['farmer_name']} | {r['crop_type']} ({r['recommended_fertilizer']}) on {r['created_at']}"
        for _, r in filtered_df.iterrows()
    ]

    if rec_options:
        selected_rec = st.selectbox("Select Record to Inspect / Delete", options=rec_options)
        sel_id = int(selected_rec.split(" - ")[0].replace("Record #", ""))
        row_detail = filtered_df[filtered_df["id"] == sel_id].iloc[0]

        with st.container():
            st.markdown(f"""
            <div class="agri-card">
                <div style="display: flex; justify-content: space-between;">
                    <h3 style="color: #1B5E20; margin: 0;">🌱 {row_detail['recommended_fertilizer']}</h3>
                    <span style="color: #718096; font-size: 0.9rem;">Logged: {row_detail['created_at']}</span>
                </div>
                <div style="margin: 10px 0; color: #2D3748;">
                    <b>Farmer:</b> {row_detail['farmer_name']} | <b>Crop:</b> {row_detail['crop_type']} | <b>Soil:</b> {row_detail['soil_type']} | <b>Model:</b> {row_detail['model_used'] or 'XGBoost'}
                </div>
                <div style="background-color: #F8FAF9; padding: 12px; border-radius: 8px; margin: 10px 0;">
                    <b>Environmental & Nutrient Factors:</b><br>
                    • Temperature: {row_detail['temperature']} °C &nbsp;&nbsp;|&nbsp;&nbsp; Humidity: {row_detail['humidity']} % &nbsp;&nbsp;|&nbsp;&nbsp; Soil Moisture: {row_detail['moisture']} %<br>
                    • Nitrogen (N): {row_detail['nitrogen']} &nbsp;&nbsp;|&nbsp;&nbsp; Phosphorous (P): {row_detail['phosphorous']} &nbsp;&nbsp;|&nbsp;&nbsp; Potassium (K): {row_detail['potassium']}
                </div>
                <div style="font-size: 0.92rem; color: #4A5568; line-height: 1.5;">
                    <b>Agronomic Rationale:</b> {row_detail['explanation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            del_col1, del_col2 = st.columns([1, 3])
            with del_col1:
                if st.button(f"🗑️ Delete Record #{sel_id}", type="secondary"):
                    delete_recommendation_history(sel_id)
                    st.warning(f"Record #{sel_id} removed.")
                    st.rerun()

    # Bulk Export & Clear
    st.markdown("---")
    foot_col1, foot_col2 = st.columns([1, 1])
    with foot_col1:
        csv_data = df_recs.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Full History as CSV",
            data=csv_data,
            file_name="recommendation_history.csv",
            mime="text/csv"
        )
    with foot_col2:
        if st.button("⚠️ Clear Entire History Log", help="Permanently deletes all recommendation history records"):
            clear_all_recommendations()
            st.warning("All recommendation history records cleared!")
            st.rerun()
