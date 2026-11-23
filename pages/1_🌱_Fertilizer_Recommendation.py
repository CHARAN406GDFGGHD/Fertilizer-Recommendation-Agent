"""
Page 1: Fertilizer Recommendation
Module 1 of Phase 2
Accepts soil, climate, and crop features, runs inference on the trained ML model,
displays confidence, agronomic explanation, and saves the recommendation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.ui_helpers import render_header, render_sidebar, apply_custom_css
from src.prediction import predictor
from src.database import add_recommendation_history, get_all_farm_records

st.set_page_config(
    page_title="Fertilizer Recommendation | AgriAgent",
    page_icon="🌱",
    layout="wide"
)

render_sidebar()
render_header(
    title="Fertilizer Recommendation Engine",
    subtitle="Enter soil nutrient status, climatic factors, and crop details to generate optimal fertilizer recommendations.",
    icon="🌱"
)

# Supported options from dataset
SOIL_TYPES = ["Loamy", "Sandy", "Clayey", "Black", "Red"]
CROP_TYPES = [
    "Sugarcane", "Cotton", "Millets", "Paddy", "Pulses",
    "Wheat", "Tobacco", "Barley", "Oil seeds", "Ground Nuts", "Maize"
]

# Quick Autofill Presets based on typical agricultural scenarios
with st.expander("💡 Load Sample Agricultural Scenarios (Quick Autofill)", expanded=False):
    preset_cols = st.columns(4)
    if preset_cols[0].button("🌾 Paddy in Clayey Soil"):
        st.session_state["p_temp"] = 28
        st.session_state["p_hum"] = 65
        st.session_state["p_moist"] = 55
        st.session_state["p_soil"] = "Clayey"
        st.session_state["p_crop"] = "Paddy"
        st.session_state["p_n"] = 12
        st.session_state["p_k"] = 0
        st.session_state["p_p"] = 38
        st.rerun()

    if preset_cols[1].button("🌿 Sugarcane in Black Soil"):
        st.session_state["p_temp"] = 32
        st.session_state["p_hum"] = 58
        st.session_state["p_moist"] = 42
        st.session_state["p_soil"] = "Black"
        st.session_state["p_crop"] = "Sugarcane"
        st.session_state["p_n"] = 38
        st.session_state["p_k"] = 0
        st.session_state["p_p"] = 0
        st.rerun()

    if preset_cols[2].button("🥜 Ground Nuts in Red Soil"):
        st.session_state["p_temp"] = 29
        st.session_state["p_hum"] = 54
        st.session_state["p_moist"] = 35
        st.session_state["p_soil"] = "Red"
        st.session_state["p_crop"] = "Ground Nuts"
        st.session_state["p_n"] = 10
        st.session_state["p_k"] = 18
        st.session_state["p_p"] = 25
        st.rerun()

    if preset_cols[3].button("🌽 Maize in Sandy Soil"):
        st.session_state["p_temp"] = 30
        st.session_state["p_hum"] = 60
        st.session_state["p_moist"] = 32
        st.session_state["p_soil"] = "Sandy"
        st.session_state["p_crop"] = "Maize"
        st.session_state["p_n"] = 22
        st.session_state["p_k"] = 0
        st.session_state["p_p"] = 22
        st.rerun()

# Input Form
with st.form("recommendation_form"):
    st.markdown("#### 1. Environmental & Soil Physical Conditions")
    col_env1, col_env2, col_env3 = st.columns(3)
    with col_env1:
        temp = st.number_input(
            "Temperature (°C)",
            min_value=10, max_value=50,
            value=st.session_state.get("p_temp", 28),
            help="Average ambient temperature in Celsius."
        )
    with col_env2:
        humidity = st.number_input(
            "Humidity (%)",
            min_value=10, max_value=100,
            value=st.session_state.get("p_hum", 55),
            help="Relative atmospheric humidity percentage."
        )
    with col_env3:
        moisture = st.number_input(
            "Soil Moisture (%)",
            min_value=5, max_value=100,
            value=st.session_state.get("p_moist", 40),
            help="Volumetric soil moisture percentage."
        )

    st.markdown("#### 2. Crop & Soil Classification")
    col_cs1, col_cs2 = st.columns(2)
    with col_cs1:
        soil_type = st.selectbox(
            "Soil Type",
            options=SOIL_TYPES,
            index=SOIL_TYPES.index(st.session_state.get("p_soil", "Loamy"))
        )
    with col_cs2:
        crop_type = st.selectbox(
            "Crop Type",
            options=CROP_TYPES,
            index=CROP_TYPES.index(st.session_state.get("p_crop", "Paddy"))
        )

    st.markdown("#### 3. Primary Soil Nutrients (NPK Ratios)")
    col_npk1, col_npk2, col_npk3 = st.columns(3)
    with col_npk1:
        nitrogen = st.number_input(
            "Nitrogen (N Ratio)",
            min_value=0, max_value=100,
            value=st.session_state.get("p_n", 20),
            help="Available Nitrogen index from soil test."
        )
    with col_npk2:
        phosphorous = st.number_input(
            "Phosphorous (P Ratio)",
            min_value=0, max_value=100,
            value=st.session_state.get("p_p", 20),
            help="Available Phosphorus index from soil test."
        )
    with col_npk3:
        potassium = st.number_input(
            "Potassium (K Ratio)",
            min_value=0, max_value=100,
            value=st.session_state.get("p_k", 5),
            help="Available Potassium index from soil test."
        )

    st.markdown("#### 4. Model Selection & Metadata")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        farmer_name = st.text_input("Farmer / Grower Name (Optional)", value="Farmer John")
    with col_m2:
        available_models = predictor.get_available_models()
        selected_model = st.selectbox("Select ML Model to Infer", options=available_models, index=0)

    submit_button = st.form_submit_button("🌱 Predict Recommended Fertilizer", use_container_width=True)

# Processing prediction
if submit_button:
    input_data = {
        "Temparature": temp,
        "Humidity": humidity,
        "Moisture": moisture,
        "Soil Type": soil_type,
        "Crop Type": crop_type,
        "Nitrogen": nitrogen,
        "Potassium": potassium,
        "Phosphorous": phosphorous
    }

    with st.spinner("Analyzing soil dynamics and running machine learning inference..."):
        try:
            result = predictor.predict(input_data, model_name=selected_model)
            rec_fertilizer = result["recommended_fertilizer"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]
            explanation = result["explanation"]
            nutrients = result["nutrient_balance"]

            # Save in session for downstream use
            st.session_state["last_rec"] = {
                "fertilizer": rec_fertilizer,
                "crop": crop_type,
                "soil": soil_type,
                "nitrogen": nitrogen,
                "phosphorous": phosphorous,
                "potassium": potassium,
                "confidence": confidence,
                "explanation": explanation
            }

            # Prominent Recommendation Display
            st.markdown(f"""
            <div class="rec-banner">
                <div style="font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.9;">
                    Machine Learning Recommendation
                </div>
                <h1 style="font-size: 2.8rem; margin: 8px 0; font-weight: 800;">
                    {rec_fertilizer}
                </h1>
                <p style="font-size: 1.05rem; opacity: 0.95;">
                    Optimal fertilizer formulation tailored for <b>{crop_type}</b> cultivated in <b>{soil_type}</b> soil.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Details Columns
            res_col1, res_col2 = st.columns([1.2, 0.8])

            with res_col1:
                st.markdown("### 📋 Agronomic Explanation & Analysis")
                st.markdown(f"""
                <div class="agri-card">
                    {explanation}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 🧪 Nutrient Status Evaluation")
                nutr_cols = st.columns(3)
                nutr_cols[0].metric("Nitrogen (N)", f"{nitrogen}", f"Status: {nutrients['nitrogen_status']}")
                nutr_cols[1].metric("Phosphorous (P)", f"{phosphorous}", f"Status: {nutrients['phosphorous_status']}")
                nutr_cols[2].metric("Potassium (K)", f"{potassium}", f"Status: {nutrients['potassium_status']}")

            with res_col2:
                st.markdown("### 🎯 Model Confidence")
                if confidence is not None:
                    st.metric(label=f"Inference Confidence ({selected_model})", value=f"{confidence * 100:.1f}%")
                    st.progress(float(confidence))

                    # Top probabilities chart
                    if probabilities:
                        st.markdown("##### Class Probability Distribution")
                        prob_df = pd.DataFrame(list(probabilities.items()), columns=["Fertilizer", "Probability"])
                        prob_df = prob_df.sort_values(by="Probability", ascending=True)

                        fig, ax = plt.subplots(figsize=(5, 3.2))
                        bars = ax.barh(prob_df["Fertilizer"], prob_df["Probability"] * 100, color="#2E7D32")
                        ax.set_xlim(0, 105)
                        ax.set_xlabel("Probability (%)", fontsize=8)
                        for bar in bars:
                            width = bar.get_width()
                            if width > 1:
                                ax.text(width + 2, bar.get_y() + bar.get_height()/2, f"{width:.1f}%",
                                        va="center", fontsize=8, color="#1B5E20", fontweight="bold")
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                else:
                    st.info("Probability estimates not supported by this model variant.")

            # Save to Database Section
            st.markdown("---")
            save_col1, save_col2 = st.columns([1, 1])

            with save_col1:
                rec_id = add_recommendation_history(
                    farmer_name=farmer_name,
                    crop_type=crop_type,
                    soil_type=soil_type,
                    temperature=temp,
                    humidity=humidity,
                    moisture=moisture,
                    nitrogen=nitrogen,
                    potassium=potassium,
                    phosphorous=phosphorous,
                    recommended_fertilizer=rec_fertilizer,
                    confidence=confidence,
                    model_used=selected_model,
                    explanation=explanation
                )
                st.success(f"✅ Recommendation saved successfully to database (Record #{rec_id})!")

            with save_col2:
                if st.button("📅 Generate Application Schedule for This Crop", use_container_width=True):
                    st.session_state["sched_crop"] = crop_type
                    st.session_state["sched_fert"] = rec_fertilizer
                    st.switch_page("pages/2_📅_Application_Schedule.py")

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
