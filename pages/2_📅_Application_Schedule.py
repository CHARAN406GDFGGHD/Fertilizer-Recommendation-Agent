"""
Page 2: Fertilizer Application Schedule
Module 2 of Phase 2
Generates stage-wise split application schedules, tracks task completion status in SQLite,
and provides clear safety and agronomic advisories.
"""

from datetime import datetime, date
import streamlit as st
import pandas as pd

from src.ui_helpers import render_header, render_sidebar, apply_custom_css
from src.schedule import generate_schedule, save_schedule_to_db, get_safety_precautions
from src.database import get_all_schedules, update_schedule_task_status, delete_schedule_task

st.set_page_config(
    page_title="Application Schedule | AgriAgent",
    page_icon="📅",
    layout="wide"
)

render_sidebar()
render_header(
    title="Fertilizer Application Schedule & Task Manager",
    subtitle="Generate stage-wise fertilization timelines, mark stages completed, and ensure safe nutrient stewardship.",
    icon="📅"
)

# Crops & Fertilizers list
CROPS_LIST = [
    "Paddy", "Wheat", "Sugarcane", "Cotton", "Maize",
    "Pulses", "Ground Nuts", "Millets", "Barley", "Tobacco", "Oil seeds"
]
FERTILIZERS_LIST = ["Urea", "DAP", "14-35-14", "28-28", "20-20", "17-17-17", "10-26-26"]

# Default values from session state if coming from recommendation page
default_crop = st.session_state.get("sched_crop", "Paddy")
if default_crop not in CROPS_LIST:
    default_crop = "Paddy"
default_fert = st.session_state.get("sched_fert", "Urea")
if default_fert not in FERTILIZERS_LIST:
    default_fert = "Urea"

# Disclaimer Banner
st.markdown("""
<div style="background-color: #FFF3CD; border-left: 4px solid #FFC107; padding: 12px 18px; border-radius: 6px; margin-bottom: 20px;">
    <b>⚠️ Important Agronomic Notice:</b>
    <p style="margin: 4px 0 0 0; color: #856404; font-size: 0.9rem;">
        Application schedules provided here are <b>general agronomic guidance</b> adapted to standard physiological growth stages. 
        Final application doses must be verified using certified laboratory soil test reports (STCR approach) and recommendations from your local State Agricultural Extension Officer. 
        This tool does not prescribe arbitrary scientific application rates.
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs: 1. Generate New Schedule, 2. Manage Active Schedules, 3. Safety Precautions
tab_gen, tab_manage, tab_safety = st.tabs(["✨ Generate New Schedule", "📋 Manage Active Tasks", "🛡️ Safety Precautions"])

# -------------------------------------------------------------
# TAB 1: Generate New Schedule
# -------------------------------------------------------------
with tab_gen:
    with st.form("generate_schedule_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            crop_name = st.selectbox("Select Crop", options=CROPS_LIST, index=CROPS_LIST.index(default_crop))
        with col2:
            fertilizer_name = st.selectbox("Recommended Fertilizer", options=FERTILIZERS_LIST, index=FERTILIZERS_LIST.index(default_fert))
        with col3:
            planting_date = st.date_input("Sowing / Planting Date", value=date.today())

        farm_name = st.text_input("Farm / Parcel Label", value="North Field Plot 1")
        submit_gen = st.form_submit_button("📅 Generate & Save Application Schedule", use_container_width=True)

    if submit_gen:
        tasks = generate_schedule(crop_name, planting_date, fertilizer_name, farm_name)
        task_ids = save_schedule_to_db(tasks)
        st.success(f"✅ Generated and registered {len(task_ids)} stage-wise application tasks into your database for {crop_name}!")
        st.rerun()

# -------------------------------------------------------------
# TAB 2: Manage Active Schedules
# -------------------------------------------------------------
with tab_manage:
    st.markdown("### 📋 Scheduled Fertilizer Tasks")
    df_schedules = get_all_schedules()

    if df_schedules.empty:
        st.info("No active schedule tasks found. Use the 'Generate New Schedule' tab above to create one!")
    else:
        # Filter options
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            crops_in_db = ["All Crops"] + sorted(df_schedules["crop_name"].unique().tolist())
            selected_filter = st.selectbox("Filter by Crop", options=crops_in_db)

        if selected_filter != "All Crops":
            df_filtered = df_schedules[df_schedules["crop_name"] == selected_filter]
        else:
            df_filtered = df_schedules

        st.caption(f"Showing {len(df_filtered)} tasks.")

        # Render tasks as cards with interactive status toggles
        for _, row in df_filtered.iterrows():
            task_id = int(row["id"])
            is_completed = (row["status"] == "Completed")
            badge_class = "status-completed" if is_completed else "status-pending"
            badge_text = "COMPLETED" if is_completed else "PENDING"

            with st.container():
                st.markdown(f"""
                <div class="agri-card" style="padding: 16px 20px; border-left: 5px solid {'#2E7D32' if is_completed else '#FFA000'};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #1B5E20;">{row['stage_name']}</h4>
                        <span class="status-badge {badge_class}">{badge_text}</span>
                    </div>
                    <div style="color: #4A5568; margin-top: 6px; font-size: 0.95rem;">
                        <b>Crop:</b> {row['crop_name']} | <b>Fertilizer:</b> {row['fertilizer_name']} | <b>Target Date:</b> {row['target_date']} | <b>Farm:</b> {row['farm_name']}
                    </div>
                    <div style="background-color: #F8FAF9; padding: 10px; border-radius: 6px; margin: 10px 0; font-size: 0.88rem; color: #2D3748;">
                        📌 <b>Method:</b> {row['application_method']}<br>
                        🧪 <b>Dosage Guidance:</b> {row['recommended_dose_desc']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                act_col1, act_col2, act_col3 = st.columns([1, 1, 3])
                with act_col1:
                    if not is_completed:
                        if st.button("Mark Completed", key=f"comp_{task_id}"):
                            today_str = date.today().strftime("%Y-%m-%d")
                            update_schedule_task_status(task_id, "Completed", completion_date=today_str)
                            st.success(f"Task #{task_id} marked as completed!")
                            st.rerun()
                    else:
                        if st.button("Mark Pending", key=f"pend_{task_id}"):
                            update_schedule_task_status(task_id, "Pending", completion_date=None)
                            st.rerun()
                with act_col2:
                    if st.button("Delete Task", key=f"del_{task_id}"):
                        delete_schedule_task(task_id)
                        st.warning(f"Task #{task_id} deleted!")
                        st.rerun()

# -------------------------------------------------------------
# TAB 3: Safety Precautions
# -------------------------------------------------------------
with tab_safety:
    st.markdown("### 🛡️ Environmental Stewardship & Farmer Safety")
    precautions = get_safety_precautions()
    for p in precautions:
        st.markdown(f"""
        <div style="display: flex; gap: 12px; align-items: flex-start; margin-bottom: 12px;">
            <div style="font-size: 1.3rem;">🌱</div>
            <div style="color: #2D3748; line-height: 1.5;">{p}</div>
        </div>
        """, unsafe_allow_html=True)
