"""
Page 3: Farm Management
Module 3 of Phase 2
Provides full CRUD (Create, Read, Update, Delete) management for farm parcels,
soil profiles, crops, fertilizer histories, and operational notes.
Backed by SQLite persistent storage.
"""

from datetime import date
import streamlit as st
import pandas as pd

from src.ui_helpers import render_header, render_sidebar, apply_custom_css
from src.database import (
    add_farm_record,
    get_all_farm_records,
    get_farm_record_by_id,
    update_farm_record,
    delete_farm_record
)

st.set_page_config(
    page_title="Farm Management | AgriAgent",
    page_icon="🚜",
    layout="wide"
)

render_sidebar()
render_header(
    title="Farm Parcel & Crop Record Management",
    subtitle="Maintain complete records of farmer parcels, soil conditions, fertilizer treatments, and field notes.",
    icon="🚜"
)

SOIL_OPTIONS = ["Loamy", "Sandy", "Clayey", "Black", "Red"]
CROP_OPTIONS = [
    "Sugarcane", "Cotton", "Millets", "Paddy", "Pulses",
    "Wheat", "Tobacco", "Barley", "Oil seeds", "Ground Nuts", "Maize"
]
FERTILIZER_OPTIONS = ["Urea", "DAP", "14-35-14", "28-28", "20-20", "17-17-17", "10-26-26", "Organic Compost", "Other"]

tab_view, tab_add, tab_edit = st.tabs(["📋 View Farm Records", "➕ Add New Parcel Record", "✏️ Edit / Delete Records"])

# -------------------------------------------------------------
# TAB 1: View Farm Records
# -------------------------------------------------------------
with tab_view:
    df_farms = get_all_farm_records()

    if df_farms.empty:
        st.info("No farm parcel records currently exist in the database. Use the 'Add New Parcel Record' tab to create your first entry.")
    else:
        st.markdown(f"### Total Tracked Parcels: **{len(df_farms)}**")

        # Search filter
        search_term = st.text_input("🔍 Search by Farmer, Farm Name, or Crop", placeholder="Type to filter...")
        if search_term:
            filtered = df_farms[
                df_farms["farmer_name"].str.contains(search_term, case=False, na=False) |
                df_farms["farm_name"].str.contains(search_term, case=False, na=False) |
                df_farms["crop_name"].str.contains(search_term, case=False, na=False)
            ]
        else:
            filtered = df_farms

        # Display Dataframe
        display_df = filtered[[
            "id", "farmer_name", "farm_name", "crop_name", "soil_type",
            "fertilizer_used", "application_date", "farm_area_acres", "notes"
        ]].copy()
        display_df.columns = ["ID", "Farmer", "Farm Name", "Crop", "Soil Type", "Fertilizer", "Application Date", "Area (Acres)", "Notes"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download CSV
        csv_data = df_farms.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Farm Records as CSV",
            data=csv_data,
            file_name="farm_management_records.csv",
            mime="text/csv"
        )

# -------------------------------------------------------------
# TAB 2: Add New Parcel Record
# -------------------------------------------------------------
with tab_add:
    st.markdown("### ➕ Register New Farm Parcel")
    with st.form("add_farm_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_farmer = st.text_input("Farmer Full Name", placeholder="e.g. Ramesh Patel")
            new_farm_name = st.text_input("Farm / Field Identifier", placeholder="e.g. Green Acres - Block B")
            new_crop = st.selectbox("Crop Cultivated", options=CROP_OPTIONS)
            new_soil = st.selectbox("Predominant Soil Type", options=SOIL_OPTIONS)

        with col2:
            new_area = st.number_input("Field Area (Acres)", min_value=0.1, max_value=1000.0, value=2.5, step=0.5)
            new_fert = st.selectbox("Fertilizer Applied / Planned", options=FERTILIZER_OPTIONS)
            new_date = st.date_input("Application / Treatment Date", value=date.today())
            new_notes = st.text_area("Operational Field Notes", placeholder="e.g. Pre-monsoon soil test completed; added 5 tons farmyard manure.")

        submit_add = st.form_submit_button("💾 Save Farm Record to Database", use_container_width=True)

    if submit_add:
        if not new_farmer.strip() or not new_farm_name.strip():
            st.error("Farmer Name and Farm Name are mandatory fields.")
        else:
            record_id = add_farm_record(
                farmer_name=new_farmer.strip(),
                farm_name=new_farm_name.strip(),
                crop_name=new_crop,
                soil_type=new_soil,
                fertilizer_used=new_fert,
                application_date=new_date.strftime("%Y-%m-%d"),
                farm_area_acres=new_area,
                notes=new_notes.strip()
            )
            st.success(f"✅ Farm parcel successfully registered with ID #{record_id}!")
            st.rerun()

# -------------------------------------------------------------
# TAB 3: Edit or Delete Existing Records
# -------------------------------------------------------------
with tab_edit:
    st.markdown("### ✏️ Edit or Remove Existing Record")
    df_farms = get_all_farm_records()

    if df_farms.empty:
        st.info("No records available to edit.")
    else:
        record_options = [
            f"ID #{row['id']} - {row['farmer_name']} ({row['farm_name']}, {row['crop_name']})"
            for _, row in df_farms.iterrows()
        ]
        selected_record_label = st.selectbox("Select Record to Modify", options=record_options)
        selected_id = int(selected_record_label.split(" - ")[0].replace("ID #", ""))

        record_data = get_farm_record_by_id(selected_id)

        if record_data:
            with st.form("edit_farm_form"):
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    edit_farmer = st.text_input("Farmer Name", value=record_data["farmer_name"])
                    edit_farm = st.text_input("Farm Name", value=record_data["farm_name"])
                    crop_idx = CROP_OPTIONS.index(record_data["crop_name"]) if record_data["crop_name"] in CROP_OPTIONS else 0
                    edit_crop = st.selectbox("Crop", options=CROP_OPTIONS, index=crop_idx)
                    soil_idx = SOIL_OPTIONS.index(record_data["soil_type"]) if record_data["soil_type"] in SOIL_OPTIONS else 0
                    edit_soil = st.selectbox("Soil Type", options=SOIL_OPTIONS, index=soil_idx)

                with col_e2:
                    edit_area = st.number_input("Area (Acres)", min_value=0.1, max_value=1000.0, value=float(record_data["farm_area_acres"] or 1.0))
                    fert_idx = FERTILIZER_OPTIONS.index(record_data["fertilizer_used"]) if record_data["fertilizer_used"] in FERTILIZER_OPTIONS else 0
                    edit_fert = st.selectbox("Fertilizer", options=FERTILIZER_OPTIONS, index=fert_idx)
                    edit_date = st.text_input("Application Date (YYYY-MM-DD)", value=record_data["application_date"] or "")
                    edit_notes = st.text_area("Notes", value=record_data["notes"] or "")

                btn_col1, btn_col2 = st.columns([1, 1])
                with btn_col1:
                    submit_update = st.form_submit_button("💾 Update Changes", use_container_width=True)

            if submit_update:
                update_farm_record(
                    record_id=selected_id,
                    farmer_name=edit_farmer,
                    farm_name=edit_farm,
                    crop_name=edit_crop,
                    soil_type=edit_soil,
                    fertilizer_used=edit_fert,
                    application_date=edit_date,
                    farm_area_acres=edit_area,
                    notes=edit_notes
                )
                st.success(f"Record #{selected_id} updated successfully!")
                st.rerun()

            st.markdown("---")
            st.markdown("#### 🗑️ Danger Zone")
            if st.button(f"Delete Record #{selected_id} Permanently", type="secondary"):
                delete_farm_record(selected_id)
                st.warning(f"Record #{selected_id} has been permanently deleted.")
                st.rerun()
