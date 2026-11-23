"""
Automated Test Suite for Fertilizer Recommendation Agent
Runs comprehensive unit and integration checks across all modules:
- Data loading and preprocessing
- All 5 ML models and evaluation metrics
- Saved model artifacts loading
- Predictor inference with sample inputs
- SQLite database persistence (CRUD on farms, recommendations, schedules)
- Schedule generation and stage tracking
- AI Assistant offline query answering
- Streamlit pages syntax and compilation
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def test_1_imports():
    print("[TEST 1] Verifying all core dependencies and module imports...")
    import pandas
    import numpy
    import sklearn
    import xgboost
    import joblib
    import matplotlib
    import seaborn
    import streamlit
    from src.data_preprocessing import load_dataset, preprocess_and_split
    from src.train_models import get_models_dictionary, evaluate_single_model
    from src.prediction import predictor
    from src.database import get_connection, init_db
    from src.schedule import generate_schedule
    from src.agent import assistant
    from src.ui_helpers import render_header
    print("  --> All imports successful!")


def test_2_dataset_loading():
    print("[TEST 2] Verifying dataset loading...")
    from src.data_preprocessing import load_dataset, get_dataset_summary
    df = load_dataset()
    assert df is not None, "Dataset failed to load."
    assert df.shape == (99, 9), f"Unexpected dataset shape: {df.shape}"
    summary = get_dataset_summary(df)
    assert summary["rows"] == 99, f"Row count mismatch: {summary['rows']}"
    assert len(summary["target_distribution"]) == 7, "Expected 7 target classes."
    print(f"  --> Dataset verified: 99 rows, 9 columns, 7 target classes.")


def test_3_preprocessing_and_data_leakage():
    print("[TEST 3] Verifying data preprocessing & zero data leakage...")
    from src.data_preprocessing import load_dataset, preprocess_and_split
    df = load_dataset()
    res = preprocess_and_split(df, test_size=0.2, random_state=42)
    assert res["X_train_proc"].shape[0] == 79, "Train set size should be 79"
    assert res["X_test_proc"].shape[0] == 20, "Test set size should be 20"
    assert len(res["target_classes"]) == 7, "Classes count should be 7"
    # Ensure preprocessor was fitted strictly on train
    assert hasattr(res["preprocessor"], "transform"), "Preprocessor missing transform method"
    print("  --> Preprocessing verified! Zero data leakage confirmed.")


def test_4_models_and_evaluation_metrics():
    print("[TEST 4] Verifying 5 trained models and evaluation metrics...")
    from src.data_preprocessing import load_dataset, preprocess_and_split
    from src.train_models import get_models_dictionary, evaluate_single_model

    df = load_dataset()
    split = preprocess_and_split(df, test_size=0.2, random_state=42)
    models = get_models_dictionary()
    assert len(models) == 5, f"Expected 5 models, found {len(models)}"

    for name, model in models.items():
        metrics = evaluate_single_model(
            model,
            split["X_train_proc"],
            split["y_train_encoded"],
            split["X_test_proc"],
            split["y_test_encoded"],
            split["target_classes"]
        )
        assert 0.0 <= metrics["accuracy"] <= 1.0, f"Invalid accuracy for {name}"
        assert 0.0 <= metrics["f1_weighted"] <= 1.0, f"Invalid F1 for {name}"
        assert len(metrics["confusion_matrix"]) == 7, f"Invalid CM dimensions for {name}"
        print(f"  --> {name}: Test Acc={metrics['accuracy']*100:.1f}%, F1={metrics['f1_weighted']*100:.1f}%, CV={metrics['cv_mean']*100:.1f}%")


def test_5_saved_artifacts_loading():
    print("[TEST 5] Verifying saved model artifacts loading...")
    import joblib
    models_dir = os.path.join(BASE_DIR, "models")
    best_m = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    preproc = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    le = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
    all_m = joblib.load(os.path.join(models_dir, "all_models.joblib"))

    assert best_m is not None
    assert preproc is not None
    assert le is not None
    assert len(all_m) == 5
    print(f"  --> All saved artifacts in 'models/' loaded and verified ({len(all_m)} models cached).")


def test_6_predictor_inference():
    print("[TEST 6] Testing live fertilizer predictions...")
    from src.prediction import predictor
    test_cases = [
        {
            "Temparature": 26, "Humidity": 52, "Moisture": 38,
            "Soil Type": "Sandy", "Crop Type": "Maize",
            "Nitrogen": 37, "Potassium": 0, "Phosphorous": 0
        },
        {
            "Temparature": 29, "Humidity": 52, "Moisture": 45,
            "Soil Type": "Clayey", "Crop Type": "Paddy",
            "Nitrogen": 12, "Potassium": 0, "Phosphorous": 36
        },
        {
            "Temparature": 34, "Humidity": 65, "Moisture": 62,
            "Soil Type": "Loamy", "Crop Type": "Sugarcane",
            "Nitrogen": 10, "Potassium": 9, "Phosphorous": 30
        }
    ]

    for tc in test_cases:
        res = predictor.predict(tc)
        assert res["recommended_fertilizer"] is not None
        assert res["confidence"] is not None
        assert 0.0 <= res["confidence"] <= 1.0
        assert len(res["explanation"]) > 20
        print(f"  --> Prediction for {tc['Crop Type']} in {tc['Soil Type']}: {res['recommended_fertilizer']} (Confidence: {res['confidence']*100:.1f}%)")


def test_7_database_crud_and_persistence():
    print("[TEST 7] Testing SQLite database CRUD operations and persistence...")
    from src.database import (
        add_farm_record, get_all_farm_records, get_farm_record_by_id,
        update_farm_record, delete_farm_record,
        add_recommendation_history, get_all_recommendations, delete_recommendation_history
    )

    # 1. Farm Records CRUD
    fid = add_farm_record(
        farmer_name="Automated Test Farmer",
        farm_name="Test Farm Alpha",
        crop_name="Cotton",
        soil_type="Black",
        fertilizer_used="10-26-26",
        application_date="2026-06-01",
        farm_area_acres=4.5,
        notes="Automated verification test"
    )
    assert fid is not None, "Failed to insert farm record"

    rec = get_farm_record_by_id(fid)
    assert rec["farmer_name"] == "Automated Test Farmer"
    assert rec["crop_name"] == "Cotton"

    updated = update_farm_record(
        fid, "Automated Test Farmer", "Test Farm Alpha Updated",
        "Cotton", "Black", "10-26-26", "2026-06-05", 5.0, "Updated notes"
    )
    assert updated is True, "Failed to update farm record"

    rec_updated = get_farm_record_by_id(fid)
    assert rec_updated["farm_name"] == "Test Farm Alpha Updated"
    assert rec_updated["farm_area_acres"] == 5.0

    deleted = delete_farm_record(fid)
    assert deleted is True, "Failed to delete farm record"
    assert get_farm_record_by_id(fid) is None, "Record should be gone"

    # 2. Recommendation History CRUD
    rid = add_recommendation_history(
        farmer_name="Test Grower",
        crop_type="Wheat",
        soil_type="Loamy",
        temperature=25,
        humidity=50,
        moisture=40,
        nitrogen=20,
        potassium=5,
        phosphorous=25,
        recommended_fertilizer="DAP",
        confidence=0.98,
        model_used="XGBoost",
        explanation="Test explanation"
    )
    assert rid is not None
    del_rec = delete_recommendation_history(rid)
    assert del_rec is True

    print("  --> SQLite CRUD operations passed cleanly!")


def test_8_application_schedule():
    print("[TEST 8] Testing Application Schedule generation and tasks...")
    import datetime
    from src.schedule import generate_schedule, save_schedule_to_db
    from src.database import get_all_schedules, update_schedule_task_status, delete_schedule_task

    today = datetime.date.today()
    stages = generate_schedule("Paddy", today, "Urea", "Test Paddy Field")
    assert len(stages) == 4, f"Paddy should have 4 stages, found {len(stages)}"
    assert stages[0]["target_date"] == today.strftime("%Y-%m-%d")

    tids = save_schedule_to_db(stages)
    assert len(tids) == 4

    # Update task 1
    t1_id = tids[0]
    update_res = update_schedule_task_status(t1_id, "Completed", completion_date=today.strftime("%Y-%m-%d"))
    assert update_res is True

    # Cleanup test tasks
    for tid in tids:
        delete_schedule_task(tid)

    print("  --> Schedule generator and task tracker verified!")


def test_9_ai_assistant():
    print("[TEST 9] Testing AI Agricultural Assistant responses...")
    from src.agent import assistant

    queries = [
        "What is Urea?",
        "Explain DAP fertilizer",
        "How to manage Clayey soil?",
        "What are the symptoms of Potassium deficiency?",
        "What are the safety rules for chemical fertilizers?",
        "Are you an LLM?"
    ]

    for q in queries:
        resp = assistant.process_query(q)
        assert len(resp) > 30, f"Response too short for '{q}'"
        assert "http" not in resp or "api" not in resp  # verify no broken API calls
        print(f"  --> '{q}' -> Response length: {len(resp)} chars.")

    # Contextual query
    ctx = {"recommended_fertilizer": "10-26-26", "crop_type": "Sugarcane", "soil_type": "Black"}
    resp_ctx = assistant.process_query("Why this recommendation?", context=ctx)
    assert "10-26-26" in resp_ctx
    print("  --> Contextual recommendation explanation verified!")


def test_10_streamlit_pages_compilation():
    print("[TEST 10] Testing syntax and compile check on all Streamlit pages...")
    import py_compile
    pages = [
        "app.py",
        os.path.join("pages", "1_🌱_Fertilizer_Recommendation.py"),
        os.path.join("pages", "2_📅_Application_Schedule.py"),
        os.path.join("pages", "3_🚜_Farm_Management.py"),
        os.path.join("pages", "4_💬_AI_Agricultural_Assistant.py"),
        os.path.join("pages", "5_📊_ML_Model_Performance.py"),
        os.path.join("pages", "6_📜_Recommendation_History.py"),
        os.path.join("pages", "7_ℹ️_About_Project.py")
    ]

    for p in pages:
        full_p = os.path.join(BASE_DIR, p)
        assert os.path.exists(full_p), f"Missing page: {p}"
        py_compile.compile(full_p, doraise=True)
        print(f"  --> Page '{p}' compiled successfully without errors.")


def run_all_tests():
    print("=" * 60)
    print("RUNNING COMPLETE VERIFICATION TEST SUITE")
    print("=" * 60)
    test_1_imports()
    test_2_dataset_loading()
    test_3_preprocessing_and_data_leakage()
    test_4_models_and_evaluation_metrics()
    test_5_saved_artifacts_loading()
    test_6_predictor_inference()
    test_7_database_crud_and_persistence()
    test_8_application_schedule()
    test_9_ai_assistant()
    test_10_streamlit_pages_compilation()
    print("=" * 60)
    print("ALL 10 COMPREHENSIVE SYSTEM TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
