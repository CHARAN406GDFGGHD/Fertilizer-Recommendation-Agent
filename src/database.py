"""
SQLite Database Module
Manages persistent local storage for:
1. Farm Management Records (CRUD)
2. Recommendation History
3. Fertilizer Application Schedules & Tasks
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "farm_database.db")


def get_connection():
    """Returns a SQLite connection with row factory enabled."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes database schema with required tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Farm Records Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS farm_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_name TEXT NOT NULL,
            farm_name TEXT NOT NULL,
            crop_name TEXT NOT NULL,
            soil_type TEXT NOT NULL,
            fertilizer_used TEXT,
            application_date TEXT,
            farm_area_acres REAL DEFAULT 1.0,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Recommendation History Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_name TEXT DEFAULT 'Anonymous Farmer',
            crop_type TEXT NOT NULL,
            soil_type TEXT NOT NULL,
            temperature REAL,
            humidity REAL,
            moisture REAL,
            nitrogen REAL,
            potassium REAL,
            phosphorous REAL,
            recommended_fertilizer TEXT NOT NULL,
            confidence REAL,
            model_used TEXT,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 3. Application Schedules Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS application_schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farm_name TEXT DEFAULT 'My Farm',
            crop_name TEXT NOT NULL,
            fertilizer_name TEXT NOT NULL,
            planting_date TEXT NOT NULL,
            stage_name TEXT NOT NULL,
            target_date TEXT NOT NULL,
            recommended_dose_desc TEXT,
            application_method TEXT,
            status TEXT DEFAULT 'Pending',
            completion_date TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ==========================================
# FARM MANAGEMENT CRUD OPERATIONS
# ==========================================

def add_farm_record(farmer_name, farm_name, crop_name, soil_type, fertilizer_used="", application_date="", farm_area_acres=1.0, notes=""):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO farm_records (farmer_name, farm_name, crop_name, soil_type, fertilizer_used, application_date, farm_area_acres, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (farmer_name, farm_name, crop_name, soil_type, fertilizer_used, application_date, farm_area_acres, notes))
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id


def get_all_farm_records():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM farm_records ORDER BY id DESC", conn)
    conn.close()
    return df


def get_farm_record_by_id(record_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM farm_records WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_farm_record(record_id, farmer_name, farm_name, crop_name, soil_type, fertilizer_used, application_date, farm_area_acres, notes):
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        UPDATE farm_records
        SET farmer_name = ?,
            farm_name = ?,
            crop_name = ?,
            soil_type = ?,
            fertilizer_used = ?,
            application_date = ?,
            farm_area_acres = ?,
            notes = ?,
            updated_at = ?
        WHERE id = ?
    """, (farmer_name, farm_name, crop_name, soil_type, fertilizer_used, application_date, farm_area_acres, notes, now_str, record_id))
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated


def delete_farm_record(record_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM farm_records WHERE id = ?", (record_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


# ==========================================
# RECOMMENDATION HISTORY CRUD OPERATIONS
# ==========================================

def add_recommendation_history(farmer_name, crop_type, soil_type, temperature, humidity, moisture, nitrogen, potassium, phosphorous, recommended_fertilizer, confidence=None, model_used="", explanation=""):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO recommendation_history (
            farmer_name, crop_type, soil_type, temperature, humidity, moisture,
            nitrogen, potassium, phosphorous, recommended_fertilizer, confidence,
            model_used, explanation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        farmer_name, crop_type, soil_type, temperature, humidity, moisture,
        nitrogen, potassium, phosphorous, recommended_fertilizer, confidence,
        model_used, explanation
    ))
    conn.commit()
    rec_id = cursor.lastrowid
    conn.close()
    return rec_id


def get_all_recommendations():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM recommendation_history ORDER BY id DESC", conn)
    conn.close()
    return df


def delete_recommendation_history(record_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM recommendation_history WHERE id = ?", (record_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def clear_all_recommendations():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM recommendation_history")
    conn.commit()
    conn.close()


# ==========================================
# APPLICATION SCHEDULES CRUD OPERATIONS
# ==========================================

def add_schedule_task(farm_name, crop_name, fertilizer_name, planting_date, stage_name, target_date, recommended_dose_desc="", application_method="", status="Pending", notes=""):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO application_schedules (
            farm_name, crop_name, fertilizer_name, planting_date, stage_name,
            target_date, recommended_dose_desc, application_method, status, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        farm_name, crop_name, fertilizer_name, planting_date, stage_name,
        target_date, recommended_dose_desc, application_method, status, notes
    ))
    conn.commit()
    task_id = cursor.lastrowid
    conn.close()
    return task_id


def get_all_schedules(filter_crop=None):
    conn = get_connection()
    if filter_crop:
        df = pd.read_sql_query(
            "SELECT * FROM application_schedules WHERE crop_name = ? ORDER BY target_date ASC",
            conn, params=(filter_crop,)
        )
    else:
        df = pd.read_sql_query("SELECT * FROM application_schedules ORDER BY target_date ASC", conn)
    conn.close()
    return df


def update_schedule_task_status(task_id, new_status, completion_date=None, notes=None):
    conn = get_connection()
    cursor = conn.cursor()
    if notes is not None:
        cursor.execute("""
            UPDATE application_schedules
            SET status = ?, completion_date = ?, notes = ?
            WHERE id = ?
        """, (new_status, completion_date, notes, task_id))
    else:
        cursor.execute("""
            UPDATE application_schedules
            SET status = ?, completion_date = ?
            WHERE id = ?
        """, (new_status, completion_date, task_id))
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated


def delete_schedule_task(task_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM application_schedules WHERE id = ?", (task_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


# Auto-initialize database schema on module load
init_db()
