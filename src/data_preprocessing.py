"""
Data Preprocessing Module
Handles loading, inspecting, cleaning, and preprocessing the fertilizer dataset.
"""

import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "Fertilizer Prediction.csv")

# Known schema definitions
NUMERICAL_COLS = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
CATEGORICAL_COLS = ["Soil Type", "Crop Type"]
TARGET_COL = "Fertilizer Name"


def ensure_dataset(data_path=DEFAULT_DATA_PATH):
    """
    Ensures the dataset exists. If missing, attempts to download via kagglehub.
    Returns the resolved path or raises FileNotFoundError.
    """
    if os.path.exists(data_path):
        return data_path

    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        import kagglehub
        download_path = kagglehub.dataset_download("gdabhishek/fertilizer-prediction")
        candidate = os.path.join(download_path, "Fertilizer Prediction.csv")
        if os.path.exists(candidate):
            shutil.copy(candidate, data_path)
            return data_path
    except Exception as e:
        print(f"Automatic Kaggle download failed: {e}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Please place 'Fertilizer Prediction.csv' in the 'data/' directory."
        )
    return data_path


def load_dataset(data_path=DEFAULT_DATA_PATH):
    """
    Loads and normalizes the dataset columns (strips leading/trailing spaces).
    """
    resolved_path = ensure_dataset(data_path)
    df = pd.read_csv(resolved_path)
    # Strip whitespace from column names (e.g. 'Humidity ' -> 'Humidity')
    df.columns = df.columns.str.strip()
    return df


def get_dataset_summary(df):
    """
    Generates a structured dictionary containing exploratory data analysis summary.
    """
    summary = {
        "shape": df.shape,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(val) for col, val in df.isnull().sum().items()},
        "duplicates": int(df.duplicated().sum()),
        "target_distribution": {k: int(v) for k, v in df[TARGET_COL].value_counts().items()} if TARGET_COL in df.columns else {},
        "unique_soil_types": sorted(df["Soil Type"].unique().tolist()) if "Soil Type" in df.columns else [],
        "unique_crop_types": sorted(df["Crop Type"].unique().tolist()) if "Crop Type" in df.columns else [],
        "numerical_stats": df[NUMERICAL_COLS].describe().round(2).to_dict() if all(c in df.columns for c in NUMERICAL_COLS) else {}
    }
    return summary


def preprocess_and_split(df, test_size=0.2, random_state=42):
    """
    Preprocesses the dataset:
    - Splits into Train and Test sets using Stratification on target.
    - Fits Scaler and One-Hot Encoder exclusively on Training set to avoid data leakage.
    - Encodes target variable using LabelEncoder.
    Returns a dictionary of split datasets, preprocessors, and feature names.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from DataFrame.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Stratified split to maintain class ratios across train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Preprocessing Pipeline: StandardScaler for numerical, OneHotEncoder for categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS)
        ]
    )

    # Fit ONLY on training data to prevent data leakage
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Target Label Encoding
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
    all_feature_names = NUMERICAL_COLS + cat_feature_names

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "y_train_raw": y_train,
        "y_test_raw": y_test,
        "y_train_encoded": y_train_encoded,
        "y_test_encoded": y_test_encoded,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "feature_names": all_feature_names,
        "target_classes": label_encoder.classes_.tolist()
    }
