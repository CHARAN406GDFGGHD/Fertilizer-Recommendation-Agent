# Intelligent Fertilizer Recommendation Agent Using Machine Learning

**Live Demo:** https://charan406gdfgghd-fertilizer-recommendation-agent-app-asfgmr.streamlit.app/

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![ML-Scikit--Learn_XGBoost](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![Database-SQLite3](https://img.shields.io/badge/Database-SQLite3-003B57.svg)](https://sqlite.org/)
[![Academic-B.Tech_Data_Science](https://img.shields.io/badge/Project-B.Tech%20Data%20Science-2E7D32.svg)](#)

A comprehensive, production-ready, full-stack Data Science mini project that combines supervised machine learning classification, agronomic decision-support logic, and an interactive Streamlit dashboard to recommend optimal fertilizers for precision agriculture.

---

## 📌 1. Project Overview & Introduction

Chemical fertilizers have played a critical role in increasing global crop yields; however, their improper, unscientific, and excessive application leads to severe soil acidification, freshwater eutrophication, greenhouse gas emissions, and economic strain on farming communities.

The **Intelligent Fertilizer Recommendation Agent** provides a reliable, data-driven solution. By analyzing environmental variables (Temperature, Humidity, Soil Moisture), physical soil properties, and chemical nutrient levels (Available Nitrogen, Phosphorous, and Potassium ratios), the system classifies and prescribes the most effective fertilizer formulation tailored to specific crop requirements.

The project is structured into two main phases:
- **Phase 1: Machine Learning Benchmark** — Rigorous comparison and evaluation of 5 supervised machine learning models with zero data leakage.
- **Phase 2: Intelligent Agricultural Agent** — 5 interactive functional modules providing end-to-end decision support, stage-wise scheduling, parcel management, and domain Q&A.

---

## 🎯 2. Problem Statement

Smallholder and modern farmers frequently face the challenge of determining the exact fertilizer blend and timing for their crops. Agricultural extension services are often inaccessible or delayed, leading farmers to rely on conventional rule-of-thumb practices or over-apply high-nitrogen fertilizers like Urea. This causes:
1. **Nutrient Imbalance:** Supplying excess Nitrogen while crops suffer from Phosphorus or Potassium deficiencies.
2. **Environmental Degradation:** Nitrate runoff into groundwater and water bodies.
3. **Yield Loss:** Sub-optimal crop growth and disease vulnerability due to inappropriate nutrient ratios.

This project delivers an offline-capable, cost-free, and interpretable software agent to guide fertilizer selection, schedule applications, and track parcel records.

---

## 📚 3. Objectives

- **Empirical ML Comparison:** Train, evaluate, and benchmark 5 distinct classification algorithms on real-world agricultural data.
- **Strict Data Leakage Prevention:** Implement disciplined `ColumnTransformer` pipelines where scaling and encoding parameters are learned strictly from the training partition.
- **Explainable Recommendations:** Pair model predictions with agronomic rationales and confidence metrics.
- **Crop-Specific Scheduling:** Generate stage-wise split application timelines (Basal, Tillering, Flowering, Grain Filling).
- **Persistent Data Management:** Deliver an SQLite-backed farm parcel and recommendation audit trail.
- **Offline AI Assistant:** Offer a domain-expert chatbot that functions 100% offline without requiring paid third-party cloud API keys.

---

## 📁 4. Dataset Source & Description

- **Dataset Source:** Kaggle Fertilizer Prediction Dataset by Abhishek (gdabhishek)  
  URL: [https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction](https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction)
- **Total Records:** 99 agricultural soil & crop observations
- **Total Columns:** 9 (8 input features, 1 multiclass target)
- **Data Quality:** 0 missing values, 0 duplicate records

### Feature Schema:
| Column Name | Type | Description | Observed Range |
| :--- | :--- | :--- | :--- |
| `Temparature` | Numeric (Integer) | Ambient temperature in °C | 25 – 38 °C |
| `Humidity` | Numeric (Integer) | Relative atmospheric humidity (%) | 50 – 72 % |
| `Moisture` | Numeric (Integer) | Soil moisture index (%) | 25 – 65 % |
| `Soil Type` | Categorical (String) | Soil texture classification | Loamy (21), Sandy (20), Clayey (20), Black (19), Red (19) |
| `Crop Type` | Categorical (String) | Crop grown | Sugarcane (13), Cotton (12), Millets (11), Paddy (10), Pulses (10), Wheat (9), Tobacco (7), Barley (7), Oil seeds (7), Ground Nuts (7), Maize (6) |
| `Nitrogen` | Numeric (Integer) | Soil available Nitrogen index | 4 – 42 |
| `Potassium` | Numeric (Integer) | Soil available Potassium index | 0 – 19 |
| `Phosphorous`| Numeric (Integer) | Soil available Phosphorous index | 0 – 42 |
| **`Fertilizer Name`** | Categorical (Target) | Recommended fertilizer formulation | **7 Classes:** Urea (22), DAP (18), 28-28 (17), 14-35-14 (14), 20-20 (14), 17-17-17 (7), 10-26-26 (7) |

---

## 🛠️ 5. Technology Stack

- **Core Programming Language:** Python 3.10+
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning & Preprocessing:** Scikit-learn, XGBoost, Joblib
- **Data Visualization:** Matplotlib, Seaborn
- **Interactive Web Framework:** Streamlit
- **Relational Database:** SQLite3 (Serverless, zero-config local persistence)
- **Dataset Retrieval:** KaggleHub API

---

## 🔬 6. Phase 1: Machine Learning Pipeline

### Data Preprocessing & Leakage Prevention
1. **Target Stratification:** The dataset is split into **80% Training (79 samples)** and **20% Testing (20 samples)** using stratified sampling on `Fertilizer Name` to preserve the exact class distribution across subsets.
2. **Pipeline Architecture:**
   - **Numerical Features** (`Temparature`, `Humidity`, `Moisture`, `Nitrogen`, `Potassium`, `Phosphorous`): Scaled via `StandardScaler`.
   - **Categorical Features** (`Soil Type`, `Crop Type`): One-hot encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
   - **Target Variable**: Encoded to discrete class integers via `LabelEncoder`.
3. **Strict Boundary:** The preprocessor is fitted **only on `X_train`** and applied downstream to `X_test` and new user inputs, completely preventing information leakage.

### Five Model Benchmark
We evaluate 5 distinct classification paradigms:
1. **Random Forest Classifier:** Bagged ensemble of 100 decision trees.
2. **XGBoost Classifier:** Regularized gradient boosting framework (`eval_metric='mlogloss'`).
3. **Support Vector Machine (SVM):** Non-linear Support Vector Classifier using Radial Basis Function (RBF) kernel with Platt probability calibration.
4. **K-Nearest Neighbors (KNN):** Distance-weighted neighborhood voting ($k=3$).
5. **Multi-Layer Perceptron (MLP):** Feedforward neural network architecture (Hidden layers: 64, 32; ReLU activation; Adam optimizer).

### Actual Model Evaluation Results:
*(Empirical benchmark on the held-out stratified test set and 5-fold cross-validation)*

| Model | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | 5-Fold CV Accuracy (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier (Best)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **96.08% ± 5.29%** |
| **Random Forest Classifier** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **91.25% ± 8.48%** |
| **Multi-Layer Perceptron (MLP)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **84.75% ± 10.26%** |
| **Support Vector Machine (SVM)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **78.58% ± 12.76%** |
| **K-Nearest Neighbors (KNN)** | **90.00%** | **0.9375** | **0.9000** | **0.9019** | **72.17% ± 16.07%** |

- **Best Model Selection:** **XGBoost** is selected as the primary production model because it achieved 100% test accuracy with the highest cross-validation score (96.08%) and smallest variance across validation folds.
- **Model Serialization:** All trained models, pipelines, and label encoders are saved using Joblib in `models/`.

---

## 🌾 7. Phase 2: Intelligent Agricultural Agent

### Module 1: Fertilizer Recommendation
- Provides an intuitive form accepting all 8 dataset features with quick scenario autofill buttons.
- Generates instant fertilizer predictions along with prediction confidence percentage.
- Displays probability distributions across all candidate fertilizer types.
- Produces an agronomic explanation justifying why the fertilizer fits the current N-P-K balance.
- Saves recommendations directly to the SQLite audit log.

### Module 2: Fertilizer Application Schedule
- Generates crop-specific split application stages (Basal, Vegetative, Flowering/Tillering, Maturity).
- Calculates exact calendar target dates based on the crop's sowing date.
- Features interactive task checkboxes to mark stages as **Completed** or **Pending**.
- Includes safety precautions and clear agronomic disclaimers stating that schedules serve as general guidance and should be verified with soil tests and local extension experts.

### Module 3: Farm Parcel Management (SQLite CRUD)
- **Create:** Register new farm parcels, farmer names, crops, soil types, acreage, and operational notes.
- **Read:** Search and filter parcels dynamically; export records as CSV.
- **Update:** Edit existing parcel records in place.
- **Delete:** Remove parcels with permanent deletion safeguard.
- All records persist across application restarts in `data/farm_database.db`.

### Module 4: AI Agricultural Assistant
- Interactive conversational agent answering questions about fertilizers, soil health, crop nutrient requirements, and safety.
- Operates **100% offline** via an integrated domain knowledge base and agronomic heuristic engine.
- Never pretends an external LLM is connected; requires **zero paid API keys**.

### Module 5: Recommendation History
- Complete historical record of all past fertilizer predictions.
- Displays prediction timestamp, environmental conditions, input nutrients, recommended fertilizer, and confidence score.
- Filter by crop or fertilizer, inspect individual records, or export the log as CSV.

---

## 🏗️ 8. System Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │       Kaggle Dataset (Fertilizer Pred)       │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │          src/data_preprocessing.py           │
                  │   StandardScaler (Num) + OneHotEncoder (Cat) │
                  │     Fitted EXCLUSIVELY on Train Subset       │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │             src/train_models.py              │
                  │   RF | XGBoost | SVM | KNN | MLP (5-Fold CV) │
                  │       Best Model: XGBoost (96.08% CV)        │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │       Joblib Artifacts & SQLite Storage      │
                  │  models/*.joblib  │  data/farm_database.db   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                      STREAMLIT MULTI-PAGE APPLICATION                  │
    ├──────────────────┬──────────────────┬──────────────────┬───────────────┤
    │  Home Dashboard  │ Fertilizer Rec.  │ App. Schedule    │ Farm Parcel DB│
    │     (app.py)     │    (Page 1)      │    (Page 2)      │   (Page 3)    │
    ├──────────────────┼──────────────────┼──────────────────┼───────────────┤
    │  AI Assistant    │ ML Benchmark     │ Rec. History     │ About Project │
    │    (Page 4)      │    (Page 5)      │    (Page 6)      │   (Page 7)    │
    └──────────────────┴──────────────────┴──────────────────┴───────────────┘
```

---

## 📂 9. Project Folder Structure

```
MINI.PROJECT/
├── app.py                             # Main Streamlit Application & Home Dashboard
├── requirements.txt                   # Project dependencies
├── README.md                          # Comprehensive project documentation
├── .gitignore                         # Git exclusion rules
├── .streamlit/
│   └── config.toml                    # Fresh green light theme configuration
├── data/
│   ├── Fertilizer Prediction.csv      # Actual Kaggle benchmark dataset
│   └── farm_database.db               # Persistent SQLite database
├── models/
│   ├── best_model.joblib              # Serialized best model (XGBoost)
│   ├── preprocessor.joblib            # Fitted ColumnTransformer pipeline
│   ├── label_encoder.joblib           # Fitted LabelEncoder for target classes
│   ├── all_models.joblib              # Dictionary of all 5 trained models
│   └── model_metadata.json            # Model performance metadata
├── results/
│   ├── model_comparison.csv           # 5-model comparative evaluation metrics
│   ├── evaluation_results.json        # Detailed classification reports & CM data
│   ├── dataset_summary.json           # Dataset statistics and EDA metadata
│   ├── model_comparison.png           # Comparative Accuracy & F1 bar chart
│   ├── best_confusion_matrix.png      # Confusion matrix for XGBoost
│   └── all_confusion_matrices.png     # Side-by-side confusion matrices for all 5 models
├── src/
│   ├── __init__.py                    # Source package marker
│   ├── data_preprocessing.py          # Data ingestion, cleaning, split, pipeline
│   ├── train_models.py                # 5-model training, evaluation & plot export
│   ├── prediction.py                  # Predictor class, inference & agronomic explanations
│   ├── database.py                    # SQLite connection & CRUD functions
│   ├── schedule.py                    # Growth-stage split application generator
│   ├── agent.py                       # Offline AI Agricultural Assistant engine
│   └── ui_helpers.py                  # Streamlit CSS styling, header & sidebar helpers
├── pages/
│   ├── 1_🌱_Fertilizer_Recommendation.py
│   ├── 2_📅_Application_Schedule.py
│   ├── 3_🚜_Farm_Management.py
│   ├── 4_💬_AI_Agricultural_Assistant.py
│   ├── 5_📊_ML_Model_Performance.py
│   ├── 6_📜_Recommendation_History.py
│   └── 7_ℹ️_About_Project.py
└── tests/
    └── test_system.py                 # Automated 10-point test suite
```

---

## 🚀 10. Installation & Execution Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed.
- (Optional) Virtual environment:
  ```bash
  python -m venv venv
  # On Windows:
  venv\Scripts\activate
  # On macOS/Linux:
  source venv/bin/activate
  ```

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Retrain Models
To regenerate all model artifacts, evaluation JSONs, and visualization charts from scratch:
```bash
python -m src.train_models
```

### Step 3: Run Automated Verification Tests
Run the 10-point verification test suite covering imports, preprocessing, model inference, database CRUD, and Streamlit page syntax:
```bash
python -m tests.test_system
```

### Step 4: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your web browser and navigate to `http://localhost:8501`.

---

## 🧪 11. Testing & Quality Verification

All 10 verification checks have been executed and confirmed passing:
- [x] **Test 1: Dependencies & Imports** — All libraries load correctly.
- [x] **Test 2: Dataset Loading** — Verified 99 rows, 9 columns, 7 target classes.
- [x] **Test 3: Preprocessing & Data Leakage** — Preprocessing strictly fitted on train set only.
- [x] **Test 4: 5 Models Benchmark** — All 5 models trained with cross-validation.
- [x] **Test 5: Saved Artifacts** — Joblib models and pipelines load correctly.
- [x] **Test 6: Inference Engine** — Predictions return valid classes, confidences, and explanations.
- [x] **Test 7: SQLite Persistence** — Farm records and recommendations support Insert, Select, Update, and Delete.
- [x] **Test 8: Schedule Generator** — Split-application stages correctly calculated from sowing date.
- [x] **Test 9: AI Assistant** — Domain responses generated offline without errors or API dependencies.
- [x] **Test 10: Streamlit Compilation** — All 8 application pages compile without syntax errors.

---

## 🔮 12. Future Enhancements

1. **IoT Sensor Integration:** Real-time ingestion of soil NPK and moisture sensor data via MQTT/ESP32 microcontrollers.
2. **Weather API Sync:** Ingestion of 7-day rainfall forecasts to dynamically reschedule top-dressing applications before heavy storms.
3. **Multilingual Support:** Localization into regional agricultural languages (Hindi, Telugu, Tamil, Marathi, Punjabi) for wider rural accessibility.
4. **Soil Health Card OCR:** Optical Character Recognition to scan paper soil test reports directly into the recommendation form.

---

## 📜 13. Academic Citation & License

This project was built for educational, academic, and research purposes as part of the **B.Tech Data Science** curriculum.
- **Dataset Citation:** Abhishek, *Fertilizer Prediction Dataset*, Kaggle (2020).
- **License:** Open Source under the MIT License.
