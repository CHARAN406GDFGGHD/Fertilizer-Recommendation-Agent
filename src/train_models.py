"""
Model Training & Evaluation Module
Trains and compares 5 machine learning models:
1. Random Forest Classifier
2. XGBoost Classifier
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (KNN)
5. Multi-Layer Perceptron (MLP)

Evaluates accuracy, precision, recall, F1-score, confusion matrices,
and saves the best model and preprocessors using Joblib.
"""

import os
import json
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.data_preprocessing import (
    load_dataset,
    get_dataset_summary,
    preprocess_and_split,
    BASE_DIR
)

# Output paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_models_dictionary():
    """
    Returns the initial dictionary of 5 model configurations.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            eval_metric="mlogloss",
            random_state=42
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=3,
            weights="distance"
        ),
        "Multi-Layer Perceptron": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42
        )
    }


def evaluate_single_model(model, X_train, y_train, X_test, y_test, target_classes):
    """
    Fits and evaluates a single model, returning comprehensive evaluation metrics.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # 5-Fold Stratified Cross-Validation on training set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=target_classes,
        output_dict=True,
        zero_division=0
    )
    clf_report_text = classification_report(
        y_test,
        y_pred,
        target_names=target_classes,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": acc,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "classification_report": clf_report,
        "classification_report_text": clf_report_text,
        "confusion_matrix": cm.tolist()
    }


def plot_and_save_visualizations(comparison_df, evaluation_results, target_classes):
    """
    Generates and saves:
    1. Model accuracy & F1 comparison bar chart
    2. Combined confusion matrices visualization
    3. Individual confusion matrices
    """
    # 1. Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    bar_width = 0.35
    index = np.arange(len(comparison_df))

    plt.bar(index, comparison_df["Accuracy"] * 100, bar_width, label="Test Accuracy (%)", color="#2E7D32")
    plt.bar(index + bar_width, comparison_df["F1-Score (Weighted)"] * 100, bar_width, label="F1-Score (%)", color="#81C784")

    plt.xlabel("Machine Learning Models", fontweight="bold", fontsize=11)
    plt.ylabel("Score (%)", fontweight="bold", fontsize=11)
    plt.title("Comparison of Model Performance (Kaggle Fertilizer Dataset)", fontweight="bold", fontsize=13)
    plt.xticks(index + bar_width / 2, comparison_df["Model"], rotation=15, ha="right", fontsize=10)
    plt.ylim(0, 110)
    plt.legend(frameon=True, facecolor="#F4F7F5")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()

    # 2. Combined Confusion Matrix Plot
    n_models = len(evaluation_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, evaluation_results.items()):
        cm = np.array(res["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=target_classes,
            yticklabels=target_classes,
            cbar=False,
            ax=ax
        )
        ax.set_title(f"{model_name}\nAcc: {res['accuracy']*100:.1f}%", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.suptitle("Confusion Matrices - Model Comparison", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    all_cm_path = os.path.join(RESULTS_DIR, "all_confusion_matrices.png")
    plt.savefig(all_cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Individual Confusion Matrix for Best Model
    best_model_name = comparison_df.iloc[0]["Model"]
    plt.figure(figsize=(7, 6))
    best_cm = np.array(evaluation_results[best_model_name]["confusion_matrix"])
    sns.heatmap(
        best_cm,
        annot=True,
        fmt="d",
        cmap="YlGn",
        xticklabels=target_classes,
        yticklabels=target_classes,
        cbar=True
    )
    plt.title(f"Confusion Matrix - Best Model: {best_model_name}", fontweight="bold", fontsize=12)
    plt.xlabel("Predicted Fertilizer", fontweight="bold")
    plt.ylabel("Actual Fertilizer", fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    best_cm_path = os.path.join(RESULTS_DIR, "best_confusion_matrix.png")
    plt.savefig(best_cm_path, dpi=300)
    plt.close()


def train_and_evaluate_pipeline():
    """
    Orchestrates data loading, preprocessing, 5-model training, evaluation,
    artifact saving, and report generation.
    """
    print("=" * 60)
    print("FERTILIZER RECOMMENDATION AGENT - ML PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading actual Kaggle dataset...")
    df = load_dataset()
    summary = get_dataset_summary(df)
    print(f"Loaded {summary['rows']} records and {summary['columns']} columns.")
    print(f"Target classes ({len(summary['target_distribution'])}): {list(summary['target_distribution'].keys())}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    # 2. Preprocess and split
    print("\n[2/5] Preprocessing and performing stratified split...")
    split_data = preprocess_and_split(df, test_size=0.2, random_state=42)
    X_train_proc = split_data["X_train_proc"]
    X_test_proc = split_data["X_test_proc"]
    y_train = split_data["y_train_encoded"]
    y_test = split_data["y_test_encoded"]
    target_classes = split_data["target_classes"]
    preprocessor = split_data["preprocessor"]
    label_encoder = split_data["label_encoder"]

    # 3. Train all 5 models
    print("\n[3/5] Training and evaluating 5 machine learning models...")
    models = get_models_dictionary()
    trained_models = {}
    evaluation_results = {}
    comparison_rows = []

    for name, model in models.items():
        print(f"  --> Training {name}...")
        res = evaluate_single_model(model, X_train_proc, y_train, X_test_proc, y_test, target_classes)
        trained_models[name] = model
        evaluation_results[name] = res

        comparison_rows.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "Precision (Weighted)": res["precision_weighted"],
            "Recall (Weighted)": res["recall_weighted"],
            "F1-Score (Weighted)": res["f1_weighted"],
            "Precision (Macro)": res["precision_macro"],
            "Recall (Macro)": res["recall_macro"],
            "F1-Score (Macro)": res["f1_macro"],
            "CV Accuracy (Mean)": res["cv_mean"],
            "CV Accuracy (Std)": res["cv_std"]
        })

    # Convert to DataFrame and sort by F1-Score, then Accuracy, then CV Mean
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values(
        by=["F1-Score (Weighted)", "Accuracy", "CV Accuracy (Mean)"],
        ascending=False
    ).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    # 4. Save results and plots
    print("\n[4/5] Generating evaluation charts and saving metrics...")
    csv_comp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(csv_comp_path, index=False)

    eval_json_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(eval_json_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)

    plot_and_save_visualizations(comparison_df, evaluation_results, target_classes)

    # 5. Determine and save best model
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n[5/5] Best Model Selected: {best_model_name} (Acc: {comparison_df.iloc[0]['Accuracy']*100:.2f}%, F1: {comparison_df.iloc[0]['F1-Score (Weighted)']*100:.2f}%)")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    joblib.dump(trained_models, os.path.join(MODELS_DIR, "all_models.joblib"))

    # Metadata for runtime predictions
    metadata = {
        "best_model_name": best_model_name,
        "best_accuracy": float(comparison_df.iloc[0]["Accuracy"]),
        "best_f1_score": float(comparison_df.iloc[0]["F1-Score (Weighted)"]),
        "target_classes": target_classes,
        "feature_names": split_data["feature_names"],
        "num_classes": len(target_classes),
        "soil_types": summary["unique_soil_types"],
        "crop_types": summary["unique_crop_types"]
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nAll artifacts saved successfully to 'models/' and 'results/' directories!")
    return comparison_df, evaluation_results, metadata


if __name__ == "__main__":
    train_and_evaluate_pipeline()
