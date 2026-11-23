"""
Prediction Engine Module
Loads the trained models and preprocessor pipeline, validates input features,
and produces fertilizer recommendations with confidence scores and agronomic explanations.
"""

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class FertilizerPredictor:
    """
    Singleton-style predictor that caches trained model artifacts for high performance.
    """
    def __init__(self):
        self.preprocessor = None
        self.label_encoder = None
        self.best_model = None
        self.all_models = None
        self.metadata = None
        self._load_artifacts()

    def _load_artifacts(self):
        preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
        label_encoder_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
        best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
        all_models_path = os.path.join(MODELS_DIR, "all_models.joblib")

        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
        if os.path.exists(label_encoder_path):
            self.label_encoder = joblib.load(label_encoder_path)
        if os.path.exists(best_model_path):
            self.best_model = joblib.load(best_model_path)
        if os.path.exists(all_models_path):
            self.all_models = joblib.load(all_models_path)

    def is_ready(self):
        return (
            self.preprocessor is not None
            and self.label_encoder is not None
            and (self.best_model is not None or self.all_models is not None)
        )

    def get_available_models(self):
        if self.all_models:
            return list(self.all_models.keys())
        return ["Best Model"] if self.best_model else []

    def generate_explanation(self, fertilizer, n, p, k, soil_type, crop_type):
        """
        Generates agronomic explanation for why the model recommended this fertilizer.
        """
        explanations = {
            "Urea": (
                f"**Urea (46-0-0)** is recommended because of high vegetative nitrogen demand and low soil nitrogen ({n}). "
                f"Urea supplies 46% readily available nitrogen to boost leafy canopy growth, chlorophyll synthesis, and vegetative vigor in {crop_type} grown in {soil_type} soil."
            ),
            "DAP": (
                f"**DAP (Diammonium Phosphate 18-46-0)** is recommended due to low soil phosphorus ({p}). "
                f"DAP provides 46% water-soluble phosphate for rapid root proliferation, early seedling vigor, and uniform plant establishment for {crop_type}."
            ),
            "14-35-14": (
                f"**14-35-14 (NPK Complex)** is recommended because your soil indicates high phosphate demand ({p}) combined with moderate nitrogen and potassium requirements. "
                f"It fosters deep root anchoring and early flower bud differentiation for {crop_type}."
            ),
            "28-28": (
                f"**28-28-0 (Nitrophosphate Complex)** is recommended to supply an equal boost of nitrogen and phosphorus ({n} N, {p} P). "
                f"This balance accelerates stem elongation and tillering without causing excessive potassium accumulation."
            ),
            "20-20": (
                f"**20-20-0 (Ammonium Phosphate Sulphate)** is recommended as a balanced starter fertilizer. "
                f"It provides a 1:1 ratio of nitrogen and phosphorus, ideal for early vegetative and root development in {soil_type} soil."
            ),
            "17-17-17": (
                f"**17-17-17 (Complete Balanced NPK)** is recommended because soil analysis indicates a balanced requirement across all three primary macronutrients ({n} N, {p} P, {k} K). "
                f"It ensures harmonious vegetative, reproductive, and root development."
            ),
            "10-26-26": (
                f"**10-26-26 (High Potash & Phosphate Complex)** is recommended due to elevated potassium ({k}) and phosphorus ({p}) requirements relative to nitrogen. "
                f"Potassium enhances drought resistance, disease tolerance, grain weight, and harvest quality in {crop_type}."
            )
        }

        default_exp = (
            f"**{fertilizer}** is recommended based on the soil nutrient profile (N:{n}, P:{p}, K:{k}), "
            f"soil characteristics ({soil_type}), and nutritional requirements of {crop_type}."
        )

        return explanations.get(fertilizer, default_exp)

    def predict(self, input_data: dict, model_name: str = None):
        """
        input_data must have:
        ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
        """
        if not self.is_ready():
            self._load_artifacts()
            if not self.is_ready():
                raise RuntimeError("Models are not trained yet. Please run training pipeline first.")

        df_input = pd.DataFrame([input_data])

        # Normalize column names
        df_input.columns = df_input.columns.str.strip()

        # Select model
        selected_model = self.best_model
        used_model_name = "Best Model (Auto-selected)"
        if model_name and self.all_models and model_name in self.all_models:
            selected_model = self.all_models[model_name]
            used_model_name = model_name

        # Transform inputs using fitted preprocessor
        X_proc = self.preprocessor.transform(df_input)

        # Predict
        pred_encoded = selected_model.predict(X_proc)[0]
        fertilizer_name = self.label_encoder.inverse_transform([pred_encoded])[0]

        # Check probability support
        confidence = None
        probabilities = {}
        if hasattr(selected_model, "predict_proba"):
            probs = selected_model.predict_proba(X_proc)[0]
            classes = self.label_encoder.classes_
            for cls_name, prob in zip(classes, probs):
                probabilities[cls_name] = float(prob)
            confidence = float(np.max(probs))

        # Agronomic explanation
        n = input_data.get("Nitrogen", 0)
        p = input_data.get("Phosphorous", 0)
        k = input_data.get("Potassium", 0)
        soil = input_data.get("Soil Type", "Unknown")
        crop = input_data.get("Crop Type", "Unknown")
        explanation = self.generate_explanation(fertilizer_name, n, p, k, soil, crop)

        # Nutrient balance analysis
        nutrient_balance = {
            "nitrogen_status": "Low" if n < 15 else ("High" if n > 30 else "Moderate"),
            "phosphorous_status": "Low" if p < 15 else ("High" if p > 30 else "Moderate"),
            "potassium_status": "Low" if k < 5 else ("High" if k > 12 else "Moderate")
        }

        return {
            "recommended_fertilizer": fertilizer_name,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_used": used_model_name,
            "explanation": explanation,
            "nutrient_balance": nutrient_balance
        }


# Global predictor instance
predictor = FertilizerPredictor()
