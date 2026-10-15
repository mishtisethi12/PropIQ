"""
PropIQ - Multi-Model Inference & Valuation Engine
Serves live predictions across Decision Forest, k-NN, Naive Bayes, SVM, and XGBoost.
Generates dual valuations for both Property Price and Monthly Rental Yield.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml.train import NaiveBayesBinnedRegressor
from src.ml.feature_engineering import FeatureEngineer, FEATURE_COLUMNS


class PropertyValuationService:
    def __init__(self, artifacts_dir: str = "artifacts/ml"):
        self.artifacts_dir = os.path.join(BASE_DIR, artifacts_dir) if not os.path.isabs(artifacts_dir) else artifacts_dir
        self.models_path = os.path.join(self.artifacts_dir, "models_bundle.joblib")
        self.fe_path = os.path.join(self.artifacts_dir, "feature_engineer.joblib")

        if os.path.exists(self.models_path) and os.path.exists(self.fe_path):
            self.bundle = joblib.load(self.models_path)
            self.fe = joblib.load(self.fe_path)
            self.is_ready = True

            if isinstance(self.bundle, dict) and "price" in self.bundle and "rent" in self.bundle:
                self.price_models = self.bundle["price"]
                self.rent_models = self.bundle["rent"]
            else:
                self.price_models = self.bundle
                self.rent_models = {}
        else:
            self.is_ready = False
            self.price_models = {}
            self.rent_models = {}

    def predict_property(self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs valuation on property input dictionary:
        {
            'carpet_area_sqft': 1150,
            'bhk_count': 2,
            'bathrooms': 2,
            'balconies': 2,
            'parking': 1,
            'gym': 1,
            'swimming_pool': 1,
            'transit_score': 88,
            'safety_score': 85,
            'school_score': 87,
            'is_furnished': 1,
            'latitude': 28.56,
            'longitude': 77.38
        }
        """
        if not self.is_ready:
            raise RuntimeError("Models not yet trained or artifacts not found.")

        df_input = pd.DataFrame([input_features])
        X_scaled, _, _ = self.fe.transform_dataframe(df_input, is_training=False)

        predictions = {}
        # Iterate over all price models
        for model_name, p_model in self.price_models.items():
            pred_price = float(p_model.predict(X_scaled)[0])
            pred_price = max(100000.0, pred_price)

            # Predict rent with matching rent model if available
            if model_name in self.rent_models:
                pred_rent = float(self.rent_models[model_name].predict(X_scaled)[0])
                pred_rent = max(3000.0, pred_rent)
            else:
                pred_rent = round((pred_price * 0.041) / 12.0)

            yield_pct = round(((pred_rent * 12.0) / pred_price) * 100.0, 2)

            predictions[model_name] = {
                "predicted_price_inr": round(pred_price),
                "predicted_price_lakhs": round(pred_price / 100000.0, 2),
                "estimated_monthly_rent_inr": round(pred_rent),
                "projected_rental_yield_pct": yield_pct
            }

        # Ensemble consensus (average of Decision Forest, SVM, and XGBoost if available)
        consensus_models = [m for m in ["Decision Forest", "SVM", "XGBoost"] if m in predictions]
        if consensus_models:
            avg_price = float(np.mean([predictions[m]["predicted_price_inr"] for m in consensus_models]))
            avg_rent = float(np.mean([predictions[m]["estimated_monthly_rent_inr"] for m in consensus_models]))
        else:
            avg_price = float(np.mean([p["predicted_price_inr"] for p in predictions.values()]))
            avg_rent = float(np.mean([p["estimated_monthly_rent_inr"] for p in predictions.values()]))

        avg_yield = round(((avg_rent * 12.0) / avg_price) * 100.0, 2)

        predictions["Consensus Best Estimate"] = {
            "predicted_price_inr": round(avg_price),
            "predicted_price_lakhs": round(avg_price / 100000.0, 2),
            "estimated_monthly_rent_inr": round(avg_rent),
            "projected_rental_yield_pct": avg_yield
        }

        return predictions
