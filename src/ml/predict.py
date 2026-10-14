"""
PropIQ - Multi-Model Inference & Valuation Engine
Serves live predictions across Decision Forest, k-NN, XGBoost, and Ridge.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

class PropertyValuationService:
    def __init__(self, artifacts_dir: str = "artifacts/ml"):
        self.artifacts_dir = artifacts_dir
        self.models_path = os.path.join(artifacts_dir, "models_bundle.joblib")
        self.fe_path = os.path.join(artifacts_dir, "feature_engineer.joblib")
        
        if os.path.exists(self.models_path) and os.path.exists(self.fe_path):
            self.models = joblib.load(self.models_path)
            self.fe = joblib.load(self.fe_path)
            self.is_ready = True
        else:
            self.is_ready = False

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
        for model_name, model in self.models.items():
            pred_price = float(model.predict(X_scaled)[0])
            pred_price = max(100000.0, pred_price)
            # Estimate rent (typically 3.8% to 4.5% annual yield)
            est_rent = round((pred_price * 0.042) / 12.0)
            yield_pct = round(((est_rent * 12.0) / pred_price) * 100.0, 2)

            predictions[model_name] = {
                "predicted_price_inr": round(pred_price),
                "predicted_price_lakhs": round(pred_price / 100000.0, 2),
                "estimated_monthly_rent_inr": est_rent,
                "projected_rental_yield_pct": yield_pct
            }

        # Ensemble consensus (average of Decision Forest and XGBoost)
        ensemble_price = (predictions["Decision Forest"]["predicted_price_inr"] + 
                          predictions["XGBoost"]["predicted_price_inr"]) / 2.0
        ensemble_rent = round((ensemble_price * 0.042) / 12.0)
        
        predictions["Consensus Best Estimate"] = {
            "predicted_price_inr": round(ensemble_price),
            "predicted_price_lakhs": round(ensemble_price / 100000.0, 2),
            "estimated_monthly_rent_inr": ensemble_rent,
            "projected_rental_yield_pct": round(((ensemble_rent * 12.0) / ensemble_price) * 100.0, 2)
        }

        return predictions
