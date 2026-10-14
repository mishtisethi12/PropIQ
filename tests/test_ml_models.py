import os
import json
import pytest
from src.ml.predict import PropertyValuationService

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "artifacts/ml/model_comparison_metrics.json")

def test_ml_metrics_generated():
    assert os.path.exists(METRICS_PATH), "Model metrics JSON file missing"
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "metrics" in data
    assert "Decision Forest" in data["metrics"]
    assert "k-NN" in data["metrics"]
    assert "XGBoost" in data["metrics"]
    assert data["metrics"]["Decision Forest"]["r2_score"] > 0.85

def test_valuation_service_prediction():
    service = PropertyValuationService(os.path.join(BASE_DIR, "artifacts/ml"))
    input_data = {
        "carpet_area_sqft": 1050,
        "bhk_count": 2,
        "bathrooms": 2,
        "balconies": 2,
        "parking": 1,
        "gym": 1,
        "swimming_pool": 1,
        "transit_score": 88,
        "safety_score": 85,
        "school_score": 87,
        "is_furnished": 1,
        "latitude": 28.56,
        "longitude": 77.38
    }
    preds = service.predict_property(input_data)
    assert "Decision Forest" in preds
    assert "k-NN" in preds
    assert "XGBoost" in preds
    assert preds["Decision Forest"]["predicted_price_inr"] > 500000
    assert preds["Decision Forest"]["projected_rental_yield_pct"] > 0
