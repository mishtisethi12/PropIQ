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
    assert "price_metrics" in data
    assert "rent_metrics" in data

    # Required 4 models for Price
    for model_name in ["Decision Forest", "k-NN", "Naive Bayes", "SVM"]:
        assert model_name in data["price_metrics"], f"Missing {model_name} in price_metrics"
        m = data["price_metrics"][model_name]
        assert m["mae_inr"] > 0
        assert "mae_ci_lower" in m
        assert "mae_ci_upper" in m
        assert m["mae_ci_lower"] <= m["mae_inr"] <= m["mae_ci_upper"]
        assert m["r2_score"] > 0.70

    # Required 4 models for Rent
    for model_name in ["Decision Forest", "k-NN", "Naive Bayes", "SVM"]:
        assert model_name in data["rent_metrics"], f"Missing {model_name} in rent_metrics"
        m = data["rent_metrics"][model_name]
        assert m["mae_inr"] > 0
        assert "mae_ci_lower" in m
        assert "mae_ci_upper" in m
        assert m["mae_ci_lower"] <= m["mae_inr"] <= m["mae_ci_upper"]
        assert m["r2_score"] > 0.70

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
    for model_name in ["Decision Forest", "k-NN", "Naive Bayes", "SVM"]:
        assert model_name in preds, f"Missing prediction for {model_name}"
        assert preds[model_name]["predicted_price_inr"] > 500000
        assert preds[model_name]["estimated_monthly_rent_inr"] > 5000
        assert preds[model_name]["projected_rental_yield_pct"] > 0
    assert "Consensus Best Estimate" in preds
