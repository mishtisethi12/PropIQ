"""
PropIQ - Model Training & Cross-Model Comparison Pipeline
Trains and compares:
1. Decision Forest (Random Forest Regressor)
2. k-Nearest Neighbors (k-NN)
3. Gradient Boosting (XGBoost)
4. Ridge Linear Regression (Baseline)
"""

import os
import time
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.ml.feature_engineering import FeatureEngineer, FEATURE_COLUMNS

def train_and_compare_models(data_df: pd.DataFrame, artifacts_dir: str = "artifacts/ml") -> dict:
    os.makedirs(artifacts_dir, exist_ok=True)

    print("\n========================================================")
    print("STEP 4: TRAINING & COMPARING MACHINE LEARNING MODELS")
    print("========================================================")

    fe = FeatureEngineer()
    X, y, feature_names = fe.transform_dataframe(data_df, is_training=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Decision Forest": RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_split=4, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=120, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1),
        "k-NN": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "Ridge Baseline": Ridge(alpha=10.0)
    }

    results = {}
    trained_model_objs = {}

    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        inference_time_ms = ((time.perf_counter() - t1) / len(X_test)) * 1000

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100.0)

        results[name] = {
            "r2_score": round(r2, 4),
            "rmse_inr": round(rmse, 2),
            "mae_inr": round(mae, 2),
            "mape_pct": round(mape, 2),
            "training_time_sec": round(train_time, 4),
            "inference_latency_per_sample_ms": round(inference_time_ms, 4)
        }
        trained_model_objs[name] = model

        print(f"  [{name}] -> R2: {r2:.4f} | RMSE: INR {rmse:,.0f} | MAE: INR {mae:,.0f} | MAPE: {mape:.2f}% | Latency: {inference_time_ms:.3f}ms")

    # Feature Importance from Decision Forest
    rf_model = trained_model_objs["Decision Forest"]
    importances = rf_model.feature_importances_
    feat_importance_dict = {
        feat: round(float(imp), 4)
        for feat, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    }

    # Save artifacts
    joblib.dump(trained_model_objs, os.path.join(artifacts_dir, "models_bundle.joblib"))
    joblib.dump(fe, os.path.join(artifacts_dir, "feature_engineer.joblib"))

    comparison_path = os.path.join(artifacts_dir, "model_comparison_metrics.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": results,
            "feature_importance": feat_importance_dict
        }, f, indent=2)

    print(f"[OK] Saved all trained models and metrics to {artifacts_dir}")
    return results

if __name__ == "__main__":
    df = pd.read_parquet("data/warehouse/fact_property_listings.parquet")
    metrics = train_and_compare_models(df)
