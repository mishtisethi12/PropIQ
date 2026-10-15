"""
PropIQ - Model Training, Evaluation & Cross-Model Comparison Pipeline
Trains, benchmarks and compares:
1. Decision Forest (Random Forest Regressor)
2. k-Nearest Neighbors (k-NN Regressor)
3. Naive Bayes Regressor (Discretized/Binned Target Expected Value)
4. Support Vector Machine (SVM / SVR Regressor)
5. Gradient Boosting (XGBoost Regressor)

Supports both Price (INR) and Rent (INR/month) targets.
Evaluates MAE, R2, RMSE, MAPE, plus 95% Bootstrap Confidence Intervals.
"""

import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml.feature_engineering import FeatureEngineer, FEATURE_COLUMNS


from src.ml.models import NaiveBayesBinnedRegressor, compute_bootstrap_ci


def get_model_definitions() -> Dict[str, Any]:
    """Returns standardized instances of models to be trained and compared"""
    return {
        "Decision Forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=14,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        "k-NN": KNeighborsRegressor(
            n_neighbors=7,
            weights="distance"
        ),
        "Naive Bayes": NaiveBayesBinnedRegressor(
            n_bins=15,
            strategy="quantile"
        ),
        "SVM": TransformedTargetRegressor(
            regressor=SVR(C=25.0, epsilon=0.05, kernel="rbf"),
            transformer=StandardScaler()
        ),
        "XGBoost": XGBRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
    }


def train_single_target(
    models: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_name: str = "Price"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains a suite of models on a given target and evaluates metrics with bootstrap CI"""
    metrics_dict = {}
    fitted_models = {}

    print(f"\nTraining & Benchmarking Models for Target: [{target_name}]")
    print("-" * 65)

    for name, model_tmpl in models.items():
        # Create fresh clone
        from sklearn.base import clone
        try:
            model = clone(model_tmpl)
        except Exception:
            model = model_tmpl

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        inference_time_ms = ((time.perf_counter() - t1) / len(X_test)) * 1000.0

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100.0)

        # 95% Bootstrap Confidence Interval on error
        ci_stats = compute_bootstrap_ci(y_test, y_pred, n_bootstraps=1000, confidence=0.95, random_state=42)

        metrics_dict[name] = {
            "target": target_name,
            "r2_score": round(r2, 4),
            "r2_ci_lower": ci_stats["r2_ci_lower"],
            "r2_ci_upper": ci_stats["r2_ci_upper"],
            "mae_inr": round(mae, 2),
            "mae_ci_lower": ci_stats["mae_ci_lower"],
            "mae_ci_upper": ci_stats["mae_ci_upper"],
            "mae_ci_margin": ci_stats["mae_ci_margin"],
            "rmse_inr": round(rmse, 2),
            "mape_pct": round(mape, 2),
            "training_time_sec": round(train_time, 4),
            "inference_latency_per_sample_ms": round(inference_time_ms, 4),
            "bootstrap_samples": ci_stats["bootstrap_samples"]
        }
        fitted_models[name] = model

        print(
            f"  [{name:15}] -> MAE: INR {mae:,.0f} (95% CI: [{ci_stats['mae_ci_lower']:,.0f} - {ci_stats['mae_ci_upper']:,.0f}]) | "
            f"R2: {r2:.4f} (95% CI: [{ci_stats['r2_ci_lower']:.4f} - {ci_stats['r2_ci_upper']:.4f}]) | "
            f"Latency: {inference_time_ms:.3f}ms"
        )

    return metrics_dict, fitted_models


def train_and_compare_models(data_df: pd.DataFrame, artifacts_dir: str = "artifacts/ml") -> dict:
    os.makedirs(artifacts_dir, exist_ok=True)

    print("\n========================================================")
    print("STEP 4: TRAINING & COMPARING MACHINE LEARNING MODELS")
    print("Decision Forest, k-NN, Naive Bayes, SVM, XGBoost")
    print("========================================================")

    fe = FeatureEngineer()
    X, (y_price, y_rent), feature_names = fe.transform_dataframe(data_df, is_training=True, target_col="both")

    # If rent is missing or zero, generate realistic rent proxy
    if y_rent is None or len(y_rent) == 0:
        y_rent = np.round((y_price * 0.041) / 12.0)

    # Proper Train/Test Split (80/20)
    X_train, X_test, y_p_train, y_p_test, y_r_train, y_r_test = train_test_split(
        X, y_price, y_rent, test_size=0.20, random_state=42
    )

    models_suite = get_model_definitions()

    # 1. Train Price Models
    price_metrics, price_models = train_single_target(
        models_suite, X_train, X_test, y_p_train, y_p_test, target_name="Property Price (INR)"
    )

    # 2. Train Rent Models
    rent_metrics, rent_models = train_single_target(
        models_suite, X_train, X_test, y_r_train, y_r_test, target_name="Monthly Rent (INR)"
    )

    # Extract Feature Importance from Decision Forest
    rf_price_model = price_models["Decision Forest"]
    rf_price_imps = rf_price_model.feature_importances_
    feat_importance_price = {
        feat: round(float(imp), 4)
        for feat, imp in sorted(zip(feature_names, rf_price_imps), key=lambda x: x[1], reverse=True)
    }

    rf_rent_model = rent_models["Decision Forest"]
    rf_rent_imps = rf_rent_model.feature_importances_
    feat_importance_rent = {
        feat: round(float(imp), 4)
        for feat, imp in sorted(zip(feature_names, rf_rent_imps), key=lambda x: x[1], reverse=True)
    }

    # Save artifact bundle: preserve both nested and top-level price model access for full backwards compatibility
    model_bundle = {
        "price": price_models,
        "rent": rent_models,
        # Direct top-level access for legacy consumers
        **price_models
    }

    joblib.dump(model_bundle, os.path.join(artifacts_dir, "models_bundle.joblib"))
    joblib.dump(fe, os.path.join(artifacts_dir, "feature_engineer.joblib"))

    full_payload = {
        "metrics": price_metrics,
        "price_metrics": price_metrics,
        "rent_metrics": rent_metrics,
        "feature_importance": feat_importance_price,
        "rent_feature_importance": feat_importance_rent,
        "metadata": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "split_ratio": "80/20",
            "bootstrap_samples": 1000,
            "confidence_level": 0.95
        }
    }

    comparison_path = os.path.join(artifacts_dir, "model_comparison_metrics.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print(f"\n[OK] Saved all trained models and evaluation benchmarks to {artifacts_dir}")
    return full_payload


if __name__ == "__main__":
    df = pd.read_parquet("data/warehouse/fact_property_listings.parquet")
    metrics = train_and_compare_models(df)
