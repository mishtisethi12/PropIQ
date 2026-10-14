"""
PropIQ - Machine Learning Feature Engineering
Prepares numerical, spatial, and categorical features for price & rent modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

FEATURE_COLUMNS = [
    "carpet_area_sqft",
    "bhk_count",
    "bathrooms",
    "balconies",
    "parking",
    "gym",
    "swimming_pool",
    "transit_score",
    "safety_score",
    "school_score",
    "is_furnished",
    "latitude",
    "longitude"
]

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def transform_dataframe(self, df: pd.DataFrame, is_training: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extracts and scales model features, returns X and y matrices"""
        df_feats = df.copy()

        # Handle furnishing boolean/numeric
        if "furnishing_status" in df_feats.columns:
            df_feats["is_furnished"] = df_feats["furnishing_status"].apply(
                lambda x: 1 if str(x).lower() in ["fully furnished", "semi-furnished"] else 0
            )
        else:
            df_feats["is_furnished"] = 0

        # Ensure infrastructure columns present
        for col in ["transit_score", "safety_score", "school_score"]:
            if col not in df_feats.columns:
                df_feats[col] = 85

        for col in ["parking", "gym", "swimming_pool"]:
            if col not in df_feats.columns:
                df_feats[col] = 1

        X_raw = df_feats[FEATURE_COLUMNS].copy()

        # Fill any missing values
        X_raw = X_raw.fillna(X_raw.median())

        if is_training:
            X_scaled = self.scaler.fit_transform(X_raw)
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X_raw)

        y = df_feats["price_inr"].values if "price_inr" in df_feats.columns else None

        return X_scaled, y, FEATURE_COLUMNS
