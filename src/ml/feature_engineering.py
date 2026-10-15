"""
PropIQ - Machine Learning Feature Engineering
Prepares numerical, spatial, and categorical features for price & rent modeling.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        is_training: bool = False,
        target_col: str = "price_inr"
    ) -> Tuple[np.ndarray, Any, List[str]]:
        """Extracts and scales model features, returns X and y matrices"""
        df_feats = df.copy()

        # Auto-enrich with dimensional tables if keys exist and columns are missing
        if "location_key" in df_feats.columns and ("transit_score" not in df_feats.columns or df_feats["transit_score"].isnull().all()):
            dim_loc_path = os.path.join(BASE_DIR, "data/warehouse/dim_location.parquet")
            if os.path.exists(dim_loc_path):
                dim_loc = pd.read_parquet(dim_loc_path)
                join_cols = [c for c in ["location_key", "transit_score", "safety_score", "school_score"] if c in dim_loc.columns]
                df_feats = df_feats.drop(columns=[c for c in ["transit_score", "safety_score", "school_score"] if c in df_feats.columns], errors="ignore")
                df_feats = df_feats.merge(dim_loc[join_cols], on="location_key", how="left")

        if "property_type_key" in df_feats.columns and ("bhk_count" not in df_feats.columns or "furnishing_status" not in df_feats.columns):
            dim_prop_path = os.path.join(BASE_DIR, "data/warehouse/dim_property_type.parquet")
            if os.path.exists(dim_prop_path):
                dim_prop = pd.read_parquet(dim_prop_path)
                join_cols = [c for c in ["property_type_key", "bhk_count", "furnishing_status"] if c in dim_prop.columns]
                df_feats = df_feats.drop(columns=[c for c in ["bhk_count", "furnishing_status"] if c in df_feats.columns], errors="ignore")
                df_feats = df_feats.merge(dim_prop[join_cols], on="property_type_key", how="left")

        # Handle furnishing boolean/numeric
        if "furnishing_status" in df_feats.columns:
            df_feats["is_furnished"] = df_feats["furnishing_status"].apply(
                lambda x: 1 if str(x).lower() in ["fully furnished", "semi-furnished"] else 0
            )
        else:
            if "is_furnished" not in df_feats.columns:
                df_feats["is_furnished"] = 0

        # Ensure infrastructure columns present
        for col in ["transit_score", "safety_score", "school_score"]:
            if col not in df_feats.columns:
                df_feats[col] = 85.0

        for col in ["parking", "gym", "swimming_pool"]:
            if col not in df_feats.columns:
                df_feats[col] = 1.0

        if "bhk_count" not in df_feats.columns:
            df_feats["bhk_count"] = 2.0

        if "bathrooms" not in df_feats.columns:
            df_feats["bathrooms"] = 2.0

        if "balconies" not in df_feats.columns:
            df_feats["balconies"] = 1.0

        if "carpet_area_sqft" not in df_feats.columns:
            df_feats["carpet_area_sqft"] = 1100.0

        if "latitude" not in df_feats.columns:
            df_feats["latitude"] = 28.56

        if "longitude" not in df_feats.columns:
            df_feats["longitude"] = 77.38

        X_raw = df_feats[FEATURE_COLUMNS].copy()

        # Fill any missing values with median or default
        medians = X_raw.median()
        medians = medians.fillna(0.0)
        X_raw = X_raw.fillna(medians)

        if is_training:
            X_scaled = self.scaler.fit_transform(X_raw)
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X_raw)

        # Target extraction
        if target_col == "both":
            y_price = df_feats["price_inr"].values if "price_inr" in df_feats.columns else None
            y_rent = df_feats["rent_inr"].values if "rent_inr" in df_feats.columns else None
            return X_scaled, (y_price, y_rent), FEATURE_COLUMNS
        elif target_col and target_col in df_feats.columns:
            y = df_feats[target_col].values
            return X_scaled, y, FEATURE_COLUMNS
        else:
            y = df_feats["price_inr"].values if "price_inr" in df_feats.columns else None
            return X_scaled, y, FEATURE_COLUMNS
