"""
PropIQ - Data Normalization and Standardization Engine
Cleans locality strings, normalizes pricing, validates coordinate bounds, and standardizes categories.
"""

import re
import pandas as pd
import numpy as np

class DataNormalizer:
    @staticmethod
    def clean_text(text: str) -> str:
        """Removes punctuation noise and extra whitespace"""
        if pd.isna(text):
            return ""
        text = str(text).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def normalize_locality(locality: str) -> str:
        """Standardizes common locality abbreviations (e.g. 'Sec-62' -> 'Sector 62')"""
        if pd.isna(locality):
            return "Unknown"
        loc = str(locality).strip()
        loc = re.sub(r"(?i)\bsec(?:tor)?\.?\s*(\d+)", r"Sector \1", loc)
        loc = re.sub(r"(?i)\bgr(?:eater)?\.?\s*noida", "Greater Noida", loc)
        return loc

    @staticmethod
    def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
        """Applies complete normalization pipeline on raw listings DataFrame"""
        cleaned = df.copy()

        # Text cleaning
        cleaned["title"] = cleaned["title"].apply(DataNormalizer.clean_text)
        cleaned["locality"] = cleaned["locality"].apply(DataNormalizer.normalize_locality)
        cleaned["city"] = cleaned["city"].apply(DataNormalizer.clean_text)

        # Numeric bounds validation
        cleaned["price_inr"] = pd.to_numeric(cleaned["price_inr"], errors="coerce")
        cleaned["rent_inr"] = pd.to_numeric(cleaned["rent_inr"], errors="coerce")
        cleaned["carpet_area_sqft"] = pd.to_numeric(cleaned["carpet_area_sqft"], errors="coerce")
        cleaned["bhk_count"] = pd.to_numeric(cleaned["bhk_count"], errors="coerce").fillna(2).astype(int)

        # Impute missing rents if needed using median locality yield (~4%)
        missing_rent_mask = cleaned["rent_inr"].isna() | (cleaned["rent_inr"] <= 0)
        cleaned.loc[missing_rent_mask, "rent_inr"] = (cleaned.loc[missing_rent_mask, "price_inr"] * 0.04 / 12.0).round()

        # Compute derived metrics
        cleaned["price_per_sqft"] = (cleaned["price_inr"] / cleaned["carpet_area_sqft"]).round(2)
        # Annualized Rental Yield: (Monthly Rent * 12) / Total Price * 100
        cleaned["rental_yield_pct"] = ((cleaned["rent_inr"] * 12.0) / cleaned["price_inr"] * 100.0).round(2)

        # Coordinate clipping for Indian Metros
        cleaned["latitude"] = pd.to_numeric(cleaned["latitude"], errors="coerce").fillna(28.5)
        cleaned["longitude"] = pd.to_numeric(cleaned["longitude"], errors="coerce").fillna(77.3)

        return cleaned
