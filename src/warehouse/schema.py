"""
PropIQ - Star Schema Data Modeling Layer
Defines Kimball Star Schema for Real Estate Analytics:
- Fact: fact_property_listings
- Dimensions: dim_location, dim_property_type, dim_amenities, dim_date
"""

import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple

class StarSchemaBuilder:
    @staticmethod
    def generate_surrogate_key(*args) -> str:
        """Generates deterministic surrogate key hash from attribute tuples"""
        key_str = "|".join(str(a).strip().lower() for a in args)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def build_dim_date(self, start_date: str = "2025-01-01", end_date: str = "2026-12-31") -> pd.DataFrame:
        """Generates comprehensive calendar date dimension table"""
        dates = pd.date_range(start=start_date, end=end_date)
        records = []
        for d in dates:
            date_key = int(d.strftime("%Y%m%d"))
            records.append({
                "date_key": date_key,
                "full_date": d.strftime("%Y-%m-%d"),
                "day_of_month": d.day,
                "day_name": d.strftime("%A"),
                "day_of_week": d.dayofweek + 1,
                "month": d.month,
                "month_name": d.strftime("%B"),
                "quarter": d.quarter,
                "year": d.year,
                "year_month": d.strftime("%Y-%m"),
                "is_weekend": 1 if d.dayofweek in [5, 6] else 0
            })
        return pd.DataFrame(records)

    def build_dim_location(self, cleaned_df: pd.DataFrame, api_locality_df: pd.DataFrame) -> pd.DataFrame:
        """Builds Location dimension enriched with API infrastructure metrics"""
        # Unique combinations of city and locality
        loc_grouped = cleaned_df[["city", "locality"]].drop_duplicates().copy()
        
        # Merge with API locality scores if available
        if not api_locality_df.empty:
            merged = pd.merge(
                loc_grouped,
                api_locality_df,
                left_on=["city", "locality"],
                right_on=["city", "locality_name"],
                how="left"
            )
        else:
            merged = loc_grouped

        # Fill default benchmark infrastructure scores
        merged["transit_score"] = merged.get("transit_score", pd.Series(85, index=merged.index)).fillna(85).astype(int)
        merged["safety_score"] = merged.get("safety_score", pd.Series(85, index=merged.index)).fillna(85).astype(int)
        merged["school_score"] = merged.get("school_score", pd.Series(85, index=merged.index)).fillna(85).astype(int)
        merged["metro_distance_km"] = merged.get("metro_distance_km", pd.Series(1.5, index=merged.index)).fillna(1.5)
        merged["hospital_count"] = merged.get("hospital_count", pd.Series(5, index=merged.index)).fillna(5).astype(int)

        records = []
        for _, row in merged.iterrows():
            loc_key = self.generate_surrogate_key("loc", row["city"], row["locality"])
            records.append({
                "location_key": loc_key,
                "city": row["city"],
                "locality": row["locality"],
                "transit_score": row["transit_score"],
                "safety_score": row["safety_score"],
                "school_score": row["school_score"],
                "metro_distance_km": row["metro_distance_km"],
                "hospital_count": row["hospital_count"]
            })

        return pd.DataFrame(records).drop_duplicates(subset=["location_key"])

    def build_dim_property_type(self, cleaned_df: pd.DataFrame) -> pd.DataFrame:
        """Builds Property Type & Furnishing dimension"""
        prop_types = cleaned_df[["bhk_count", "property_type", "furnishing_status"]].drop_duplicates().copy()
        records = []
        for _, row in prop_types.iterrows():
            type_key = self.generate_surrogate_key("prop", row["bhk_count"], row["property_type"], row["furnishing_status"])
            records.append({
                "property_type_key": type_key,
                "bhk_count": int(row["bhk_count"]),
                "bhk_label": f"{int(row['bhk_count'])} BHK",
                "property_category": row["property_type"],
                "furnishing_status": row["furnishing_status"]
            })
        return pd.DataFrame(records)

    def build_dim_amenities(self, cleaned_df: pd.DataFrame) -> pd.DataFrame:
        """Builds Amenities dimension with feature bundling tiers"""
        amenity_cols = ["parking", "gym", "swimming_pool", "clubhouse", "security", "power_backup"]
        unique_amenities = cleaned_df[amenity_cols].drop_duplicates().copy()
        records = []
        for _, row in unique_amenities.iterrows():
            amenity_key = self.generate_surrogate_key("amen", row["parking"], row["gym"], row["swimming_pool"], row["clubhouse"])
            total_amenities = sum([row[c] for c in amenity_cols])
            tier = "Luxury" if total_amenities >= 5 else ("Standard" if total_amenities >= 3 else "Basic")
            records.append({
                "amenity_key": amenity_key,
                "has_parking": int(row["parking"]),
                "has_gym": int(row["gym"]),
                "has_swimming_pool": int(row["swimming_pool"]),
                "has_clubhouse": int(row["clubhouse"]),
                "has_security": int(row["security"]),
                "has_power_backup": int(row["power_backup"]),
                "amenity_tier": tier
            })
        return pd.DataFrame(records)

    def build_fact_listings(
        self,
        cleaned_df: pd.DataFrame,
        dim_location: pd.DataFrame,
        dim_property_type: pd.DataFrame,
        dim_amenities: pd.DataFrame
    ) -> pd.DataFrame:
        """Builds Fact Table with surrogate foreign keys and numerical measures"""
        fact = cleaned_df.copy()

        # Map location surrogate key
        loc_map = dict(zip(zip(dim_location["city"], dim_location["locality"]), dim_location["location_key"]))
        fact["location_key"] = fact.apply(lambda r: loc_map.get((r["city"], r["locality"]), "LOC-UNKNOWN"), axis=1)

        # Map property type surrogate key
        type_map = dict(zip(zip(dim_property_type["bhk_count"], dim_property_type["property_category"], dim_property_type["furnishing_status"]), dim_property_type["property_type_key"]))
        fact["property_type_key"] = fact.apply(lambda r: type_map.get((r["bhk_count"], r["property_type"], r["furnishing_status"]), "PROP-UNKNOWN"), axis=1)

        # Map amenities surrogate key
        amen_map = dict(zip(zip(dim_amenities["has_parking"], dim_amenities["has_gym"], dim_amenities["has_swimming_pool"], dim_amenities["has_clubhouse"]), dim_amenities["amenity_key"]))
        fact["amenity_key"] = fact.apply(lambda r: amen_map.get((r["parking"], r["gym"], r["swimming_pool"], r["clubhouse"]), "AMEN-UNKNOWN"), axis=1)

        # Map date key
        fact["posted_date_dt"] = pd.to_datetime(fact["posted_date"], errors="coerce").fillna(pd.Timestamp("2026-01-15"))
        fact["date_key"] = fact["posted_date_dt"].dt.strftime("%Y%m%d").astype(int)
        fact["year_month"] = fact["posted_date_dt"].dt.strftime("%Y-%m")

        # Days on market
        now = pd.Timestamp.now()
        fact["days_on_market"] = (now - fact["posted_date_dt"]).dt.days.clip(lower=1)

        fact_table = fact[[
            "listing_id",
            "date_key",
            "location_key",
            "property_type_key",
            "amenity_key",
            "city",
            "year_month",
            "price_inr",
            "rent_inr",
            "carpet_area_sqft",
            "price_per_sqft",
            "rental_yield_pct",
            "bathrooms",
            "balconies",
            "days_on_market",
            "latitude",
            "longitude",
            "source_portal"
        ]].copy()

        return fact_table
