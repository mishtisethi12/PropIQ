"""
PropIQ - Multi-Source CSV Ingestion Engine
Handles ingestion of heterogeneous real estate listings:
- Portal Alpha: Standard structured CSV format
- Portal Beta: Messy semi-structured format with embedded units, string coordinates, and lakh prices
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Any

class CSVIngestionEngine:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def ingest_portal_alpha(self, file_path: str) -> pd.DataFrame:
        """Ingests cleanly formatted Portal Alpha listings"""
        df = pd.read_csv(file_path)
        df["source_portal"] = "Portal_Alpha"
        df["ingested_at"] = pd.Timestamp.now().isoformat()
        return df

    def ingest_portal_beta(self, file_path: str) -> pd.DataFrame:
        """
        Ingests messy Portal Beta listings and maps them to a standardized raw format:
        - Parses `bedrooms` like '2 BHK' -> int 2
        - Parses `sqft` like '1050 sq.ft' -> float 1050.0
        - Converts `cost_lakhs` to total INR (lakhs * 100,000)
        - Parses `lat_long` string into float latitude and longitude
        - Parses `location_text` into locality and city
        """
        raw_df = pd.read_csv(file_path)
        parsed_records = []

        for _, row in raw_df.iterrows():
            # Parse BHK
            bhk_match = re.search(r"(\d+)", str(row.get("bedrooms", "")))
            bhk = int(bhk_match.group(1)) if bhk_match else 2

            # Parse Carpet Area
            sqft_match = re.search(r"(\d+(\.\d+)?)", str(row.get("sqft", "")))
            carpet_sqft = float(sqft_match.group(1)) if sqft_match else 1000.0

            # Convert Lakhs to INR
            cost_lakhs = float(row.get("cost_lakhs", 0))
            price_inr = int(cost_lakhs * 100000)

            # Monthly rent
            rent_inr = int(row.get("monthly_rent_inr", 0))

            # Parse Lat/Long
            lat, lng = 0.0, 0.0
            lat_long_str = str(row.get("lat_long", ""))
            if "," in lat_long_str:
                parts = lat_long_str.split(",")
                try:
                    lat = float(parts[0].strip())
                    lng = float(parts[1].strip())
                except ValueError:
                    pass

            # Extract locality from location_text
            loc_text = str(row.get("location_text", ""))
            parts = [p.strip() for p in loc_text.split(",")]
            locality = parts[0] if parts else "Unknown"
            city = row.get("city_name", parts[1] if len(parts) > 1 else "Unknown")

            # Parse amenities string
            amenities_str = str(row.get("amenities_list", ""))
            parking = 1 if "Parking: 1" in amenities_str else 0
            gym = 1 if "Gym: 1" in amenities_str else 0
            pool = 1 if "Pool: 1" in amenities_str else 0
            security = 1 if "Security: 1" in amenities_str else 1

            parsed_records.append({
                "listing_id": str(row.get("id")),
                "title": str(row.get("prop_name")),
                "city": city,
                "locality": locality,
                "property_type": "Apartment",
                "bhk_count": bhk,
                "carpet_area_sqft": carpet_sqft,
                "price_inr": price_inr,
                "rent_inr": rent_inr,
                "bathrooms": int(row.get("baths", bhk)),
                "balconies": max(1, bhk - 1),
                "furnishing_status": str(row.get("furnish", "Semi-Furnished")),
                "parking": parking,
                "gym": gym,
                "swimming_pool": pool,
                "clubhouse": pool,
                "security": security,
                "power_backup": 1,
                "latitude": lat,
                "longitude": lng,
                "posted_date": str(row.get("upload_ts", ""))[:10],
                "source_portal": "Portal_Beta",
                "ingested_at": pd.Timestamp.now().isoformat()
            })

        return pd.DataFrame(parsed_records)

    def load_all_sources(self, alpha_path: str, beta_path: str) -> pd.DataFrame:
        """Loads and combines all CSV sources into unified raw listings DataFrame"""
        df_alpha = self.ingest_portal_alpha(alpha_path)
        df_beta = self.ingest_portal_beta(beta_path)
        combined_df = pd.concat([df_alpha, df_beta], ignore_index=True)
        return combined_df
