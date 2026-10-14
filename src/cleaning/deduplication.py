"""
PropIQ - Intelligent Real Estate De-Duplication Engine
Detects and resolves cross-portal duplicate listings using:
1. Spatial-Attribute Blocking
2. Haversine Geo-Proximity Clustering (< 75 meters)
3. Price and Carpet Area Variance Thresholding
4. Audit Trail & Canonical Golden Record Creation
"""

import math
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

class RealEstateDeduplicator:
    def __init__(
        self,
        geo_proximity_meters: float = 75.0,
        price_variance_tol: float = 0.05,
        area_variance_tol: float = 0.05
    ):
        self.geo_proximity_meters = geo_proximity_meters
        self.price_variance_tol = price_variance_tol
        self.area_variance_tol = area_variance_tol

    @staticmethod
    def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Computes great-circle distance between two GPS coordinates in meters"""
        R = 6371000.0  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (math.sin(delta_phi / 2.0) ** 2 +
             math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return R * c

    def deduplicate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes deduplication across the dataset.
        Returns:
            - unique_df: Cleaned dataframe with one golden record per property
            - audit_df: Detailed record of duplicates found and merged
        """
        records = df.to_dict(orient="records")
        n = len(records)
        visited = [False] * n
        
        clusters: List[List[int]] = []
        audit_trail: List[Dict[str, Any]] = []

        # Step 1: Blocking by (city, bhk_count) to avoid O(N^2) global scan
        blocks: Dict[Tuple[str, int], List[int]] = {}
        for idx, rec in enumerate(records):
            block_key = (str(rec["city"]).strip().lower(), int(rec["bhk_count"]))
            blocks.setdefault(block_key, []).append(idx)

        # Step 2: Compare pairs within each block
        for block_key, indices in blocks.items():
            for i_idx, i in enumerate(indices):
                if visited[i]:
                    continue
                current_cluster = [i]
                visited[i] = True
                rec_a = records[i]

                for j in indices[i_idx + 1:]:
                    if visited[j]:
                        continue
                    rec_b = records[j]

                    # Check geo proximity
                    dist = self.haversine_distance_meters(
                        rec_a["latitude"], rec_a["longitude"],
                        rec_b["latitude"], rec_b["longitude"]
                    )
                    
                    if dist <= self.geo_proximity_meters:
                        # Check price variance
                        avg_price = (rec_a["price_inr"] + rec_b["price_inr"]) / 2.0
                        price_diff_ratio = abs(rec_a["price_inr"] - rec_b["price_inr"]) / (avg_price + 1e-5)

                        # Check area variance
                        avg_area = (rec_a["carpet_area_sqft"] + rec_b["carpet_area_sqft"]) / 2.0
                        area_diff_ratio = abs(rec_a["carpet_area_sqft"] - rec_b["carpet_area_sqft"]) / (avg_area + 1e-5)

                        if price_diff_ratio <= self.price_variance_tol and area_diff_ratio <= self.area_variance_tol:
                            visited[j] = True
                            current_cluster.append(j)

                clusters.append(current_cluster)

        # Step 3: Elect golden record for each cluster
        golden_records = []
        for cluster in clusters:
            if len(cluster) == 1:
                rec = records[cluster[0]]
                rec["is_duplicate"] = False
                rec["duplicate_count"] = 0
                rec["master_listing_id"] = rec["listing_id"]
                golden_records.append(rec)
            else:
                # Multiple listings of same property found
                cluster_recs = [records[k] for k in cluster]
                
                # Sort to prioritize Portal_Alpha as primary source if present
                cluster_recs.sort(key=lambda r: 0 if r.get("source_portal") == "Portal_Alpha" else 1)
                primary = cluster_recs[0]
                primary["is_duplicate"] = True
                primary["duplicate_count"] = len(cluster) - 1
                primary["master_listing_id"] = primary["listing_id"]
                golden_records.append(primary)

                for dup in cluster_recs[1:]:
                    audit_trail.append({
                        "master_listing_id": primary["listing_id"],
                        "master_source": primary.get("source_portal"),
                        "master_price": primary["price_inr"],
                        "duplicate_listing_id": dup["listing_id"],
                        "duplicate_source": dup.get("source_portal"),
                        "duplicate_price": dup["price_inr"],
                        "locality": primary["locality"],
                        "city": primary["city"],
                        "bhk": primary["bhk_count"],
                        "resolved_action": "merged_into_golden_record"
                    })

        unique_df = pd.DataFrame(golden_records)
        audit_df = pd.DataFrame(audit_trail)

        return unique_df, audit_df
