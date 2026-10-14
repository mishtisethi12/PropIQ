"""
PropIQ - Locality Infrastructure & Amenities Public API Ingestion
Handles polling and ingesting urban infrastructure scores (transit, safety, schools, metro proximity)
"""

import json
import os
import time
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class LocalityMetadata(BaseModel):
    locality_id: str
    city: str
    locality_name: str
    latitude: float
    longitude: float
    transit_score: int = Field(ge=0, le=100)
    safety_score: int = Field(ge=0, le=100)
    school_score: int = Field(ge=0, le=100)
    hospital_count: int = Field(ge=0)
    metro_distance_km: float
    commercial_hub_proximity_km: float
    green_cover_pct: int
    last_audited: str

class APIIngestionEngine:
    def __init__(self, endpoint_file: str):
        self.endpoint_file = endpoint_file

    def fetch_locality_data(self) -> List[Dict[str, Any]]:
        """
        Simulates calling a REST API endpoint with retry logic and Pydantic validation.
        In production, this queries OpenStreetMap / Municipal GIS endpoints.
        """
        if not os.path.exists(self.endpoint_file):
            raise FileNotFoundError(f"API source file {self.endpoint_file} not found.")

        # Simulate network latency
        time.sleep(0.05)

        with open(self.endpoint_file, "r", encoding="utf-8") as f:
            raw_payload = json.load(f)

        validated_records = []
        for item in raw_payload:
            validated = LocalityMetadata(**item)
            validated_records.append(validated.model_dump())

        return validated_records
