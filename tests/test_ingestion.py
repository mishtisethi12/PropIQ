import os
import pytest
import pandas as pd
from src.ingestion.csv_ingestion import CSVIngestionEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_portal_alpha_ingestion():
    engine = CSVIngestionEngine()
    alpha_path = os.path.join(BASE_DIR, "data/raw/csv_sources/portal_alpha_listings.csv")
    assert os.path.exists(alpha_path), "Portal Alpha CSV file missing"
    df = engine.ingest_portal_alpha(alpha_path)
    assert len(df) > 0
    assert "listing_id" in df.columns
    assert "price_inr" in df.columns
    assert "source_portal" in df.columns
    assert df["source_portal"].iloc[0] == "Portal_Alpha"

def test_portal_beta_ingestion_parsing():
    engine = CSVIngestionEngine()
    beta_path = os.path.join(BASE_DIR, "data/raw/csv_sources/portal_beta_listings.csv")
    assert os.path.exists(beta_path), "Portal Beta CSV file missing"
    df = engine.ingest_portal_beta(beta_path)
    assert len(df) > 0
    assert "price_inr" in df.columns
    assert "bhk_count" in df.columns
    assert "carpet_area_sqft" in df.columns
    assert df["bhk_count"].dtype in [int, "int64", "int32"]
    assert df["price_inr"].iloc[0] > 100000
