"""
PropIQ - Ingestion Pipeline Runner
Orchestrates multi-source ingestion:
1. Portal Alpha & Beta CSVs
2. Locality Metadata Public API
3. Scraped Unstructured Texts & Lease Documents
Persists raw tables to OLTP storage (SQLite/PostgreSQL) and staging parquet.
"""

import os
import yaml
import sqlite3
import pandas as pd
from src.ingestion.csv_ingestion import CSVIngestionEngine
from src.ingestion.api_ingestion import APIIngestionEngine
from src.ingestion.scraper_ingestion import ScraperIngestionEngine

def run_ingestion(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    alpha_csv = os.path.join(base_dir, cfg["ingestion"]["csv_sources"][0]["path"])
    beta_csv = os.path.join(base_dir, cfg["ingestion"]["csv_sources"][1]["path"])
    api_endpoint = os.path.join(base_dir, cfg["ingestion"]["api_config"]["endpoint_mock_file"])
    scraped_dir = os.path.join(base_dir, cfg["ingestion"]["scraper_config"]["unstructured_html_dir"])
    oltp_db_path = os.path.join(base_dir, cfg["storage"]["sqlite_raw_path"])
    staging_dir = os.path.join(base_dir, cfg["storage"]["staging_dir"])
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(os.path.dirname(oltp_db_path), exist_ok=True)

    print("\n========================================================")
    print("STEP 1: INGESTING RAW DATA FROM MULTIPLE SOURCES")
    print("========================================================")

    # 1. Ingest CSVs
    csv_engine = CSVIngestionEngine()
    df_listings_raw = csv_engine.load_all_sources(alpha_csv, beta_csv)
    print(f"[OK] Ingested {len(df_listings_raw)} total raw listings from Portal Alpha & Portal Beta.")

    # 2. Ingest API Data
    api_engine = APIIngestionEngine(api_endpoint)
    locality_records = api_engine.fetch_locality_data()
    df_locality_raw = pd.DataFrame(locality_records)
    print(f"[OK] Ingested {len(df_locality_raw)} locality metadata records from Public API.")

    # 3. Ingest Scraped Documents
    scraper_engine = ScraperIngestionEngine(scraped_dir)
    scraped_docs = scraper_engine.load_scraped_documents()
    df_docs_raw = pd.DataFrame(scraped_docs)
    print(f"[OK] Ingested {len(df_docs_raw)} unstructured documents/lease contracts.")

    # 4. Save to OLTP Database (Simulating PostgreSQL Raw Tables)
    conn = sqlite3.connect(oltp_db_path)
    df_listings_raw.to_sql("raw_property_listings", conn, if_exists="replace", index=False)
    df_locality_raw.to_sql("raw_locality_infrastructure", conn, if_exists="replace", index=False)
    df_docs_raw.to_sql("raw_scraped_documents", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[OK] Stored raw OLTP tables into {oltp_db_path}")

    # 5. Save staging copy as parquet
    listings_staging_path = os.path.join(staging_dir, "raw_listings.parquet")
    df_listings_raw.to_parquet(listings_staging_path, index=False)
    locality_staging_path = os.path.join(staging_dir, "raw_locality.parquet")
    df_locality_raw.to_parquet(locality_staging_path, index=False)

    return {
        "raw_listings_count": len(df_listings_raw),
        "raw_locality_count": len(df_locality_raw),
        "raw_docs_count": len(df_docs_raw),
        "oltp_db_path": oltp_db_path,
        "staging_path": listings_staging_path
    }

if __name__ == "__main__":
    res = run_ingestion()
    print("Ingestion output:", res)
