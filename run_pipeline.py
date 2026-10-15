"""
PropIQ - Master End-to-End Execution Pipeline
Runs the entire data engineering and AI lifecycle:
1. Ingestion from multiple sources (CSV Alpha, CSV Beta, Public API, Scraped text)
2. Normalization & Deduplication (resolves same property listed twice)
3. Data Quality validation tests
4. Star schema ETL into DuckDB OLAP warehouse
5. Physical table partitioning & pruning benchmark
6. Machine Learning model training & comparison (Decision Forest vs k-NN vs XGBoost)
7. Hybrid GenAI Copilot query test
"""

import os
import sys
import yaml
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ingestion.pipeline_runner import run_ingestion
from src.cleaning.normalizer import DataNormalizer
from src.cleaning.deduplication import RealEstateDeduplicator
from src.cleaning.data_quality import DataQualitySuite
from src.warehouse.etl_star_schema import run_star_schema_etl
from src.ml.train import train_and_compare_models
from src.rag.assistant import PropIQAssistant

def main():
    print(r"""
    ========================================================================
       ____                   ___ ___  
      / __ \_________  ____  /   |__ \ 
     / /_/ / ___/ __ \/ __ \/ /| | / / 
    / ____/ /  / /_/ / /_/ / ___ |/ /_ 
   /_/   /_/   \____/ .___/_/  |_/____/
                   /_/                 
    End-to-End Real Estate Data Platform with GenAI Assistant
    ========================================================================
    """)

    # 1. Multi-source Ingestion
    ingest_res = run_ingestion()
    raw_df = pd.read_parquet(ingest_res["staging_path"])
    raw_locality = pd.read_parquet(os.path.join(BASE_DIR, "data/staging/raw_locality.parquet"))

    # 2. Cleaning & Normalization
    print("\n========================================================")
    print("STEP 2: DATA CLEANING, NORMALIZATION & DE-DUPLICATION")
    print("========================================================")
    normalizer = DataNormalizer()
    normalized_df = normalizer.clean_listings(raw_df)
    print(f"[OK] Normalized {len(normalized_df)} listings.")

    # 3. Intelligent Deduplication
    dedup = RealEstateDeduplicator(geo_proximity_meters=75.0, price_variance_tol=0.05, area_variance_tol=0.05)
    unique_df, audit_df = dedup.deduplicate(normalized_df)
    print(f"[OK] De-duplication complete: {len(normalized_df)} raw -> {len(unique_df)} golden unique records.")
    print(f"[OK] Identified and merged {len(audit_df)} duplicate listing pairs.")

    # Save deduplication audit log
    audit_path = os.path.join(BASE_DIR, "data/staging/deduplication_audit.csv")
    audit_df.to_csv(audit_path, index=False)
    print(f"[OK] Saved de-duplication audit trail to {audit_path}")

    # 4. Data Quality Testing (Great Expectations Assertion Battery)
    print("\n========================================================")
    print("STEP 3: RUNNING GREAT EXPECTATIONS DATA QUALITY SUITE")
    print("========================================================")
    dq = DataQualitySuite(unique_df)
    dq_results = dq.run_all_checks(halt_on_failure=True)
    print(f"[OK] Great Expectations Suite Passed: {dq_results['passed_tests']}/{dq_results['total_tests']} tests ({dq_results['pass_rate_pct']}%)")

    # 5. Star Schema ETL & DuckDB Warehouse Loading
    warehouse_res = run_star_schema_etl(unique_df, raw_locality)

    # 6. ML Model Training & Comparison
    fact_parquet = os.path.join(BASE_DIR, "data/warehouse/fact_property_listings.parquet")
    fact_df = pd.read_parquet(fact_parquet)
    
    # Join dimensional attributes for ML features
    dim_loc = pd.read_parquet(os.path.join(BASE_DIR, "data/warehouse/dim_location.parquet"))
    dim_prop = pd.read_parquet(os.path.join(BASE_DIR, "data/warehouse/dim_property_type.parquet"))
    
    ml_train_df = fact_df.merge(dim_loc[["location_key", "transit_score", "safety_score", "school_score"]], on="location_key", how="left")
    ml_train_df = ml_train_df.merge(dim_prop[["property_type_key", "bhk_count", "furnishing_status"]], on="property_type_key", how="left")
    ml_train_df["parking"] = 1
    ml_train_df["gym"] = 1
    ml_train_df["swimming_pool"] = 1

    ml_metrics = train_and_compare_models(ml_train_df)

    # 7. GenAI Assistant Test Query
    print("\n========================================================")
    print("STEP 5: GENAI ASSISTANT TEST QUERY")
    print("========================================================")
    test_query = "Which 2BHK areas in Noida have the best rental yield?"
    print(f"User Query: '{test_query}'")
    duckdb_path = warehouse_res["duckdb_path"]
    docs_dir = os.path.join(BASE_DIR, "data/raw/scraped_text")
    assistant = PropIQAssistant(duckdb_path, docs_dir)
    answer_pkg = assistant.answer_query(test_query)
    print("\nAssistant Output:")
    print(answer_pkg["answer"])

    print("\n========================================================")
    print("PROPIQ PIPELINE EXECUTION FINISHED SUCCESSFULLY!")
    print("Launch UI: streamlit run src/ui/app.py")
    print("========================================================")

if __name__ == "__main__":
    main()
