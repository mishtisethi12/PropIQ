"""
PropIQ - Star Schema ETL Pipeline
Transforms cleaned staging listings into dimensional Kimball Star Schema.
Loads into DuckDB OLAP warehouse, Parquet files, Power BI export files, and generates partitions.
"""

import os
import yaml
import duckdb
import pandas as pd
from src.warehouse.schema import StarSchemaBuilder
from src.warehouse.partitioning import PartitioningEngine

def run_star_schema_etl(cleaned_listings_df: pd.DataFrame, api_locality_df: pd.DataFrame, config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    warehouse_dir = os.path.join(base_dir, cfg["storage"]["warehouse_dir"])
    duckdb_path = os.path.join(base_dir, cfg["storage"]["duckdb_path"])
    powerbi_dir = os.path.join(warehouse_dir, "export_powerbi")
    partitions_dir = os.path.join(warehouse_dir, "partitions")

    os.makedirs(warehouse_dir, exist_ok=True)
    os.makedirs(powerbi_dir, exist_ok=True)
    os.makedirs(partitions_dir, exist_ok=True)

    print("\n========================================================")
    print("STEP 3: STAR SCHEMA ETL & OLAP WAREHOUSE LOADING")
    print("========================================================")

    builder = StarSchemaBuilder()

    # 1. Build Dimensions
    dim_date = builder.build_dim_date()
    dim_location = builder.build_dim_location(cleaned_listings_df, api_locality_df)
    dim_property_type = builder.build_dim_property_type(cleaned_listings_df)
    dim_amenities = builder.build_dim_amenities(cleaned_listings_df)

    print(f"[OK] Generated dim_date: {len(dim_date)} rows")
    print(f"[OK] Generated dim_location: {len(dim_location)} rows")
    print(f"[OK] Generated dim_property_type: {len(dim_property_type)} rows")
    print(f"[OK] Generated dim_amenities: {len(dim_amenities)} rows")

    # 2. Build Fact Table
    fact_listings = builder.build_fact_listings(
        cleaned_listings_df,
        dim_location,
        dim_property_type,
        dim_amenities
    )
    print(f"[OK] Generated fact_property_listings: {len(fact_listings)} rows")

    # 3. Load into DuckDB Warehouse
    con = duckdb.connect(database=duckdb_path)
    con.execute("CREATE OR REPLACE TABLE dim_date AS SELECT * FROM dim_date")
    con.execute("CREATE OR REPLACE TABLE dim_location AS SELECT * FROM dim_location")
    con.execute("CREATE OR REPLACE TABLE dim_property_type AS SELECT * FROM dim_property_type")
    con.execute("CREATE OR REPLACE TABLE dim_amenities AS SELECT * FROM dim_amenities")
    con.execute("CREATE OR REPLACE TABLE fact_property_listings AS SELECT * FROM fact_listings")
    
    # Create analytical view for fast BI queries
    con.execute("""
        CREATE OR REPLACE VIEW view_real_estate_market_summary AS
        SELECT 
            f.city,
            l.locality,
            p.bhk_label,
            p.furnishing_status,
            a.amenity_tier,
            COUNT(*) as total_properties,
            ROUND(AVG(f.price_inr), 2) as avg_price_inr,
            ROUND(AVG(f.rent_inr), 2) as avg_rent_inr,
            ROUND(AVG(f.carpet_area_sqft), 2) as avg_carpet_sqft,
            ROUND(AVG(f.price_per_sqft), 2) as avg_price_per_sqft,
            ROUND(AVG(f.rental_yield_pct), 2) as avg_rental_yield_pct,
            ROUND(AVG(l.transit_score), 1) as avg_transit_score,
            ROUND(AVG(l.safety_score), 1) as avg_safety_score
        FROM fact_property_listings f
        JOIN dim_location l ON f.location_key = l.location_key
        JOIN dim_property_type p ON f.property_type_key = p.property_type_key
        JOIN dim_amenities a ON f.amenity_key = a.amenity_key
        GROUP BY f.city, l.locality, p.bhk_label, p.furnishing_status, a.amenity_tier
    """)
    con.close()
    print(f"[OK] Successfully loaded Star Schema into DuckDB: {duckdb_path}")

    # 4. Export Parquet & Power BI/Tableau CSV files
    fact_parquet_path = os.path.join(warehouse_dir, "fact_property_listings.parquet")
    fact_listings.to_parquet(fact_parquet_path, index=False)
    dim_location.to_parquet(os.path.join(warehouse_dir, "dim_location.parquet"), index=False)
    dim_property_type.to_parquet(os.path.join(warehouse_dir, "dim_property_type.parquet"), index=False)
    dim_amenities.to_parquet(os.path.join(warehouse_dir, "dim_amenities.parquet"), index=False)
    dim_date.to_parquet(os.path.join(warehouse_dir, "dim_date.parquet"), index=False)

    # CSV exports for Power BI / Tableau
    fact_listings.to_csv(os.path.join(powerbi_dir, "fact_property_listings.csv"), index=False)
    dim_location.to_csv(os.path.join(powerbi_dir, "dim_location.csv"), index=False)
    dim_property_type.to_csv(os.path.join(powerbi_dir, "dim_property_type.csv"), index=False)
    dim_amenities.to_csv(os.path.join(powerbi_dir, "dim_amenities.csv"), index=False)
    dim_date.to_csv(os.path.join(powerbi_dir, "dim_date.csv"), index=False)
    print(f"[OK] Exported Parquet & CSV files for Power BI/Tableau to {powerbi_dir}")

    # 5. Partitioning & Benchmark
    part_engine = PartitioningEngine(partitions_dir)
    part_engine.create_partitions(fact_listings)
    print(f"[OK] Created physical partitions in {partitions_dir}")

    benchmark_res = part_engine.benchmark_partition_pruning(fact_parquet_path, target_city="Noida")
    print(f"[OK] Partition Pruning Benchmark: {benchmark_res['speedup_factor']} on target '{benchmark_res['target_city']}'")

    return {
        "duckdb_path": duckdb_path,
        "fact_rows": len(fact_listings),
        "benchmark": benchmark_res
    }
