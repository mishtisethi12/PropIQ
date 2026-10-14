"""
PropIQ - Direct PostgreSQL Row Count Verification Script
Queries PostgreSQL directly via SQLAlchemy and prints verified table row counts.
"""

import yaml
from sqlalchemy import create_engine, text

def verify_postgres():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    postgres_url = cfg["storage"]["postgres_url"]
    if postgres_url.startswith("postgresql://"):
        postgres_url = postgres_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    print(f"Connecting to PostgreSQL: {postgres_url.split('@')[-1]}")
    engine = create_engine(postgres_url)

    tables = ["raw_property_listings", "raw_locality_infrastructure", "raw_scraped_documents"]
    
    print("\n---------------------------------------------------------")
    print("POSTGRESQL OLTP RAW LAYER VERIFICATION (SQLAlchemy Query)")
    print("---------------------------------------------------------")
    
    with engine.connect() as conn:
        for t in tables:
            row_count = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
            print(f"  TABLE: {t.ljust(30)} -> {row_count:>6} ROWS VERIFIED")

            # Preview sample data
            sample = conn.execute(text(f"SELECT * FROM {t} LIMIT 1")).mappings().fetchone()
            sample_keys = list(sample.keys())[:5] if sample else []
            print(f"    Sample Columns: {', '.join(sample_keys)}...")

    print("---------------------------------------------------------")
    print("ALL POSTGRESQL TABLES VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    verify_postgres()
