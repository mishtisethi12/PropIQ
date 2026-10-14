import os
import pytest
import duckdb
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUCKDB_PATH = os.path.join(BASE_DIR, "data/warehouse/propiq_warehouse.duckdb")

def test_duckdb_star_schema_tables():
    assert os.path.exists(DUCKDB_PATH), "DuckDB warehouse file not found"
    con = duckdb.connect(database=DUCKDB_PATH, read_only=True)
    tables = con.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]
    con.close()

    expected_tables = ["dim_date", "dim_location", "dim_property_type", "dim_amenities", "fact_property_listings"]
    for t in expected_tables:
        assert t in table_names, f"Missing table: {t}"

def test_foreign_key_integrity():
    con = duckdb.connect(database=DUCKDB_PATH, read_only=True)
    
    # Check that every location_key in fact exists in dim_location
    orphan_locs = con.execute("""
        SELECT COUNT(*) FROM fact_property_listings f
        LEFT JOIN dim_location l ON f.location_key = l.location_key
        WHERE l.location_key IS NULL
    """).fetchone()[0]

    assert orphan_locs == 0, f"Found {orphan_locs} orphaned location keys in fact table"
    con.close()
