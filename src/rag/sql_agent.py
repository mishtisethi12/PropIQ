"""
PropIQ - Text-to-SQL Analytics Agent
Translates natural language questions into Star Schema SQL queries on DuckDB.
"""

import re
import duckdb
import pandas as pd
from typing import Optional, Dict, Any, Tuple

class SQLAnalyticsAgent:
    def __init__(self, duckdb_path: str):
        self.duckdb_path = duckdb_path

    def parse_intent_and_generate_sql(self, user_query: str) -> Tuple[Optional[str], str]:
        """
        Parses intent for real estate queries:
        e.g., 'Which 2BHK areas in Noida have the best rental yield?'
        """
        query_lower = user_query.lower()

        # Extract City
        city = None
        for c in ["noida", "greater noida", "gurgaon", "delhi", "bangalore"]:
            if c in query_lower:
                city = c.title() if c != "greater noida" else "Greater Noida"
                break

        # Extract BHK
        bhk_match = re.search(r"(\d+)\s*bhk", query_lower)
        bhk = int(bhk_match.group(1)) if bhk_match else None

        # Check if rental yield query
        if "rental yield" in query_lower or "best yield" in query_lower or "highest rent" in query_lower:
            where_clauses = []
            if city:
                where_clauses.append(f"f.city = '{city}'")
            if bhk:
                where_clauses.append(f"p.bhk_count = {bhk}")

            where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            sql = f"""
                SELECT 
                    f.city,
                    l.locality,
                    p.bhk_label,
                    COUNT(*) as total_properties,
                    ROUND(AVG(f.price_inr), 0) as avg_price,
                    ROUND(AVG(f.rent_inr), 0) as avg_rent,
                    ROUND(AVG(f.price_per_sqft), 0) as avg_price_sqft,
                    ROUND(AVG(f.rental_yield_pct), 2) as avg_rental_yield_pct,
                    ROUND(AVG(l.transit_score), 1) as transit_score,
                    ROUND(AVG(l.safety_score), 1) as safety_score
                FROM fact_property_listings f
                JOIN dim_location l ON f.location_key = l.location_key
                JOIN dim_property_type p ON f.property_type_key = p.property_type_key
                {where_str}
                GROUP BY f.city, l.locality, p.bhk_label
                ORDER BY avg_rental_yield_pct DESC
                LIMIT 10;
            """
            return sql, "rental_yield_ranking"

        # Check if cheapest / most affordable price query
        elif "cheapest" in query_lower or "affordable" in query_lower or "lowest price" in query_lower:
            where_clauses = []
            if city:
                where_clauses.append(f"f.city = '{city}'")
            if bhk:
                where_clauses.append(f"p.bhk_count = {bhk}")

            where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            sql = f"""
                SELECT 
                    f.city,
                    l.locality,
                    p.bhk_label,
                    COUNT(*) as total_properties,
                    ROUND(AVG(f.price_inr), 0) as avg_price,
                    ROUND(AVG(f.price_per_sqft), 0) as avg_price_sqft,
                    ROUND(AVG(f.rental_yield_pct), 2) as avg_rental_yield_pct
                FROM fact_property_listings f
                JOIN dim_location l ON f.location_key = l.location_key
                JOIN dim_property_type p ON f.property_type_key = p.property_type_key
                {where_str}
                GROUP BY f.city, l.locality, p.bhk_label
                ORDER BY avg_price ASC
                LIMIT 10;
            """
            return sql, "price_ranking"

        # General aggregate query
        elif city or bhk:
            where_clauses = []
            if city:
                where_clauses.append(f"f.city = '{city}'")
            if bhk:
                where_clauses.append(f"p.bhk_count = {bhk}")

            where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            sql = f"""
                SELECT 
                    f.city,
                    l.locality,
                    p.bhk_label,
                    ROUND(AVG(f.price_inr), 0) as avg_price,
                    ROUND(AVG(f.rent_inr), 0) as avg_rent,
                    ROUND(AVG(f.rental_yield_pct), 2) as avg_rental_yield_pct
                FROM fact_property_listings f
                JOIN dim_location l ON f.location_key = l.location_key
                JOIN dim_property_type p ON f.property_type_key = p.property_type_key
                {where_str}
                GROUP BY f.city, l.locality, p.bhk_label
                ORDER BY avg_rental_yield_pct DESC
                LIMIT 10;
            """
            return sql, "market_overview"

        return None, "unstructured_search"

    def execute_sql(self, sql_query: str, backend: str = "duckdb") -> pd.DataFrame:
        """Executes generated SQL against DuckDB or Snowflake warehouse"""
        if backend.lower() == "snowflake":
            try:
                from src.warehouse.snowflake_loader import SnowflakeWarehouseManager
                manager = SnowflakeWarehouseManager()
                if manager.is_configured():
                    df = manager.query(sql_query)
                    df.columns = [c.lower() for c in df.columns]
                    return df
            except Exception as e:
                print(f"Warning: Snowflake query failed ({e}), falling back to DuckDB.")

        con = duckdb.connect(database=self.duckdb_path, read_only=True)
        try:
            df = con.execute(sql_query).fetchdf()
        finally:
            con.close()
        return df

