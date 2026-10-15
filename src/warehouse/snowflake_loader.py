"""
PropIQ - Snowflake Cloud Data Warehouse Loader & Query Service
Provisions Star Schema on Snowflake, uploads dimensional marts, and serves live queries.
"""

import os
import sys
import yaml
import pandas as pd
from typing import Dict, Any, Optional

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

try:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    snowflake = None
    write_pandas = None
    SNOWFLAKE_AVAILABLE = False


class SnowflakeWarehouseManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        sf_cfg = cfg.get("snowflake", {})
        self.account = os.environ.get("SNOWFLAKE_ACCOUNT", sf_cfg.get("account", ""))
        self.user = os.environ.get("SNOWFLAKE_USER", sf_cfg.get("user", ""))
        self.password = os.environ.get("SNOWFLAKE_PASSWORD", sf_cfg.get("password", ""))
        self.role = os.environ.get("SNOWFLAKE_ROLE", sf_cfg.get("role", "ACCOUNTADMIN"))
        self.warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE", sf_cfg.get("warehouse", "PROPIQ_WH"))
        self.database = os.environ.get("SNOWFLAKE_DATABASE", sf_cfg.get("database", "PROPIQ_DW"))
        self.schema = os.environ.get("SNOWFLAKE_SCHEMA", sf_cfg.get("schema", "CORE_STAR_SCHEMA"))
        self.warehouse_dir = os.path.join(base_dir, cfg["storage"]["warehouse_dir"])

    def is_configured(self) -> bool:
        """Returns True if snowflake connector is installed and credentials are set"""
        return bool(SNOWFLAKE_AVAILABLE and self.account and self.user and self.password)

    def get_connection(self, target_db: bool = True):
        """Creates an authenticated Snowflake connection"""
        if not (self.account and self.user and self.password):
            raise ValueError(
                "Snowflake credentials missing. Please set account, user, and password in "
                "config/config.yaml or via SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD env vars."
            )
        if not SNOWFLAKE_AVAILABLE:
            raise ImportError(
                "snowflake-connector-python is not installed. Please install it to use Snowflake."
            )

        kwargs = {
            "account": self.account,
            "user": self.user,
            "password": self.password,
            "role": self.role
        }
        if target_db:
            kwargs["warehouse"] = self.warehouse
            kwargs["database"] = self.database
            kwargs["schema"] = self.schema

        return snowflake.connector.connect(**kwargs)

    def provision_environment(self):
        """Creates Virtual Warehouse, Database, and Schema if not exists"""
        print(f"Connecting to Snowflake account: {self.account}...")
        conn = self.get_connection(target_db=False)
        cs = conn.cursor()
        try:
            print(f"[1/4] Provisioning Snowflake Warehouse '{self.warehouse}'...")
            cs.execute(f"""
                CREATE WAREHOUSE IF NOT EXISTS {self.warehouse}
                WITH WAREHOUSE_SIZE = 'XSMALL'
                AUTO_SUSPEND = 60
                AUTO_RESUME = TRUE
                INITIALLY_SUSPENDED = TRUE
                COMMENT = 'PropIQ Real Estate Analytics Warehouse';
            """)

            print(f"[2/4] Provisioning Database '{self.database}' & Schema '{self.schema}'...")
            cs.execute(f"CREATE DATABASE IF NOT EXISTS {self.database};")
            cs.execute(f"CREATE SCHEMA IF NOT EXISTS {self.database}.{self.schema};")
            cs.execute(f"USE WAREHOUSE {self.warehouse};")
            cs.execute(f"USE DATABASE {self.database};")
            cs.execute(f"USE SCHEMA {self.schema};")
            print("[OK] Snowflake warehouse environment provisioned.")
        finally:
            cs.close()
            conn.close()

    def execute_ddl(self, ddl_path: str = "src/warehouse/ddl_snowflake.sql"):
        """Executes table DDLs in Snowflake"""
        print(f"[3/4] Creating Star Schema tables in {self.database}.{self.schema}...")
        conn = self.get_connection(target_db=True)
        cs = conn.cursor()
        try:
            full_ddl_path = os.path.join(base_dir, ddl_path)
            with open(full_ddl_path, "r", encoding="utf-8") as f:
                ddl_statements = f.read().split(";")

            for stmt in ddl_statements:
                clean_stmt = stmt.strip()
                if clean_stmt and not clean_stmt.startswith("--"):
                    cs.execute(clean_stmt)
            print("[OK] Star schema DDL executed successfully.")
        finally:
            cs.close()
            conn.close()

    def load_star_schema(self) -> Dict[str, int]:
        """Loads local Parquet datasets into Snowflake tables using write_pandas"""
        print(f"[4/4] Bulk uploading Star Schema tables to Snowflake...")
        conn = self.get_connection(target_db=True)

        table_mappings = [
            ("dim_date.parquet", "DIM_DATE"),
            ("dim_location.parquet", "DIM_LOCATION"),
            ("dim_property_type.parquet", "DIM_PROPERTY_TYPE"),
            ("dim_amenities.parquet", "DIM_AMENITIES"),
            ("fact_property_listings.parquet", "FACT_PROPERTY_LISTINGS")
        ]

        loaded_counts = {}

        try:
            for parquet_file, table_name in table_mappings:
                fpath = os.path.join(self.warehouse_dir, parquet_file)
                if not os.path.exists(fpath):
                    print(f"  [WARN] Missing file {fpath}, skipping.")
                    continue

                df = pd.read_parquet(fpath)
                # Snowflake column names are uppercase
                df.columns = [c.upper() for c in df.columns]

                # Bulk load using write_pandas
                success, nchunks, nrows, _ = write_pandas(
                    conn,
                    df,
                    table_name=table_name,
                    database=self.database,
                    schema=self.schema,
                    overwrite=True,
                    auto_create_table=False
                )
                loaded_counts[table_name] = nrows
                print(f"  - Loaded {table_name}: {nrows} rows (success={success})")

            return loaded_counts
        finally:
            conn.close()

    def verify_counts_and_query(self) -> pd.DataFrame:
        """Executes live verification query against Snowflake"""
        conn = self.get_connection(target_db=True)
        cs = conn.cursor()
        try:
            print("\n---------------------------------------------------------")
            print("SNOWFLAKE CLOUD WAREHOUSE VERIFICATION")
            print("---------------------------------------------------------")
            for tbl in ["FACT_PROPERTY_LISTINGS", "DIM_LOCATION", "DIM_PROPERTY_TYPE", "DIM_AMENITIES", "DIM_DATE"]:
                cs.execute(f"SELECT COUNT(*) FROM {self.database}.{self.schema}.{tbl}")
                cnt = cs.fetchone()[0]
                print(f"  TABLE: {tbl.ljust(25)} -> {cnt:>6} ROWS VERIFIED")

            # Sample Analytical Query
            query = f"""
                SELECT 
                    f.CITY,
                    l.LOCALITY,
                    p.BHK_LABEL,
                    COUNT(*) as TOTAL_PROPERTIES,
                    ROUND(AVG(f.PRICE_INR), 0) as AVG_PRICE,
                    ROUND(AVG(f.RENT_INR), 0) as AVG_RENT,
                    ROUND(AVG(f.RENTAL_YIELD_PCT), 2) as AVG_YIELD_PCT
                FROM {self.database}.{self.schema}.FACT_PROPERTY_LISTINGS f
                JOIN {self.database}.{self.schema}.DIM_LOCATION l ON f.LOCATION_KEY = l.LOCATION_KEY
                JOIN {self.database}.{self.schema}.DIM_PROPERTY_TYPE p ON f.PROPERTY_TYPE_KEY = p.PROPERTY_TYPE_KEY
                WHERE f.CITY = 'Noida' AND p.BHK_COUNT = 2
                GROUP BY f.CITY, l.LOCALITY, p.BHK_LABEL
                ORDER BY AVG_YIELD_PCT DESC
                LIMIT 5;
            """
            cs.execute(query)
            cols = [col[0] for col in cs.description]
            rows = cs.fetchall()
            df_result = pd.DataFrame(rows, columns=cols)
            print("\nTop 2BHK Yields in Noida (Computed directly in Snowflake):")
            print(df_result.to_string(index=False))
            print("---------------------------------------------------------")
            return df_result
        finally:
            cs.close()
            conn.close()

    def query(self, sql_query: str) -> pd.DataFrame:
        """Direct query executor returning pandas DataFrame"""
        conn = self.get_connection(target_db=True)
        try:
            return pd.read_sql(sql_query, conn)
        finally:
            conn.close()

def main():
    manager = SnowflakeWarehouseManager()
    if not manager.is_configured():
        print("""
========================================================================
PROPIQ - SNOWFLAKE CLOUD DATA WAREHOUSE SETUP
========================================================================
Snowflake credentials are not yet configured in config/config.yaml.

To connect your Snowflake trial account:
1. Open config/config.yaml and fill in the snowflake section:
   snowflake:
     account: "YOUR_ACCOUNT_IDENTIFIER"  # e.g. "xy12345.ap-south-1" or "ORG-ACCOUNT"
     user: "YOUR_USERNAME"
     password: "YOUR_PASSWORD"
     role: "ACCOUNTADMIN"
     warehouse: "PROPIQ_WH"
     database: "PROPIQ_DW"
     schema: "CORE_STAR_SCHEMA"

2. Or export environment variables in PowerShell:
   $env:SNOWFLAKE_ACCOUNT="your_account"
   $env:SNOWFLAKE_USER="your_username"
   $env:SNOWFLAKE_PASSWORD="your_password"

3. Re-run: python src/warehouse/snowflake_loader.py
========================================================================
        """)
        return

    manager.provision_environment()
    manager.execute_ddl()
    manager.load_star_schema()
    manager.verify_counts_and_query()

if __name__ == "__main__":
    main()
