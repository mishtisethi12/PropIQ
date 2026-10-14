"""
PropIQ - Table Partitioning & Partition Pruning Benchmark Engine
Implements Hive-style physical partitioning by `city` and by `month`.
Benchmarks query performance showing speedup from partition pruning.
"""

import os
import time
import duckdb
import pandas as pd

class PartitioningEngine:
    def __init__(self, partition_base_dir: str):
        self.partition_base_dir = partition_base_dir

    def create_partitions(self, fact_df: pd.DataFrame):
        """
        Writes partitioned Parquet datasets:
        1. By City (e.g. by_city/city=Noida/...)
        2. By Month (e.g. by_month/year_month=2026-01/...)
        """
        by_city_dir = os.path.join(self.partition_base_dir, "by_city")
        by_month_dir = os.path.join(self.partition_base_dir, "by_month")
        os.makedirs(by_city_dir, exist_ok=True)
        os.makedirs(by_month_dir, exist_ok=True)

        # Partition by City
        for city, group in fact_df.groupby("city"):
            city_folder = os.path.join(by_city_dir, f"city={city}")
            os.makedirs(city_folder, exist_ok=True)
            group.to_parquet(os.path.join(city_folder, "data.parquet"), index=False)

        # Partition by Month
        for ym, group in fact_df.groupby("year_month"):
            month_folder = os.path.join(by_month_dir, f"year_month={ym}")
            os.makedirs(month_folder, exist_ok=True)
            group.to_parquet(os.path.join(month_folder, "data.parquet"), index=False)

    def benchmark_partition_pruning(self, unpartitioned_parquet_path: str, target_city: str = "Noida") -> dict:
        """
        Runs an analytical aggregate query against:
        A) Full unpartitioned Parquet file (Full table scan)
        B) Partitioned directory where DuckDB executes partition pruning
        """
        con = duckdb.connect(database=":memory:")

        # 1. Benchmark Unpartitioned Scan
        t0 = time.perf_counter()
        query_unpartitioned = f"""
            SELECT 
                city,
                COUNT(*) as listing_count,
                ROUND(AVG(price_inr), 2) as avg_price,
                ROUND(AVG(rent_inr), 2) as avg_rent,
                ROUND(AVG(rental_yield_pct), 2) as avg_yield
            FROM read_parquet('{unpartitioned_parquet_path}')
            WHERE city = '{target_city}'
            GROUP BY city
        """
        res_unpart = con.execute(query_unpartitioned).fetchall()
        t_unpart = time.perf_counter() - t0

        # 2. Benchmark Partition Pruned Scan
        by_city_path = os.path.join(self.partition_base_dir, "by_city", f"city={target_city}", "data.parquet").replace("\\", "/")
        t1 = time.perf_counter()
        query_partitioned = f"""
            SELECT 
                '{target_city}' as city,
                COUNT(*) as listing_count,
                ROUND(AVG(price_inr), 2) as avg_price,
                ROUND(AVG(rent_inr), 2) as avg_rent,
                ROUND(AVG(rental_yield_pct), 2) as avg_yield
            FROM read_parquet('{by_city_path}')
        """
        res_part = con.execute(query_partitioned).fetchall()
        t_part = time.perf_counter() - t1

        con.close()

        speedup_factor = round(t_unpart / max(t_part, 1e-6), 2)

        return {
            "target_city": target_city,
            "unpartitioned_scan_time_sec": round(t_unpart, 5),
            "partition_pruned_scan_time_sec": round(t_part, 5),
            "speedup_factor": f"{speedup_factor}x faster",
            "results": res_part
        }
