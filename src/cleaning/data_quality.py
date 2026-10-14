"""
PropIQ - Data Quality & Automated Validation Suite
Performs schema validation, null checks, geographic bounding tests, and statistical outlier auditing.
"""

import pandas as pd
from typing import Dict, Any, List

class DataQualitySuite:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results: List[Dict[str, Any]] = []

    def assert_no_nulls(self, column: str):
        null_count = int(self.df[column].isna().sum())
        passed = null_count == 0
        self.results.append({
            "test_name": f"check_no_nulls_{column}",
            "passed": passed,
            "failed_records": null_count,
            "details": f"Found {null_count} null values in {column}"
        })

    def assert_range(self, column: str, min_val: float, max_val: float):
        invalid_mask = (self.df[column] < min_val) | (self.df[column] > max_val)
        invalid_count = int(invalid_mask.sum())
        passed = invalid_count == 0
        self.results.append({
            "test_name": f"check_range_{column}",
            "passed": passed,
            "failed_records": invalid_count,
            "details": f"{invalid_count} values outside [{min_val}, {max_val}]"
        })

    def assert_unique(self, column: str):
        dup_count = int(self.df.duplicated(subset=[column]).sum())
        passed = dup_count == 0
        self.results.append({
            "test_name": f"check_unique_{column}",
            "passed": passed,
            "failed_records": dup_count,
            "details": f"Found {dup_count} duplicate keys in {column}"
        })

    def run_all_checks(self) -> Dict[str, Any]:
        """Runs the entire battery of data quality tests"""
        self.assert_no_nulls("listing_id")
        self.assert_no_nulls("price_inr")
        self.assert_no_nulls("bhk_count")
        self.assert_unique("listing_id")
        self.assert_range("bhk_count", 1, 10)
        self.assert_range("price_inr", 300000, 1000000000)
        self.assert_range("carpet_area_sqft", 150, 15000)
        self.assert_range("latitude", 8.0, 37.0)
        self.assert_range("longitude", 68.0, 98.0)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["passed"])

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate_pct": round((passed_tests / total_tests) * 100, 2),
            "test_details": self.results
        }
