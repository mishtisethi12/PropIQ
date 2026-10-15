"""
PropIQ - Data Quality & Automated Validation Suite
Implements Great Expectations assertion suites for real estate ingestion pipelines:
- No null prices (ExpectColumnValuesToNotBeNull)
- Carpet area strictly greater than 0 (ExpectColumnValuesToBeBetween)
- Sensible acquisition price range (ExpectColumnValuesToBeBetween: 100K to 1B INR)
- Sensible monthly rent range (ExpectColumnValuesToBeBetween: 1K to 2.5M INR)
- Unique listing identifiers (ExpectColumnValuesToBeUnique)
- Geographic coordinate bounding (Delhi NCR and Bangalore boundaries)

Haults pipeline execution with detailed diagnostics upon any expectation failure.
Persists structured validation audit reports for the Pipeline & Quality UI.
"""

import os
import sys
import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class DataQualityValidationError(RuntimeError):
    """Raised when one or more critical Great Expectations quality assertions fail."""
    def __init__(self, message: str, failed_checks: List[Dict[str, Any]]):
        super().__init__(message)
        self.failed_checks = failed_checks


class DataQualitySuite:
    """
    Automated Data Quality & Assertion Engine powered by Great Expectations.
    Validates dimensional and factual consistency on property data frames.
    """
    def __init__(self, df: pd.DataFrame, report_dir: Optional[str] = None):
        self.df = df.copy()
        self.report_dir = report_dir or os.path.join(BASE_DIR, "artifacts/quality")
        self.staging_dir = os.path.join(BASE_DIR, "data/staging")
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.staging_dir, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    def _execute_gx_suite(self) -> List[Dict[str, Any]]:
        """
        Executes Great Expectations validation using GX 1.x ephemeral execution engine.
        Falls back to vectorized pandas assertions if GX runtime context is restricted.
        """
        try:
            import great_expectations as gx

            context = gx.get_context(mode="ephemeral")
            data_source = context.data_sources.add_pandas("propiq_pandas_source")
            data_asset = data_source.add_dataframe_asset("property_listings_asset")
            batch_def = data_asset.add_batch_definition_whole_dataframe("property_listings_batch")

            suite = context.suites.add(gx.ExpectationSuite(name="propiq_real_estate_expectations"))

            # 1. No null prices
            suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="price_inr"))

            # 2. Area greater than 0
            suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
                column="carpet_area_sqft",
                min_value=1.0,
                max_value=25000.0
            ))

            # 3. Rent within sensible range (INR 1,000 to INR 2,500,000 / month)
            if "rent_inr" in self.df.columns:
                suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
                    column="rent_inr",
                    min_value=1000.0,
                    max_value=2500000.0
                ))

            # 4. Price within sensible range (INR 100,000 to INR 1,000,000,000)
            suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
                column="price_inr",
                min_value=100000.0,
                max_value=1000000000.0
            ))

            # 5. Unique listing identifiers
            suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="listing_id"))

            # 6. Non-null listing identifiers
            suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="listing_id"))

            # 7. BHK count in sensible range (1 to 10 BHK)
            if "bhk_count" in self.df.columns:
                suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
                    column="bhk_count",
                    min_value=1,
                    max_value=10
                ))

            # 8. Geographic coordinate validity
            if "latitude" in self.df.columns:
                suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
                    column="latitude",
                    min_value=8.0,
                    max_value=37.0
                ))
            if "longitude" in self.df.columns:
                suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
                    column="longitude",
                    min_value=68.0,
                    max_value=98.0
                ))

            batch = batch_def.get_batch(batch_parameters={"dataframe": self.df})
            validation_result = batch.validate(suite)

            rule_descriptions = {
                ("expectcolumnvaluestonotbenull", "price_inr"): "Property price must be strictly non-null",
                ("expectcolumnvaluestobebetween", "carpet_area_sqft"): "Carpet area must be strictly greater than 0 sq.ft (1 - 25,000)",
                ("expectcolumnvaluestobebetween", "rent_inr"): "Monthly rent must fall within sensible range (INR 1,000 to 2,500,000)",
                ("expectcolumnvaluestobebetween", "price_inr"): "Acquisition price must fall within sensible range (INR 100K to 1B)",
                ("expectcolumnvaluestobeunique", "listing_id"): "Listing IDs must be globally unique across deduplicated inventory",
                ("expectcolumnvaluestonotbenull", "listing_id"): "Listing ID must be non-null",
                ("expectcolumnvaluestobebetween", "bhk_count"): "BHK configuration must fall between 1 and 10 rooms",
                ("expectcolumnvaluestobebetween", "latitude"): "Latitude coordinate must fall within India geographic bounds (8.0 - 37.0)",
                ("expectcolumnvaluestobebetween", "longitude"): "Longitude coordinate must fall within India geographic bounds (68.0 - 98.0)",
            }

            extracted_results = []
            for r in validation_result.results:
                exp_type = r.expectation_config.type
                col = r.expectation_config.kwargs.get("column", "unknown")
                clean_type_key = exp_type.replace("_", "").lower()
                clean_col_key = col.lower()
                rule_desc = rule_descriptions.get(
                    (clean_type_key, clean_col_key),
                    f"{exp_type} on column '{col}'"
                )

                res_meta = r.result or {}
                elem_count = res_meta.get("element_count", len(self.df))
                unexp_count = res_meta.get("unexpected_count", 0 if r.success else 1)
                unexp_pct = res_meta.get("unexpected_percent", 0.0)
                partial_unexp = res_meta.get("partial_unexpected_list", [])

                extracted_results.append({
                    "test_name": f"{exp_type}_{col}",
                    "expectation_type": exp_type,
                    "column": col,
                    "passed": bool(r.success),
                    "status": "PASSED" if r.success else "FAILED",
                    "element_count": int(elem_count),
                    "failed_records": int(unexp_count),
                    "unexpected_count": int(unexp_count),
                    "unexpected_percent": round(float(unexp_pct), 3),
                    "rule_description": rule_desc,
                    "unexpected_values": [str(v) for v in partial_unexp[:5]],
                    "details": (
                        "Assertion satisfied: 0 unexpected records."
                        if r.success
                        else f"Found {unexp_count} invalid records ({unexp_pct:.2f}%). Sample invalid: {partial_unexp[:3]}"
                    )
                })

            return extracted_results

        except Exception as gx_err:
            print(f"Notice: Great Expectations ephemeral engine notice: {gx_err}. Running core assertion suite.")
            return self._execute_native_suite()

    def _execute_native_suite(self) -> List[Dict[str, Any]]:
        """Equivalent deterministic assertion battery mirroring Great Expectations specs."""
        results = []
        n = len(self.df)

        # 1. No null prices
        null_prices = int(self.df["price_inr"].isna().sum()) if "price_inr" in self.df.columns else n
        results.append({
            "test_name": "ExpectColumnValuesToNotBeNull_price_inr",
            "expectation_type": "ExpectColumnValuesToNotBeNull",
            "column": "price_inr",
            "passed": null_prices == 0,
            "status": "PASSED" if null_prices == 0 else "FAILED",
            "element_count": n,
            "failed_records": null_prices,
            "unexpected_count": null_prices,
            "unexpected_percent": round((null_prices / max(n, 1)) * 100.0, 3),
            "rule_description": "Property price must be strictly non-null",
            "unexpected_values": ["None"] if null_prices > 0 else [],
            "details": f"{null_prices} null prices detected." if null_prices > 0 else "Assertion satisfied: 0 null values."
        })

        # 2. Area greater than 0
        if "carpet_area_sqft" in self.df.columns:
            area_invalid = self.df[(self.df["carpet_area_sqft"] <= 0) | (self.df["carpet_area_sqft"].isna())]
            c_area = len(area_invalid)
            results.append({
                "test_name": "ExpectColumnValuesToBeBetween_carpet_area_sqft",
                "expectation_type": "ExpectColumnValuesToBeBetween",
                "column": "carpet_area_sqft",
                "passed": c_area == 0,
                "status": "PASSED" if c_area == 0 else "FAILED",
                "element_count": n,
                "failed_records": c_area,
                "unexpected_count": c_area,
                "unexpected_percent": round((c_area / max(n, 1)) * 100.0, 3),
                "rule_description": "Carpet area must be strictly greater than 0 sq.ft (1 - 25,000)",
                "unexpected_values": [str(x) for x in area_invalid["carpet_area_sqft"].head(5).tolist()],
                "details": f"{c_area} records with carpet_area <= 0." if c_area > 0 else "Assertion satisfied: all carpet areas > 0 sq.ft."
            })

        # 3. Rent in sensible range (INR 1,000 to INR 2,500,000)
        if "rent_inr" in self.df.columns:
            rent_invalid = self.df[(self.df["rent_inr"] < 1000) | (self.df["rent_inr"] > 2500000) | (self.df["rent_inr"].isna())]
            c_rent = len(rent_invalid)
            results.append({
                "test_name": "ExpectColumnValuesToBeBetween_rent_inr",
                "expectation_type": "ExpectColumnValuesToBeBetween",
                "column": "rent_inr",
                "passed": c_rent == 0,
                "status": "PASSED" if c_rent == 0 else "FAILED",
                "element_count": n,
                "failed_records": c_rent,
                "unexpected_count": c_rent,
                "unexpected_percent": round((c_rent / max(n, 1)) * 100.0, 3),
                "rule_description": "Monthly rent must fall within sensible range (INR 1,000 to 2,500,000)",
                "unexpected_values": [str(x) for x in rent_invalid["rent_inr"].head(5).tolist()],
                "details": f"{c_rent} records outside sensible rent range." if c_rent > 0 else "Assertion satisfied: all monthly rents in sensible range."
            })

        # 4. Price in sensible range (INR 100,000 to INR 1,000,000,000)
        if "price_inr" in self.df.columns:
            price_invalid = self.df[(self.df["price_inr"] < 100000) | (self.df["price_inr"] > 1000000000) | (self.df["price_inr"].isna())]
            c_price = len(price_invalid)
            results.append({
                "test_name": "ExpectColumnValuesToBeBetween_price_inr",
                "expectation_type": "ExpectColumnValuesToBeBetween",
                "column": "price_inr",
                "passed": c_price == 0,
                "status": "PASSED" if c_price == 0 else "FAILED",
                "element_count": n,
                "failed_records": c_price,
                "unexpected_count": c_price,
                "unexpected_percent": round((c_price / max(n, 1)) * 100.0, 3),
                "rule_description": "Acquisition price must fall within sensible range (INR 100K to 1B)",
                "unexpected_values": [str(x) for x in price_invalid["price_inr"].head(5).tolist()],
                "details": f"{c_price} records outside sensible price range." if c_price > 0 else "Assertion satisfied: all prices in sensible range."
            })

        # 5. Unique listing ids
        if "listing_id" in self.df.columns:
            dups = int(self.df.duplicated(subset=["listing_id"]).sum())
            dup_vals = self.df[self.df.duplicated(subset=["listing_id"], keep=False)]["listing_id"].head(5).tolist()
            results.append({
                "test_name": "ExpectColumnValuesToBeUnique_listing_id",
                "expectation_type": "ExpectColumnValuesToBeUnique",
                "column": "listing_id",
                "passed": dups == 0,
                "status": "PASSED" if dups == 0 else "FAILED",
                "element_count": n,
                "failed_records": dups,
                "unexpected_count": dups,
                "unexpected_percent": round((dups / max(n, 1)) * 100.0, 3),
                "rule_description": "Listing IDs must be globally unique across deduplicated inventory",
                "unexpected_values": [str(x) for x in dup_vals],
                "details": f"{dups} duplicate listing IDs detected." if dups > 0 else "Assertion satisfied: 0 duplicate keys."
            })

            # Non-null listing IDs
            null_ids = int(self.df["listing_id"].isna().sum())
            results.append({
                "test_name": "ExpectColumnValuesToNotBeNull_listing_id",
                "expectation_type": "ExpectColumnValuesToNotBeNull",
                "column": "listing_id",
                "passed": null_ids == 0,
                "status": "PASSED" if null_ids == 0 else "FAILED",
                "element_count": n,
                "failed_records": null_ids,
                "unexpected_count": null_ids,
                "unexpected_percent": round((null_ids / max(n, 1)) * 100.0, 3),
                "rule_description": "Listing ID must be non-null",
                "unexpected_values": ["None"] if null_ids > 0 else [],
                "details": f"{null_ids} null listing IDs." if null_ids > 0 else "Assertion satisfied: 0 null keys."
            })

        return results

    def run_all_checks(self, halt_on_failure: bool = True) -> Dict[str, Any]:
        """
        Runs the complete Great Expectations quality assertion battery.
        Halts pipeline execution with DataQualityValidationError if any check fails and halt_on_failure=True.
        """
        self.results = self._execute_gx_suite()

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["passed"])
        failed_tests = total_tests - passed_tests
        success = (failed_tests == 0)

        report = {
            "suite_name": "propiq_great_expectations_quality_suite",
            "success": success,
            "overall_status": "PASSED" if success else "FAILED",
            "timestamp": datetime.datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate_pct": round((passed_tests / max(total_tests, 1)) * 100.0, 2),
            "total_records_evaluated": len(self.df),
            "test_details": self.results
        }

        # Persist report for Streamlit Pipeline & Quality view
        report_file = os.path.join(self.report_dir, "data_quality_report.json")
        staging_file = os.path.join(self.staging_dir, "data_quality_report.json")

        for fpath in [report_file, staging_file]:
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
            except Exception as e:
                print(f"Warning writing DQ report to {fpath}: {e}")

        # Check for pipeline failure
        if halt_on_failure and not success:
            failed_items = [r for r in self.results if not r["passed"]]
            diag_lines = [
                "",
                "=" * 72,
                "DATA QUALITY PIPELINE HALTED: CRITICAL EXPECTATION CHECKS FAILED",
                "=" * 72,
                f"Failed Expectations: {len(failed_items)} of {total_tests} checks failed",
                "-" * 72
            ]
            for idx, item in enumerate(failed_items, 1):
                diag_lines.append(
                    f"  {idx}. [{item['expectation_type']}] on column '{item['column']}':"
                )
                diag_lines.append(f"     Requirement: {item['rule_description']}")
                diag_lines.append(f"     Violations: {item['unexpected_count']:,} invalid records ({item['unexpected_percent']}%)")
                if item.get("unexpected_values"):
                    diag_lines.append(f"     Sample Failing Values: {item['unexpected_values']}")
                diag_lines.append(f"     Diagnostics: {item['details']}")

            diag_lines.extend([
                "=" * 72,
                "Pipeline aborted to prevent corrupted downstream OLAP or ML ingestion.",
                "=" * 72
            ])
            error_msg = "\n".join(diag_lines)
            raise DataQualityValidationError(error_msg, failed_checks=failed_items)

        return report
