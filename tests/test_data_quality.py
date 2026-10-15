import os
import json
import pytest
import pandas as pd
from src.cleaning.data_quality import DataQualitySuite, DataQualityValidationError

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def clean_sample_dataframe():
    return pd.DataFrame({
        "listing_id": [f"TEST_PROP_{i}" for i in range(1, 11)],
        "price_inr": [5000000.0 + i * 500000.0 for i in range(10)],
        "rent_inr": [20000.0 + i * 2000.0 for i in range(10)],
        "carpet_area_sqft": [800.0 + i * 100.0 for i in range(10)],
        "bhk_count": [2 for _ in range(10)],
        "latitude": [28.5 + (i * 0.01) for i in range(10)],
        "longitude": [77.3 + (i * 0.01) for i in range(10)]
    })


def test_clean_dataframe_passes_all_checks(clean_sample_dataframe):
    dq = DataQualitySuite(clean_sample_dataframe)
    report = dq.run_all_checks(halt_on_failure=True)
    assert report["success"] is True
    assert report["failed_tests"] == 0
    assert report["passed_tests"] == report["total_tests"]
    assert report["pass_rate_pct"] == 100.0


def test_null_price_fails_and_halts(clean_sample_dataframe):
    df_corrupt = clean_sample_dataframe.copy()
    df_corrupt.loc[2, "price_inr"] = None

    dq = DataQualitySuite(df_corrupt)
    with pytest.raises(DataQualityValidationError) as exc_info:
        dq.run_all_checks(halt_on_failure=True)

    err = str(exc_info.value)
    assert "DATA QUALITY PIPELINE HALTED" in err
    assert "price_inr" in err
    assert any(check["column"] == "price_inr" and not check["passed"] for check in exc_info.value.failed_checks)


def test_zero_or_negative_area_fails_and_halts(clean_sample_dataframe):
    df_corrupt = clean_sample_dataframe.copy()
    df_corrupt.loc[1, "carpet_area_sqft"] = 0.0

    dq = DataQualitySuite(df_corrupt)
    with pytest.raises(DataQualityValidationError) as exc_info:
        dq.run_all_checks(halt_on_failure=True)

    err = str(exc_info.value)
    assert "DATA QUALITY PIPELINE HALTED" in err
    assert "carpet_area_sqft" in err


def test_out_of_range_price_fails_and_halts(clean_sample_dataframe):
    df_corrupt = clean_sample_dataframe.copy()
    df_corrupt.loc[0, "price_inr"] = 50.0  # Absurdly low price

    dq = DataQualitySuite(df_corrupt)
    with pytest.raises(DataQualityValidationError) as exc_info:
        dq.run_all_checks(halt_on_failure=True)

    err = str(exc_info.value)
    assert "DATA QUALITY PIPELINE HALTED" in err
    assert "price_inr" in err


def test_out_of_range_rent_fails_and_halts(clean_sample_dataframe):
    df_corrupt = clean_sample_dataframe.copy()
    df_corrupt.loc[0, "rent_inr"] = 10000000.0  # 1 Crore per month rent on regular unit

    dq = DataQualitySuite(df_corrupt)
    with pytest.raises(DataQualityValidationError) as exc_info:
        dq.run_all_checks(halt_on_failure=True)

    err = str(exc_info.value)
    assert "DATA QUALITY PIPELINE HALTED" in err
    assert "rent_inr" in err


def test_duplicate_listing_ids_fails_and_halts(clean_sample_dataframe):
    df_corrupt = clean_sample_dataframe.copy()
    df_corrupt.loc[1, "listing_id"] = df_corrupt.loc[0, "listing_id"]  # Duplicate key

    dq = DataQualitySuite(df_corrupt)
    with pytest.raises(DataQualityValidationError) as exc_info:
        dq.run_all_checks(halt_on_failure=True)

    err = str(exc_info.value)
    assert "DATA QUALITY PIPELINE HALTED" in err
    assert "listing_id" in err


def test_report_json_persisted(clean_sample_dataframe):
    dq = DataQualitySuite(clean_sample_dataframe)
    dq.run_all_checks(halt_on_failure=False)
    report_path = os.path.join(BASE_DIR, "artifacts/quality/data_quality_report.json")
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "total_tests" in data
    assert "test_details" in data
    assert len(data["test_details"]) > 0
