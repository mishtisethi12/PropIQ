import os
import pytest
from src.warehouse.snowflake_loader import SnowflakeWarehouseManager

def test_snowflake_manager_initialization():
    manager = SnowflakeWarehouseManager()
    assert manager.warehouse == "PROPIQ_WH"
    assert manager.database == "PROPIQ_DW"
    assert manager.schema == "CORE_STAR_SCHEMA"

def test_snowflake_credentials_validation():
    manager = SnowflakeWarehouseManager()
    # With empty default config, is_configured should be False
    if not (os.environ.get("SNOWFLAKE_ACCOUNT") and os.environ.get("SNOWFLAKE_USER") and os.environ.get("SNOWFLAKE_PASSWORD")):
        assert manager.is_configured() is False
        with pytest.raises(ValueError, match="Snowflake credentials missing"):
            manager.get_connection()
