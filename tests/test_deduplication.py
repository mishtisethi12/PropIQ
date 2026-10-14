import pytest
import pandas as pd
from src.cleaning.deduplication import RealEstateDeduplicator

def test_haversine_distance():
    # Distance between same coordinates should be zero
    dist_zero = RealEstateDeduplicator.haversine_distance_meters(28.628, 77.364, 28.628, 77.364)
    assert dist_zero == pytest.approx(0.0, abs=1e-3)

    # Coordinates ~100m apart
    dist = RealEstateDeduplicator.haversine_distance_meters(28.6280, 77.3649, 28.6289, 77.3649)
    assert 50 < dist < 150

def test_deduplicator_clusters():
    dedup = RealEstateDeduplicator(geo_proximity_meters=75.0, price_variance_tol=0.05, area_variance_tol=0.05)
    
    # Create two duplicate listings with slight coordinate and price jitter
    data = [
        {
            "listing_id": "L1", "city": "Noida", "locality": "Sector 62", "bhk_count": 2,
            "property_type": "Apartment", "carpet_area_sqft": 1050.0, "price_inr": 8000000,
            "latitude": 28.62800, "longitude": 77.36490, "source_portal": "Portal_Alpha"
        },
        {
            "listing_id": "L2", "city": "Noida", "locality": "Sector 62", "bhk_count": 2,
            "property_type": "Apartment", "carpet_area_sqft": 1055.0, "price_inr": 8050000,
            "latitude": 28.62805, "longitude": 77.36495, "source_portal": "Portal_Beta"
        },
        {
            "listing_id": "L3", "city": "Noida", "locality": "Sector 150", "bhk_count": 3,
            "property_type": "Apartment", "carpet_area_sqft": 1800.0, "price_inr": 15000000,
            "latitude": 28.4386, "longitude": 77.4764, "source_portal": "Portal_Alpha"
        }
    ]
    df = pd.DataFrame(data)
    unique_df, audit_df = dedup.deduplicate(df)

    assert len(unique_df) == 2, "Expected 2 unique properties after merging duplicate pair"
    assert len(audit_df) == 1, "Expected 1 entry in audit trail"
    assert audit_df.iloc[0]["master_listing_id"] == "L1"
    assert audit_df.iloc[0]["duplicate_listing_id"] == "L2"
