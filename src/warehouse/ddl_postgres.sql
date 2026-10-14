-- ====================================================================
-- PropIQ - PostgreSQL OLTP / OLAP Star Schema DDL with Declarative Partitioning
-- ====================================================================

-- 1. Dimension Tables
CREATE TABLE IF NOT EXISTS dim_date (
    date_key INT PRIMARY KEY,
    full_date DATE NOT NULL,
    day_of_month INT,
    day_name VARCHAR(15),
    day_of_week INT,
    month INT,
    month_name VARCHAR(15),
    quarter INT,
    year INT,
    year_month VARCHAR(7),
    is_weekend INT
);

CREATE TABLE IF NOT EXISTS dim_location (
    location_key VARCHAR(32) PRIMARY KEY,
    city VARCHAR(64) NOT NULL,
    locality VARCHAR(128) NOT NULL,
    transit_score INT,
    safety_score INT,
    school_score INT,
    metro_distance_km NUMERIC(5, 2),
    hospital_count INT
);

CREATE TABLE IF NOT EXISTS dim_property_type (
    property_type_key VARCHAR(32) PRIMARY KEY,
    bhk_count INT NOT NULL,
    bhk_label VARCHAR(16),
    property_category VARCHAR(32),
    furnishing_status VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS dim_amenities (
    amenity_key VARCHAR(32) PRIMARY KEY,
    has_parking INT,
    has_gym INT,
    has_swimming_pool INT,
    has_clubhouse INT,
    has_security INT,
    has_power_backup INT,
    amenity_tier VARCHAR(16)
);

-- 2. Partitioned Fact Table by City (Declarative List Partitioning)
CREATE TABLE IF NOT EXISTS fact_property_listings (
    listing_id VARCHAR(64) NOT NULL,
    date_key INT NOT NULL,
    location_key VARCHAR(32) NOT NULL,
    property_type_key VARCHAR(32) NOT NULL,
    amenity_key VARCHAR(32) NOT NULL,
    city VARCHAR(64) NOT NULL,
    year_month VARCHAR(7) NOT NULL,
    price_inr BIGINT NOT NULL,
    rent_inr INT NOT NULL,
    carpet_area_sqft NUMERIC(8, 2) NOT NULL,
    price_per_sqft NUMERIC(10, 2),
    rental_yield_pct NUMERIC(5, 2),
    bathrooms INT,
    balconies INT,
    days_on_market INT,
    latitude NUMERIC(9, 6),
    longitude NUMERIC(9, 6),
    source_portal VARCHAR(32),
    PRIMARY KEY (listing_id, city)
) PARTITION BY LIST (city);

-- 3. City Partitions (Partition Pruning target)
CREATE TABLE IF NOT EXISTS fact_listings_noida 
    PARTITION OF fact_property_listings FOR VALUES IN ('Noida');

CREATE TABLE IF NOT EXISTS fact_listings_greater_noida 
    PARTITION OF fact_property_listings FOR VALUES IN ('Greater Noida');

CREATE TABLE IF NOT EXISTS fact_listings_gurgaon 
    PARTITION OF fact_property_listings FOR VALUES IN ('Gurgaon');

CREATE TABLE IF NOT EXISTS fact_listings_delhi 
    PARTITION OF fact_property_listings FOR VALUES IN ('Delhi');

CREATE TABLE IF NOT EXISTS fact_listings_bangalore 
    PARTITION OF fact_property_listings FOR VALUES IN ('Bangalore');

-- 4. B-Tree Indexes on Foreign Keys and Query Columns
CREATE INDEX IF NOT EXISTS idx_fact_noida_loc ON fact_listings_noida (location_key);
CREATE INDEX IF NOT EXISTS idx_fact_gurgaon_loc ON fact_listings_gurgaon (location_key);
CREATE INDEX IF NOT EXISTS idx_fact_yield ON fact_property_listings (rental_yield_pct DESC);
