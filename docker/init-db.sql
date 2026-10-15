-- ====================================================================
-- PropIQ - PostgreSQL OLTP Raw Layer & OLAP Star Schema DDL
-- Executed automatically on container initialization in propiq_db
-- ====================================================================

-- 1. Operational Ingestion Layer (Raw Tables)
CREATE TABLE IF NOT EXISTS raw_property_listings (
    id SERIAL PRIMARY KEY,
    listing_id VARCHAR(64) UNIQUE,
    title TEXT,
    city VARCHAR(64),
    locality VARCHAR(128),
    property_type VARCHAR(64),
    bhk_count INT,
    carpet_area_sqft NUMERIC(10,2),
    price_inr BIGINT,
    rent_inr INT,
    bathrooms INT,
    balconies INT,
    furnishing_status VARCHAR(64),
    parking INT,
    gym INT,
    swimming_pool INT,
    clubhouse INT,
    security_guard INT,
    power_backup INT,
    latitude NUMERIC(10,6),
    longitude NUMERIC(10,6),
    posted_date DATE,
    source_portal VARCHAR(64),
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_locality_infrastructure (
    id SERIAL PRIMARY KEY,
    locality VARCHAR(128),
    city VARCHAR(64),
    transit_score INT,
    safety_score INT,
    school_score INT,
    metro_distance_km NUMERIC(5,2),
    hospital_count INT,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_scraped_documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(256),
    doc_type VARCHAR(64),
    title TEXT,
    content_text TEXT,
    file_size_bytes INT,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Analytical Dimension Tables
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

-- 3. Partitioned Fact Table (Declarative Partitioning by City)
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
    carpet_area_sqft NUMERIC(10,2) NOT NULL,
    price_per_sqft NUMERIC(10,2) NOT NULL,
    rental_yield_pct NUMERIC(5,2) NOT NULL,
    bathrooms INT NOT NULL,
    balconies INT NOT NULL,
    days_on_market INT NOT NULL,
    latitude NUMERIC(10,6) NOT NULL,
    longitude NUMERIC(10,6) NOT NULL,
    source_portal VARCHAR(64) NOT NULL,
    PRIMARY KEY (listing_id, city)
) PARTITION BY LIST (city);

-- Declarative City Partitions
CREATE TABLE IF NOT EXISTS fact_listings_delhi 
    PARTITION OF fact_property_listings FOR VALUES IN ('Delhi');

CREATE TABLE IF NOT EXISTS fact_listings_noida 
    PARTITION OF fact_property_listings FOR VALUES IN ('Noida');

CREATE TABLE IF NOT EXISTS fact_listings_gurgaon 
    PARTITION OF fact_property_listings FOR VALUES IN ('Gurgaon');

CREATE TABLE IF NOT EXISTS fact_listings_greater_noida 
    PARTITION OF fact_property_listings FOR VALUES IN ('Greater Noida');

CREATE TABLE IF NOT EXISTS fact_listings_bangalore 
    PARTITION OF fact_property_listings FOR VALUES IN ('Bangalore');

-- Indices for analytical filtering
CREATE INDEX IF NOT EXISTS idx_fact_city ON fact_property_listings (city);
CREATE INDEX IF NOT EXISTS idx_fact_location ON fact_property_listings (location_key);
CREATE INDEX IF NOT EXISTS idx_fact_date ON fact_property_listings (date_key);
CREATE INDEX IF NOT EXISTS idx_raw_city_locality ON raw_property_listings (city, locality);
