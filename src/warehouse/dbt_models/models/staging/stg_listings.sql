-- Staging model: stg_listings.sql
WITH raw_source AS (
    SELECT 
        listing_id,
        city,
        locality,
        bhk_count,
        carpet_area_sqft,
        price_inr,
        rent_inr,
        bathrooms,
        balconies,
        parking,
        gym,
        swimming_pool,
        posted_date,
        source_portal
    FROM {{ source('raw_staging', 'raw_property_listings') }}
)
SELECT 
    listing_id,
    TRIM(city) AS city,
    TRIM(locality) AS locality,
    CAST(bhk_count AS INTEGER) AS bhk_count,
    CAST(carpet_area_sqft AS NUMERIC(10, 2)) AS carpet_area_sqft,
    CAST(price_inr AS NUMERIC(14, 2)) AS price_inr,
    CAST(rent_inr AS NUMERIC(10, 2)) AS rent_inr,
    ROUND((price_inr / NULLIF(carpet_area_sqft, 0)), 2) AS price_per_sqft,
    ROUND(((rent_inr * 12.0) / NULLIF(price_inr, 0) * 100.0), 2) AS rental_yield_pct,
    posted_date,
    source_portal
FROM raw_source
