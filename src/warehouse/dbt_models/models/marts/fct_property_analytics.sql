-- Marts model: fct_property_analytics.sql
WITH staged AS (
    SELECT * FROM {{ ref('stg_listings') }}
)
SELECT 
    city,
    locality,
    bhk_count,
    COUNT(listing_id) AS total_listings,
    ROUND(AVG(price_inr), 0) AS avg_price_inr,
    ROUND(AVG(rent_inr), 0) AS avg_monthly_rent_inr,
    ROUND(AVG(price_per_sqft), 2) AS avg_price_per_sqft,
    ROUND(AVG(rental_yield_pct), 2) AS avg_rental_yield_pct,
    MIN(price_inr) AS min_price_inr,
    MAX(price_inr) AS max_price_inr
FROM staged
GROUP BY city, locality, bhk_count
ORDER BY avg_rental_yield_pct DESC
