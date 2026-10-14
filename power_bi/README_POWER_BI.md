# PropIQ - Power BI & Tableau Integration Guide

This directory contains resources to connect Power BI and Tableau directly to the PropIQ Kimball Star Schema data warehouse.

---

## 1. Data Source Files
All warehouse tables are automatically exported to `data/warehouse/export_powerbi/`:
- `fact_property_listings.csv`: Listing level measures and keys
- `dim_location.csv`: City, locality, transit score, safety score
- `dim_property_type.csv`: BHK, category, furnishing status
- `dim_amenities.csv`: Parking, gym, pool, security, clubhouse
- `dim_date.csv`: Full date, month, quarter, year, weekend flag

---

## 2. Power BI Setup Steps
1. Open **Power BI Desktop**.
2. Click **Get Data** -> **Folder** or **CSV/Text**.
3. Select the folder `PropIQ/data/warehouse/export_powerbi/`.
4. In the **Model View**, establish the following 1-to-many relationships (Single Direction):
   - `dim_date[date_key]` (1) ---> `fact_property_listings[date_key]` (*)
   - `dim_location[location_key]` (1) ---> `fact_property_listings[location_key]` (*)
   - `dim_property_type[property_type_key]` (1) ---> `fact_property_listings[property_type_key]` (*)
   - `dim_amenities[amenity_key]` (1) ---> `fact_property_listings[amenity_key]` (*)
5. Create a new Measures Table and paste the measures from `dax_measures.txt`.

---

## 3. Recommended Visualizations
- **Card KPI**: Average Rental Yield %
- **Card KPI**: Average Price per Sq.Ft
- **Map Visual**: Latitude/Longitude plotted with bubble size as Rental Yield and color as Price
- **Bar Chart**: Top 10 Localities by Rental Yield
- **Scatter Plot**: Carpet Area vs Price with BHK as legend
