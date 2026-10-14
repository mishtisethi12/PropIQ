"""
PropIQ - Multi-Source Sample Data Generator
Generates realistic multi-source real estate data for:
1. Portal Alpha CSV (Standard schema)
2. Portal Beta CSV (Messy schema with duplicates & slight variations)
3. Public Locality API Response (JSON)
4. Scraped Unstructured Listings & Lease Agreements (Text/Markdown)
"""

import os
import json
import random
import hashlib
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

CITIES_DATA = {
    "Noida": {
        "localities": [
            {"name": "Sector 62", "lat": 28.6280, "lng": 77.3649, "base_price_sqft": 7500, "rental_yield": 0.045, "transit": 92, "safety": 88, "school": 89},
            {"name": "Sector 137", "lat": 28.5135, "lng": 77.4041, "base_price_sqft": 6800, "rental_yield": 0.048, "transit": 85, "safety": 86, "school": 84},
            {"name": "Sector 150", "lat": 28.4386, "lng": 77.4764, "base_price_sqft": 9200, "rental_yield": 0.038, "transit": 78, "safety": 92, "school": 82},
            {"name": "Sector 76", "lat": 28.5672, "lng": 77.3824, "base_price_sqft": 7100, "rental_yield": 0.044, "transit": 88, "safety": 85, "school": 87},
            {"name": "Sector 128", "lat": 28.5355, "lng": 77.3488, "base_price_sqft": 11500, "rental_yield": 0.036, "transit": 82, "safety": 94, "school": 91},
            {"name": "Sector 18", "lat": 28.5708, "lng": 77.3261, "base_price_sqft": 13000, "rental_yield": 0.042, "transit": 98, "safety": 90, "school": 86},
        ]
    },
    "Greater Noida": {
        "localities": [
            {"name": "Pari Chowk", "lat": 28.4652, "lng": 77.5108, "base_price_sqft": 4800, "rental_yield": 0.052, "transit": 86, "safety": 80, "school": 85},
            {"name": "Omega 1", "lat": 28.4550, "lng": 77.5250, "base_price_sqft": 5200, "rental_yield": 0.049, "transit": 80, "safety": 82, "school": 81},
            {"name": "Greater Noida West", "lat": 28.5983, "lng": 77.4332, "base_price_sqft": 5600, "rental_yield": 0.051, "transit": 79, "safety": 83, "school": 84},
            {"name": "Alpha 1", "lat": 28.4720, "lng": 77.5020, "base_price_sqft": 5100, "rental_yield": 0.050, "transit": 84, "safety": 81, "school": 83},
        ]
    },
    "Gurgaon": {
        "localities": [
            {"name": "Golf Course Road", "lat": 28.4485, "lng": 77.0984, "base_price_sqft": 18500, "rental_yield": 0.034, "transit": 95, "safety": 95, "school": 96},
            {"name": "Cyber City", "lat": 28.4906, "lng": 77.0900, "base_price_sqft": 15500, "rental_yield": 0.039, "transit": 98, "safety": 93, "school": 88},
            {"name": "Sohna Road", "lat": 28.3980, "lng": 77.0420, "base_price_sqft": 9200, "rental_yield": 0.043, "transit": 81, "safety": 87, "school": 89},
            {"name": "Sector 57", "lat": 28.4230, "lng": 77.0750, "base_price_sqft": 11000, "rental_yield": 0.040, "transit": 86, "safety": 89, "school": 90},
        ]
    },
    "Delhi": {
        "localities": [
            {"name": "Dwarka", "lat": 28.5921, "lng": 77.0460, "base_price_sqft": 10500, "rental_yield": 0.036, "transit": 94, "safety": 87, "school": 92},
            {"name": "Rohini", "lat": 28.7166, "lng": 77.1147, "base_price_sqft": 8900, "rental_yield": 0.037, "transit": 91, "safety": 84, "school": 88},
            {"name": "Saket", "lat": 28.5244, "lng": 77.2066, "base_price_sqft": 16500, "rental_yield": 0.033, "transit": 96, "safety": 91, "school": 93},
            {"name": "Mayur Vihar", "lat": 28.6083, "lng": 77.2954, "base_price_sqft": 11200, "rental_yield": 0.038, "transit": 95, "safety": 86, "school": 89},
        ]
    },
    "Bangalore": {
        "localities": [
            {"name": "Indiranagar", "lat": 12.9784, "lng": 77.6408, "base_price_sqft": 16000, "rental_yield": 0.041, "transit": 92, "safety": 92, "school": 94},
            {"name": "Whitefield", "lat": 12.9698, "lng": 77.7499, "base_price_sqft": 9800, "rental_yield": 0.047, "transit": 85, "safety": 89, "school": 91},
            {"name": "HSR Layout", "lat": 12.9121, "lng": 77.6446, "base_price_sqft": 12500, "rental_yield": 0.044, "transit": 89, "safety": 90, "school": 92},
            {"name": "Electronic City", "lat": 12.8452, "lng": 77.6602, "base_price_sqft": 6900, "rental_yield": 0.052, "transit": 80, "safety": 85, "school": 85},
        ]
    }
}

BHK_CONFIG = {
    1: {"area_mean": 600, "area_std": 80, "bathrooms": 1, "balconies": 1},
    2: {"area_mean": 1050, "area_std": 120, "bathrooms": 2, "balconies": 2},
    3: {"area_mean": 1650, "area_std": 180, "bathrooms": 3, "balconies": 3},
    4: {"area_mean": 2400, "area_std": 300, "bathrooms": 4, "balconies": 3},
}

FURNISHING_STATUSES = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]
PROPERTY_TYPES = ["Apartment", "Builder Floor", "Villa", "Penthouse"]

def generate_dataset(num_records=1200):
    records_alpha = []
    
    start_date = datetime(2025, 1, 1)
    
    for i in range(1, num_records + 1):
        city = random.choice(list(CITIES_DATA.keys()))
        loc = random.choice(CITIES_DATA[city]["localities"])
        locality = loc["name"]
        
        bhk = random.choices([1, 2, 3, 4], weights=[0.15, 0.45, 0.30, 0.10])[0]
        bhk_spec = BHK_CONFIG[bhk]
        
        carpet_area = int(np.random.normal(bhk_spec["area_mean"], bhk_spec["area_std"]))
        carpet_area = max(350, min(carpet_area, 5000))
        
        # Calculate price with slight market variation
        price_sqft = loc["base_price_sqft"] * np.random.uniform(0.90, 1.15)
        furnishing = random.choices(FURNISHING_STATUSES, weights=[0.25, 0.55, 0.20])[0]
        furnishing_mult = 1.0 if furnishing == "Unfurnished" else (1.08 if furnishing == "Semi-Furnished" else 1.20)
        
        total_price = int(carpet_area * price_sqft * furnishing_mult)
        # Round to nearest 50,000
        total_price = int(round(total_price / 50000.0) * 50000)
        
        # Calculate realistic monthly rent based on rental yield
        expected_annual_rent = total_price * loc["rental_yield"] * np.random.uniform(0.92, 1.08)
        monthly_rent = int(expected_annual_rent / 12.0)
        monthly_rent = int(round(monthly_rent / 500.0) * 500)
        
        bathrooms = bhk_spec["bathrooms"]
        balconies = bhk_spec["balconies"]
        prop_type = random.choices(PROPERTY_TYPES, weights=[0.70, 0.20, 0.08, 0.02])[0]
        
        # Amenities
        has_parking = random.choice([1, 1, 1, 0])
        has_gym = random.choice([1, 1, 0])
        has_pool = 1 if (prop_type in ["Apartment", "Penthouse"] and random.random() > 0.4) else 0
        has_clubhouse = 1 if has_pool else random.choice([1, 0])
        has_security = 1
        has_power_backup = random.choice([1, 1, 1, 0])
        
        # Coordinate jitter
        lat = loc["lat"] + np.random.normal(0, 0.005)
        lng = loc["lng"] + np.random.normal(0, 0.005)
        
        # Date posted
        posted_days_ago = random.randint(1, 365)
        posted_date = (datetime.now() - timedelta(days=posted_days_ago)).strftime("%Y-%m-%d")
        
        title = f"Spacious {bhk} BHK {prop_type} in {locality}, {city}"
        
        alpha_id = f"ALP-{10000 + i}"
        
        records_alpha.append({
            "listing_id": alpha_id,
            "title": title,
            "city": city,
            "locality": locality,
            "property_type": prop_type,
            "bhk_count": bhk,
            "carpet_area_sqft": carpet_area,
            "price_inr": total_price,
            "rent_inr": monthly_rent,
            "bathrooms": bathrooms,
            "balconies": balconies,
            "furnishing_status": furnishing,
            "parking": has_parking,
            "gym": has_gym,
            "swimming_pool": has_pool,
            "clubhouse": has_clubhouse,
            "security": has_security,
            "power_backup": has_power_backup,
            "latitude": round(lat, 6),
            "longitude": round(lng, 6),
            "posted_date": posted_date,
            "source_portal": "Portal_Alpha"
        })

    df_alpha = pd.DataFrame(records_alpha)
    
    # -------------------------------------------------------------
    # Generate Portal Beta (Messy schema + 25% duplicate listings)
    # -------------------------------------------------------------
    records_beta = []
    
    # Take 200 records from df_alpha to clone as duplicates with variations
    duplicate_source = df_alpha.sample(n=250, random_state=42)
    
    for idx, row in duplicate_source.iterrows():
        # Inject realistic portal variation:
        # e.g., price slightly shifted (±1.5%), title changed, messy location text
        price_shift = row["price_inr"] * random.uniform(0.985, 1.015)
        price_lakhs = round(price_shift / 100000.0, 2)
        rent_shift = int(round(row["rent_inr"] * random.uniform(0.98, 1.02) / 500.0) * 500)
        
        # Alternate phrasing in title
        bhk = row["bhk_count"]
        prop = row["property_type"]
        loc_name = row["locality"]
        city_name = row["city"]
        
        messy_titles = [
            f"{bhk}BHK Flat for Sale at {loc_name} {city_name}",
            f"Ready to Move {bhk} BHK in {loc_name}",
            f"Luxury {bhk} Bedroom Residence {loc_name}, {city_name}",
            f"Prime Location {bhk}BHK {prop} {city_name}"
        ]
        
        # Coordinate slight jitter (e.g. 15-30 meters)
        lat_jitter = row["latitude"] + random.uniform(-0.0002, 0.0002)
        lng_jitter = row["longitude"] + random.uniform(-0.0002, 0.0002)
        
        beta_id = f"BET-{50000 + idx}"
        
        records_beta.append({
            "id": beta_id,
            "prop_name": random.choice(messy_titles),
            "location_text": f"{loc_name}, {city_name}, Landmark near metro",
            "city_name": city_name,
            "bedrooms": f"{bhk} BHK",
            "sqft": f"{row['carpet_area_sqft']} sq.ft",
            "cost_lakhs": price_lakhs,
            "monthly_rent_inr": rent_shift,
            "baths": row["bathrooms"],
            "furnish": row["furnishing_status"],
            "lat_long": f"{round(lat_jitter, 6)}, {round(lng_jitter, 6)}",
            "amenities_list": f"Parking: {row['parking']}, Gym: {row['gym']}, Pool: {row['swimming_pool']}, Security: {row['security']}",
            "upload_ts": (datetime.strptime(row["posted_date"], "%Y-%m-%d") + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d %H:%M:%S"),
            "portal": "Portal_Beta"
        })
        
    # Add 400 unique listings to Portal Beta as well
    for j in range(1, 401):
        city = random.choice(list(CITIES_DATA.keys()))
        loc = random.choice(CITIES_DATA[city]["localities"])
        locality = loc["name"]
        bhk = random.choice([1, 2, 3, 4])
        bhk_spec = BHK_CONFIG[bhk]
        carpet_area = int(np.random.normal(bhk_spec["area_mean"], bhk_spec["area_std"]))
        carpet_area = max(350, min(carpet_area, 5000))
        price_sqft = loc["base_price_sqft"] * np.random.uniform(0.90, 1.15)
        furnishing = random.choice(FURNISHING_STATUSES)
        furnishing_mult = 1.0 if furnishing == "Unfurnished" else (1.08 if furnishing == "Semi-Furnished" else 1.20)
        total_price = int(carpet_area * price_sqft * furnishing_mult)
        price_lakhs = round(total_price / 100000.0, 2)
        monthly_rent = int(round((total_price * loc["rental_yield"] / 12.0) / 500.0) * 500)
        lat = loc["lat"] + np.random.normal(0, 0.005)
        lng = loc["lng"] + np.random.normal(0, 0.005)
        
        beta_id = f"BET-{60000 + j}"
        records_beta.append({
            "id": beta_id,
            "prop_name": f"{bhk} BHK Apartment in {locality}",
            "location_text": f"{locality}, {city}",
            "city_name": city,
            "bedrooms": f"{bhk} BHK",
            "sqft": f"{carpet_area} sq.ft",
            "cost_lakhs": price_lakhs,
            "monthly_rent_inr": monthly_rent,
            "baths": bhk_spec["bathrooms"],
            "furnish": furnishing,
            "lat_long": f"{round(lat, 6)}, {round(lng, 6)}",
            "amenities_list": "Parking: 1, Gym: 1, Pool: 0, Security: 1",
            "upload_ts": (datetime.now() - timedelta(days=random.randint(1, 180))).strftime("%Y-%m-%d %H:%M:%S"),
            "portal": "Portal_Beta"
        })
        
    df_beta = pd.DataFrame(records_beta)
    
    return df_alpha, df_beta

def generate_locality_api_data():
    """Generates JSON response simulating Public Infrastructure & Locality Ratings API"""
    api_payload = []
    for city, data in CITIES_DATA.items():
        for loc in data["localities"]:
            api_payload.append({
                "locality_id": f"LOC-{city[:3].upper()}-{hashlib.md5(loc['name'].encode()).hexdigest()[:4].upper()}",
                "city": city,
                "locality_name": loc["name"],
                "latitude": loc["lat"],
                "longitude": loc["lng"],
                "transit_score": loc["transit"],
                "safety_score": loc["safety"],
                "school_score": loc["school"],
                "hospital_count": random.randint(3, 12),
                "metro_distance_km": round(random.uniform(0.3, 3.5), 2),
                "commercial_hub_proximity_km": round(random.uniform(1.0, 7.5), 2),
                "green_cover_pct": random.randint(15, 45),
                "last_audited": "2026-03-01"
            })
    return api_payload

def generate_unstructured_corpus():
    """Generates real estate listing reviews, landlord notes, and lease agreements for RAG"""
    corpus = []
    
    # 1. Locality investment & rental yield deep dives
    for city, data in CITIES_DATA.items():
        for loc in data["localities"]:
            name = loc["name"]
            yield_pct = loc["rental_yield"] * 100
            corpus.append({
                "doc_id": f"LOC_REPORT_{city}_{name.replace(' ', '_')}",
                "type": "locality_report",
                "title": f"Comprehensive Locality Profile: {name}, {city}",
                "content": f"""
Locality Profile: {name}, {city}
Overview: {name} is one of the key residential and commercial corridors in {city}.
Rental Yield Analysis:
- Historical Rental Yield: Average {yield_pct:.2f}% gross annual yield.
- High-yield configurations: 2BHK and 1BHK units in {name} experience the fastest rental turnaround, with typical occupancy rates exceeding 93%.
- Average tenant profile: IT professionals, young families, and corporate executives seeking proximity to metro corridors.
Infrastructure & Amenities:
- Transit Connectivity Rating: {loc['transit']}/100. Direct access to arterial metro corridors and expressways.
- Neighborhood Safety Rating: {loc['safety']}/100. Gated societies, 24/7 CCTV surveillance, and regular police patrolling.
- School & Educational Hubs Rating: {loc['school']}/100. Top international schools within 4km radius.
Investment Outlook:
- Capital Appreciation: 8-12% YoY projected growth over the next 3-5 years.
- Maintenance & Society Dues: Typically between INR 2.50 to INR 3.80 per sq.ft.
"""
            })
            
    # 2. Standard Indian Residential Lease Agreements & Clauses
    lease_templates = [
        {
            "doc_id": "LEASE_AGREEMENT_STANDARD_NCR",
            "type": "lease_agreement",
            "title": "Standard Model Residential Lease Agreement (NCR - Noida / Gurgaon / Delhi)",
            "content": """
STANDARD MODEL RESIDENTIAL TENANCY AGREEMENT (NCR REGION)
Key Clauses & Legal Provisions:
1. Term of Tenancy: The tenancy shall be for an initial period of 11 (eleven) months with an option to renew upon mutual agreement.
2. Security Deposit:
   - The Tenant shall deposit an amount equivalent to 2 (two) months' rent as interest-free refundable security deposit.
   - The deposit shall be refunded within 15 working days following vacant possession, minus verified deductions for tenant-caused damages or unpaid utility bills.
3. Rent Escalation Clause:
   - In case of lease renewal after 11 months, the monthly rent shall escalate by a standard 5% to 7% of the prevailing monthly rental value.
4. Maintenance and Society Charges:
   - Monthly society maintenance charges (RWA dues) are to be paid directly by the Tenant unless explicitly agreed in writing as landlord-inclusive.
   - Major structural repairs (roof leaks, exterior plumbing) are the Landlord's liability; minor electrical and fixture repairs under INR 1,500 are Tenant's liability.
5. Notice Period & Lock-in:
   - A mandatory lock-in period of 3 (three) months applies from the commencement date.
   - Either party may terminate the tenancy by providing 1 (one) full calendar month written notice or rent in lieu thereof.
"""
        },
        {
            "doc_id": "LEASE_AGREEMENT_BANGALORE_MODEL",
            "type": "lease_agreement",
            "title": "Karnataka Model Residential Tenancy Agreement (Bangalore)",
            "content": """
RESIDENTIAL LEASE AND LICENSE AGREEMENT (BANGALORE URBAN)
Key Terms & Regulatory Framework:
1. Tenancy Period: 11 months renewable subject to terms.
2. Security Deposit Terms:
   - Standard security deposit in Bangalore ranges between 4 to 6 months of monthly rent (reduced from historical 10-month norms under Model Tenancy Act).
   - Deposit refund is legally binding within 30 days of handover, subject to painting charges (typically 1 month's rent deduction) and utility settlements.
3. Lock-in Period: 6 months minimum lock-in period during which early vacancy incurs deposit forfeiture.
4. Permitted Use: Solely for residential purposes. Subletting or commercial operations are strictly prohibited without written consent.
5. Dispute Resolution: Subject to jurisdiction of the Rent Court and Bengaluru Urban District Civil Courts.
"""
        }
    ]
    
    corpus.extend(lease_templates)
    return corpus

def main():
    base_dir = "C:/Users/misht/.gemini/antigravity/scratch/PropIQ"
    
    print("[1/4] Generating Portal Alpha & Portal Beta datasets...")
    df_alpha, df_beta = generate_dataset(num_records=1200)
    
    alpha_path = os.path.join(base_dir, "data/raw/csv_sources/portal_alpha_listings.csv")
    beta_path = os.path.join(base_dir, "data/raw/csv_sources/portal_beta_listings.csv")
    
    df_alpha.to_csv(alpha_path, index=False)
    df_beta.to_csv(beta_path, index=False)
    print(f"  -> Saved Portal Alpha: {len(df_alpha)} records to {alpha_path}")
    print(f"  -> Saved Portal Beta: {len(df_beta)} records (including duplicates) to {beta_path}")
    
    print("[2/4] Generating Public Infrastructure API data...")
    api_data = generate_locality_api_data()
    api_path = os.path.join(base_dir, "data/raw/api_responses/locality_infrastructure.json")
    with open(api_path, "w", encoding="utf-8") as f:
        json.dump(api_data, f, indent=2)
    print(f"  -> Saved {len(api_data)} locality records to {api_path}")
    
    print("[3/4] Generating Unstructured Listing Descriptions & Lease Documents...")
    corpus = generate_unstructured_corpus()
    docs_path = os.path.join(base_dir, "data/docs/real_estate_knowledge_base.json")
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)
        
    for item in corpus:
        doc_filename = os.path.join(base_dir, f"data/raw/scraped_text/{item['doc_id']}.md")
        with open(doc_filename, "w", encoding="utf-8") as f:
            f.write(f"# {item['title']}\n\n{item['content'].strip()}\n")
            
    print(f"  -> Saved {len(corpus)} RAG documents to {docs_path} and data/raw/scraped_text/")
    print("[4/4] Data Generation Complete!")

if __name__ == "__main__":
    main()
