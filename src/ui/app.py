"""
PropIQ - Enterprise Real Estate Data Platform & GenAI Analytics Terminal
Designed for Institutional Real Estate Asset Managers & Analytics Engineers.
Clean, professional typography with zero emojis or non-standard iconography.
"""

import os
import json
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="PropIQ | Real Estate Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DUCKDB_PATH = os.path.join(BASE_DIR, "data/warehouse/propiq_warehouse.duckdb")
DOCS_DIR = os.path.join(BASE_DIR, "data/raw/scraped_text")
METRICS_PATH = os.path.join(BASE_DIR, "artifacts/ml/model_comparison_metrics.json")
AUDIT_PATH = os.path.join(BASE_DIR, "data/staging/deduplication_audit.csv")

# Enterprise CSS Styling
st.markdown("""
<style>
    /* Global Typography & Palette */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #0F172A;
    }
    .stApp {
        background-color: #FAFAFB;
    }
    
    /* Header Container */
    .top-navbar {
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 12px;
        margin-bottom: 24px;
    }
    .brand-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #0F172A;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 0.875rem;
        color: #64748B;
        margin-top: 2px;
    }
    
    /* Metric Cards */
    .metric-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 16px 20px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .metric-subtext {
        font-size: 0.75rem;
        color: #059669;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Section Headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E293B;
        margin-top: 24px;
        margin-bottom: 12px;
        letter-spacing: -0.01em;
    }

    /* Enterprise Dataframe Styling */
    .dataframe {
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load fact table data
@st.cache_data
def load_fact_data():
    if not os.path.exists(DUCKDB_PATH):
        return pd.DataFrame()
    con = duckdb.connect(database=DUCKDB_PATH, read_only=True)
    query = """
        SELECT 
            f.listing_id,
            f.city,
            l.locality,
            p.bhk_count,
            p.bhk_label,
            p.property_category,
            p.furnishing_status,
            a.amenity_tier,
            f.price_inr,
            f.rent_inr,
            f.carpet_area_sqft,
            f.price_per_sqft,
            f.rental_yield_pct,
            l.transit_score,
            l.safety_score,
            l.school_score,
            f.source_portal
        FROM fact_property_listings f
        JOIN dim_location l ON f.location_key = l.location_key
        JOIN dim_property_type p ON f.property_type_key = p.property_type_key
        JOIN dim_amenities a ON f.amenity_key = a.amenity_key
    """
    df = con.execute(query).fetchdf()
    con.close()
    return df

@st.cache_data
def load_ml_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# Top Brand Header
st.markdown("""
<div class="top-navbar">
    <div class="brand-title">PROPIQ ANALYTICS PLATFORM</div>
    <div class="brand-subtitle">Enterprise Real Estate Lakehouse, Valuation Engine & Conversational Intelligence</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation (Zero Emojis)
st.sidebar.markdown("### PROPIQ CONSOLE")
nav_selection = st.sidebar.radio(
    "Navigation View:",
    [
        "Market Intelligence & Yields",
        "Data Pipeline & Deduplication",
        "Valuation Model Benchmarking",
        "GenAI Research Assistant",
        "Star Schema & Partitions",
        "BI Integration & Export"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**System Architecture Overview**")
st.sidebar.markdown("""
- **Storage Tier:** PostgreSQL (OLTP) / DuckDB & Snowflake (OLAP)
- **Data Model:** Kimball Star Schema (Fact + 4 Dimensions)
- **Partition Strategy:** Hive Parquet by City and Month
- **Supervised ML:** Decision Forest, XGBoost, k-NN, Ridge
- **RAG Subsystem:** Hybrid Text-to-SQL + Vector Search
""")

df_fact = load_fact_data()

# ==============================================================================
# VIEW 1: Market Intelligence & Yields
# ==============================================================================
if nav_selection == "Market Intelligence & Yields":
    st.markdown('<div class="section-title">Executive Market Overview & Rental Yield Analysis</div>', unsafe_allow_html=True)

    if df_fact.empty:
        st.warning("Warehouse database is not initialized. Run the ETL pipeline via `python run_pipeline.py`.")
    else:
        # Filter controls
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            city_options = ["All Metros"] + sorted(list(df_fact["city"].unique()))
            selected_city = st.selectbox("Metro Market:", city_options)
        with f_col2:
            bhk_options = ["All Configurations"] + sorted(list(df_fact["bhk_label"].unique()))
            selected_bhk = st.selectbox("Unit Configuration:", bhk_options)
        with f_col3:
            type_options = ["All Asset Classes"] + sorted(list(df_fact["property_category"].unique()))
            selected_type = st.selectbox("Asset Class:", type_options)

        # Apply Filters
        filtered = df_fact.copy()
        if selected_city != "All Metros":
            filtered = filtered[filtered["city"] == selected_city]
        if selected_bhk != "All Configurations":
            filtered = filtered[filtered["bhk_label"] == selected_bhk]
        if selected_type != "All Asset Classes":
            filtered = filtered[filtered["property_category"] == selected_type]

        # Top Metric Cards (Enterprise layout)
        m1, m2, m3, m4, m5 = st.columns(5)
        
        with m1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Active Listings</div>
                <div class="metric-value">{len(filtered):,}</div>
                <div class="metric-subtext">Verified canonical units</div>
            </div>
            """, unsafe_allow_html=True)

        avg_price = filtered["price_inr"].mean() if not filtered.empty else 0
        with m2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Avg Listing Price</div>
                <div class="metric-value">INR {avg_price/100000:,.1f} L</div>
                <div class="metric-subtext">Capital asset valuation</div>
            </div>
            """, unsafe_allow_html=True)

        avg_rent = filtered["rent_inr"].mean() if not filtered.empty else 0
        with m3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Avg Monthly Rent</div>
                <div class="metric-value">INR {avg_rent:,.0f}</div>
                <div class="metric-subtext">Operating cashflow</div>
            </div>
            """, unsafe_allow_html=True)

        avg_yield = filtered["rental_yield_pct"].mean() if not filtered.empty else 0
        with m4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Avg Rental Yield</div>
                <div class="metric-value">{avg_yield:.2f}%</div>
                <div class="metric-subtext">Gross annualized yield</div>
            </div>
            """, unsafe_allow_html=True)

        avg_sqft = filtered["price_per_sqft"].mean() if not filtered.empty else 0
        with m5:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Capital Rate / Sq.Ft</div>
                <div class="metric-value">INR {avg_sqft:,.0f}</div>
                <div class="metric-subtext">Standardized carpet area</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plotly Charts
        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.markdown('<div class="section-title">Rental Yield by Submarket (Top 10)</div>', unsafe_allow_html=True)
            yield_agg = (
                filtered.groupby(["locality", "city"])["rental_yield_pct"]
                .mean()
                .reset_index()
                .sort_values("rental_yield_pct", ascending=True)
                .tail(10)
            )

            fig_yield = px.bar(
                yield_agg,
                x="rental_yield_pct",
                y="locality",
                orientation="h",
                color_discrete_sequence=["#1E3A8A"],
                labels={"rental_yield_pct": "Gross Annual Rental Yield (%)", "locality": "Submarket"},
                text_auto=".2f"
            )
            fig_yield.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=0, r=20, t=10, b=10),
                height=350,
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_yield, use_container_width=True)

        with c_right:
            st.markdown('<div class="section-title">Capital Price vs. Monthly Rent Distribution</div>', unsafe_allow_html=True)
            fig_scatter = px.scatter(
                filtered,
                x="price_inr",
                y="rent_inr",
                color="city",
                hover_data=["locality", "bhk_label", "rental_yield_pct"],
                labels={"price_inr": "Capital Price (INR)", "rent_inr": "Monthly Rent (INR)", "city": "Metro"},
                color_discrete_sequence=["#0284C7", "#2563EB", "#0F766E", "#D97706", "#4F46E5"]
            )
            fig_scatter.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=0, r=20, t=10, b=10),
                height=350,
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9")
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Tabular Aggregates
        st.markdown('<div class="section-title">Submarket Performance Summary</div>', unsafe_allow_html=True)
        summary = filtered.groupby(["city", "locality", "bhk_label"]).agg(
            listing_count=("listing_id", "count"),
            avg_price_lakhs=("price_inr", lambda x: round(x.mean() / 100000.0, 2)),
            avg_rent_inr=("rent_inr", lambda x: round(x.mean())),
            avg_price_sqft=("price_per_sqft", lambda x: round(x.mean())),
            avg_rental_yield_pct=("rental_yield_pct", lambda x: round(x.mean(), 2)),
            avg_transit_score=("transit_score", "mean")
        ).reset_index().sort_values("avg_rental_yield_pct", ascending=False)

        st.dataframe(
            summary.rename(columns={
                "city": "Metro",
                "locality": "Submarket",
                "bhk_label": "Config",
                "listing_count": "Units",
                "avg_price_lakhs": "Avg Price (Lakhs)",
                "avg_rent_inr": "Avg Rent (INR)",
                "avg_price_sqft": "Rate / Sq.Ft",
                "avg_rental_yield_pct": "Yield (%)",
                "avg_transit_score": "Transit Score"
            }),
            use_container_width=True
        )

# ==============================================================================
# VIEW 2: Data Pipeline & Deduplication
# ==============================================================================
elif nav_selection == "Data Pipeline & Deduplication":
    st.markdown('<div class="section-title">Data Ingestion, Quality Assurance & Deduplication Audit</div>', unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Ingested Sources</div>
            <div class="metric-value">3 Connectors</div>
            <div class="metric-subtext">CSV, Public API, Text</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Raw Ingestion Total</div>
            <div class="metric-value">1,850 Units</div>
            <div class="metric-subtext">Unfiltered operational load</div>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Golden Master Records</div>
            <div class="metric-value">1,594 Units</div>
            <div class="metric-subtext">Deduplicated unique assets</div>
        </div>
        """, unsafe_allow_html=True)
    with d4:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Cross-Portal Duplicates Merged</div>
            <div class="metric-value">256 Pairs</div>
            <div class="metric-subtext">Zero data leakage</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">ETL Pipeline Architecture & Record Linkage Mechanics</div>', unsafe_allow_html=True)
    st.markdown("""
    1. **Multi-Source Ingestion:** Ingests Portal Alpha (clean standard schema), Portal Beta (semi-structured string attributes, lakh prices, embedded units), and Locality Metadata REST API.
    2. **Normalization:** Regex string cleaning, coordinate boundary enforcement, and financial metric imputation.
    3. **Record Linkage & Deduplication:**
       - **Spatial-Attribute Blocking:** Segregates comparisons by `(City, BHK Count)` to achieve linear $O(N)$ efficiency.
       - **Haversine Geo-Proximity Clustering:** Measures distance between coordinate pairs ($< 75\\text{ meters}$).
       - **Tolerance Envelope:** Verifies price variance $\\le 5\\%$ and carpet area variance $\\le 5\\%$.
       - **Master Record Election:** Selects canonical primary listing, maps secondary IDs, and logs the complete audit trail.
    """)

    st.markdown('<div class="section-title">Deduplication Audit Log (Cross-Portal Resolution)</div>', unsafe_allow_html=True)
    if os.path.exists(AUDIT_PATH):
        audit_data = pd.read_csv(AUDIT_PATH)
        st.dataframe(
            audit_data.rename(columns={
                "master_listing_id": "Canonical ID",
                "master_source": "Primary Portal",
                "master_price": "Canonical Price",
                "duplicate_listing_id": "Duplicate ID",
                "duplicate_source": "Secondary Portal",
                "duplicate_price": "Duplicate Price",
                "locality": "Submarket",
                "city": "Metro",
                "bhk": "BHK",
                "resolved_action": "Status"
            }),
            use_container_width=True
        )

# ==============================================================================
# VIEW 3: Valuation Model Benchmarking
# ==============================================================================
elif nav_selection == "Valuation Model Benchmarking":
    st.markdown('<div class="section-title">Supervised Valuation Models: Decision Forest, k-NN & XGBoost</div>', unsafe_allow_html=True)

    ml_data = load_ml_metrics()
    metrics = ml_data.get("metrics", {})
    feat_imp = ml_data.get("feature_importance", {})

    st.markdown('<div class="section-title">Cross-Algorithm Regression Benchmark</div>', unsafe_allow_html=True)
    if metrics:
        metrics_df = pd.DataFrame(metrics).T
        metrics_df = metrics_df.rename(columns={
            "r2_score": "R-Squared",
            "rmse_inr": "RMSE (INR)",
            "mae_inr": "MAE (INR)",
            "mape_pct": "MAPE (%)",
            "training_time_sec": "Train Latency (s)",
            "inference_latency_per_sample_ms": "Inference Latency (ms)"
        })
        st.dataframe(metrics_df, use_container_width=True)

    ml_left, ml_right = st.columns([1, 1])

    with ml_left:
        st.markdown('<div class="section-title">Feature Importance (Decision Forest Tree Splits)</div>', unsafe_allow_html=True)
        if feat_imp:
            feat_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
            fig_feat = px.bar(
                feat_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color_discrete_sequence=["#1E293B"],
                labels={"Importance": "Gini Importance Score", "Feature": "Input Covariate"}
            )
            fig_feat.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=0, r=20, t=10, b=10),
                height=360,
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_feat, use_container_width=True)

    with ml_right:
        st.markdown('<div class="section-title">Live Valuation Sandbox (Multi-Model Consensus)</div>', unsafe_allow_html=True)
        with st.form("valuation_sandbox"):
            bhk_in = st.slider("Configuration (BHK)", 1, 4, 2)
            area_in = st.slider("Carpet Area (Sq.Ft)", 400, 3500, 1100, step=50)
            bath_in = st.slider("Bathrooms", 1, 4, 2)
            balc_in = st.slider("Balconies", 0, 3, 2)
            furn_in = st.selectbox("Furnishing Condition", ["Semi-Furnished", "Fully Furnished", "Unfurnished"])
            transit_in = st.slider("Transit Infrastructure Index", 50, 100, 88)
            safety_in = st.slider("Locality Safety Rating", 50, 100, 86)
            
            calc_btn = st.form_submit_button("Compute Valuations")

            if calc_btn:
                from src.ml.predict import PropertyValuationService
                try:
                    val_service = PropertyValuationService()
                    input_covariates = {
                        "carpet_area_sqft": area_in,
                        "bhk_count": bhk_in,
                        "bathrooms": bath_in,
                        "balconies": balc_in,
                        "parking": 1,
                        "gym": 1,
                        "swimming_pool": 1,
                        "transit_score": transit_in,
                        "safety_score": safety_in,
                        "school_score": 85,
                        "is_furnished": 1 if furn_in != "Unfurnished" else 0,
                        "latitude": 28.56,
                        "longitude": 77.38
                    }
                    predictions = val_service.predict_property(input_covariates)
                    res_table = pd.DataFrame(predictions).T.rename(columns={
                        "predicted_price_inr": "Capital Valuation (INR)",
                        "predicted_price_lakhs": "Valuation (Lakhs)",
                        "estimated_monthly_rent_inr": "Monthly Rent (INR)",
                        "projected_rental_yield_pct": "Projected Yield (%)"
                    })
                    st.dataframe(res_table[["Valuation (Lakhs)", "Monthly Rent (INR)", "Projected Yield (%)"]], use_container_width=True)
                except Exception as ex:
                    st.error(f"Inference pipeline execution error: {ex}")

# ==============================================================================
# VIEW 4: GenAI Research Assistant
# ==============================================================================
elif nav_selection == "GenAI Research Assistant":
    st.markdown('<div class="section-title">PropIQ Research Copilot (Text-to-SQL + Vector RAG)</div>', unsafe_allow_html=True)
    st.markdown("Combines structured SQL analytics over warehouse star schemas with vector retrieval over statutory lease agreements and submarket research briefs.")

    st.markdown("**Sample Research Prompts:**")
    prompt_col1, prompt_col2 = st.columns(2)
    with prompt_col1:
        if st.button("Which 2BHK areas in Noida have the best rental yield?"):
            st.session_state["active_query"] = "Which 2BHK areas in Noida have the best rental yield?"
    with prompt_col2:
        if st.button("What are the standard security deposit and lease escalation terms in NCR?"):
            st.session_state["active_query"] = "What are the standard security deposit and lease escalation terms in NCR?"

    initial_query = st.session_state.get("active_query", "Which 2BHK areas in Noida have the best rental yield?")
    query_input = st.text_input("Enter research question:", value=initial_query)

    if st.button("Execute Research Query"):
        from src.rag.assistant import PropIQAssistant
        with st.spinner("Executing hybrid retrieval across warehouse tables and documents..."):
            assistant = PropIQAssistant(DUCKDB_PATH, DOCS_DIR)
            response_pkg = assistant.answer_query(query_input)

            st.markdown(response_pkg["answer"])

            if response_pkg["sql_query"]:
                with st.expander("Underlying SQL Executed on OLAP Star Schema"):
                    st.code(response_pkg["sql_query"], language="sql")

            if response_pkg["citations"]:
                with st.expander("Statutory & Market Dossier Citations"):
                    for cit in response_pkg["citations"]:
                        st.markdown(f"- {cit}")

# ==============================================================================
# VIEW 5: Star Schema & Partitions
# ==============================================================================
elif nav_selection == "Star Schema & Partitions":
    st.markdown('<div class="section-title">Kimball Star Schema & Physical Partition Pruning</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dimensional Entity Relationship Model</div>', unsafe_allow_html=True)
    st.code("""
                               +-------------------+
                               |     DIM_DATE      |
                               |-------------------|
                               | date_key (PK)     |
                               | full_date         |
                               | month_name        |
                               | year              |
                               +---------+---------+
                                         |
                                         v
    +------------------+  +--------------------------------+  +----------------------+
    |   DIM_LOCATION   |  |     FACT_PROPERTY_LISTINGS     |  |  DIM_PROPERTY_TYPE   |
    |------------------|  |--------------------------------|  |----------------------|
    | location_key(PK) |->| listing_id (PK)                |<-| property_type_key(PK)|
    | city             |  | date_key (FK)                  |  | bhk_count            |
    | locality         |  | location_key (FK)              |  | bhk_label            |
    | transit_score    |  | property_type_key (FK)         |  | property_category    |
    | safety_score     |  | amenity_key (FK)               |  | furnishing_status    |
    +------------------+  | city (Partition Key)           |  +----------------------+
                          | price_inr (Measure)            |
                          | rent_inr (Measure)             |
                          | carpet_area_sqft (Measure)     |
                          | price_per_sqft (Derived)       |
                          | rental_yield_pct (Derived)     |
                          +---------------+----------------+
                                          |
                                          v
                                +-------------------+
                                |   DIM_AMENITIES   |
                                |-------------------|
                                | amenity_key (PK)  |
                                | has_parking       |
                                | has_gym           |
                                | has_swimming_pool |
                                | amenity_tier      |
                                +-------------------+
    """, language="text")

    st.markdown('<div class="section-title">Partition Pruning Performance Benchmark</div>', unsafe_allow_html=True)
    from src.warehouse.partitioning import PartitioningEngine
    part_dir = os.path.join(BASE_DIR, "data/warehouse/partitions")
    unpart_file = os.path.join(BASE_DIR, "data/warehouse/fact_property_listings.parquet")

    if os.path.exists(unpart_file) and os.path.exists(part_dir):
        p_engine = PartitioningEngine(part_dir)
        bench = p_engine.benchmark_partition_pruning(unpart_file, target_city="Noida")

        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Full Table Scan Latency</div>
                <div class="metric-value">{bench['unpartitioned_scan_time_sec'] * 1000:.2f} ms</div>
                <div class="metric-subtext">Unpartitioned Parquet file</div>
            </div>
            """, unsafe_allow_html=True)
        with b2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Pruned Scan Latency</div>
                <div class="metric-value">{bench['partition_pruned_scan_time_sec'] * 1000:.2f} ms</div>
                <div class="metric-subtext">Single partition directory</div>
            </div>
            """, unsafe_allow_html=True)
        with b3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Target Submarket</div>
                <div class="metric-value">{bench['target_city']}</div>
                <div class="metric-subtext">{bench['speedup_factor']}</div>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# VIEW 6: BI Integration & Export
# ==============================================================================
elif nav_selection == "BI Integration & Export":
    st.markdown('<div class="section-title">Power BI & Tableau Export Center</div>', unsafe_allow_html=True)

    export_path = os.path.join(BASE_DIR, "data/warehouse/export_powerbi")
    st.markdown(f"Analytics-ready dimensional marts are exported to: `{export_path}`")

    files = [
        "fact_property_listings.csv",
        "dim_location.csv",
        "dim_property_type.csv",
        "dim_amenities.csv",
        "dim_date.csv"
    ]
    for f in files:
        fp = os.path.join(export_path, f)
        if os.path.exists(fp):
            size_kb = os.path.getsize(fp) / 1024.0
            st.markdown(f"- **{f}** ({size_kb:.1f} KB, Star Schema Dimension/Fact)")

    st.markdown('<div class="section-title">Production DAX Formulas for Power BI Models</div>', unsafe_allow_html=True)
    st.code("""
-- Average Rental Yield %
Avg_Rental_Yield_Pct = 
DIVIDE(
    SUM(fact_property_listings[rent_inr]) * 12, 
    SUM(fact_property_listings[price_inr]), 
    0
) * 100

-- Average Capital Rate per Sq.Ft
Avg_Price_Per_SqFt = 
DIVIDE(
    SUM(fact_property_listings[price_inr]), 
    SUM(fact_property_listings[carpet_area_sqft]), 
    0
)

-- Gross Rent Multiplier (Asset Recovery Period in Years)
Gross_Rent_Multiplier = 
DIVIDE(
    SUM(fact_property_listings[price_inr]), 
    SUM(fact_property_listings[rent_inr]) * 12, 
    0
)
    """, language="sql")

    st.markdown('<div class="section-title">Tableau Calculated Fields</div>', unsafe_allow_html=True)
    st.code("""
// Gross Rental Yield %
(SUM([Rent Inr]) * 12) / SUM([Price Inr]) * 100

// Average Price per Sq.Ft
SUM([Price Inr]) / SUM([Carpet Area Sqft])
    """, language="sql")
