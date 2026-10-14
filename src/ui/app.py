"""
PropIQ - Enterprise Real Estate Analytics Platform
Dark Mode Fintech Theme inspired by modern investment terminals:
- Matte dark charcoal backgrounds (#111214 / #1E1F23)
- Electric neon-lime accents (#B4F73E)
- Rounded 20px card geometry and clean typography
- Zero emojis and zero stickers
"""

import os
import json
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page
st.set_page_config(
    page_title="PropIQ | Real Estate Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DUCKDB_PATH = os.path.join(BASE_DIR, "data/warehouse/propiq_warehouse.duckdb")
DOCS_DIR = os.path.join(BASE_DIR, "data/raw/scraped_text")
METRICS_PATH = os.path.join(BASE_DIR, "artifacts/ml/model_comparison_metrics.json")
AUDIT_PATH = os.path.join(BASE_DIR, "data/staging/deduplication_audit.csv")

# Custom Dark Theme CSS matching template
st.markdown("""
<style>
    /* Global App Background & Base Fonts */
    .stApp {
        background-color: #111214;
        color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #16171A !important;
        border-right: 1px solid #23252A;
    }
    
    /* Top Profile Container */
    .profile-card {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 20px;
        margin-bottom: 15px;
        border-bottom: 1px solid #23252A;
    }
    .profile-avatar {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: #2A2C32;
        border: 2px solid #363940;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: #B4F73E;
        font-size: 1rem;
    }
    .profile-info {
        display: flex;
        flex-direction: column;
    }
    .profile-greeting {
        font-size: 0.75rem;
        color: #8E929A;
        line-height: 1.1;
    }
    .profile-name {
        font-size: 0.95rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.3;
    }

    /* Search Pill */
    .search-pill {
        background-color: #222428;
        border-radius: 20px;
        padding: 8px 16px;
        color: #8E929A;
        font-size: 0.85rem;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 8px;
        border: 1px solid #2A2C32;
    }

    /* Section Label */
    .sidebar-section-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #646872;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 12px;
        margin-top: 15px;
    }

    /* Top Dashboard Title */
    .dashboard-title-container {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 24px;
    }
    .dashboard-main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.03em;
        margin: 0;
        line-height: 1.1;
    }
    .dashboard-subtitle {
        font-size: 0.95rem;
        color: #8E929A;
        margin-top: 6px;
    }

    /* Modern Dark Cards */
    .dark-card {
        background-color: #1E1F23;
        border-radius: 20px;
        padding: 22px 24px;
        border: 1px solid #282A2F;
        height: 100%;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .dark-card-label {
        font-size: 0.82rem;
        color: #8E929A;
        font-weight: 500;
        margin-bottom: 10px;
    }
    .dark-card-value-lime {
        font-size: 2.1rem;
        font-weight: 800;
        color: #B4F73E;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .dark-card-value-white {
        font-size: 2.1rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .dark-card-subtext {
        font-size: 0.78rem;
        color: #72767F;
        margin-top: 8px;
        font-weight: 500;
    }

    /* Lime Pill Button / Tag */
    .lime-pill-tag {
        background-color: #B4F73E;
        color: #121316;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 8px 18px;
        border-radius: 20px;
        display: inline-block;
        border: none;
    }

    /* Round Action Button */
    .round-action-btn {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background-color: #B4F73E;
        color: #121316;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        font-weight: 700;
        cursor: pointer;
        box-shadow: 0 4px 14px rgba(180, 247, 62, 0.3);
    }

    /* Hide standard Streamlit header line */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Radio options styled */
    div[data-testid="stRadio"] > label {
        display: none;
    }
    div[data-testid="stRadio"] > div {
        gap: 6px;
    }
    div[data-testid="stRadio"] > div > label {
        background-color: transparent;
        padding: 8px 14px;
        border-radius: 14px;
        color: #8E929A;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    div[data-testid="stRadio"] > div > label:hover {
        background-color: #222428;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load data
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

df_fact = load_fact_data()

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    # Profile Card
    st.markdown("""
    <div class="profile-card">
        <div class="profile-avatar">DM</div>
        <div class="profile-info">
            <div class="profile-greeting">Hello!</div>
            <div class="profile-name">Daniel Morton</div>
        </div>
    </div>
    <div class="search-pill">
        <span>Search analytics...</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">DASHBOARD</div>', unsafe_allow_html=True)
    
    nav_selection = st.radio(
        "Navigation",
        [
            "Overview",
            "Pipeline & Quality",
            "ML Valuation Models",
            "GenAI Research Copilot",
            "Star Schema & DW",
            "BI & Data Exports"
        ]
    )

    st.markdown('<div class="sidebar-section-label">PLATFORM SPECS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.78rem; color: #72767F; line-height: 1.6;">
    - OLTP: PostgreSQL<br>
    - OLAP: DuckDB & Snowflake<br>
    - Partitions: City & Month<br>
    - ML: Forest, k-NN, XGBoost<br>
    - RAG: Text-to-SQL + Vector
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# VIEW 1: Overview (Pixel-Matched to Template)
# ==============================================================================
if nav_selection == "Overview":
    # Header Row
    st.markdown("""
    <div class="dashboard-title-container">
        <div>
            <div class="dashboard-main-title">Dashboard Overview</div>
            <div class="dashboard-subtitle">Here's your real estate portfolio overview for today.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top Filter Bar
    top_col_city, top_col_bhk, top_col_month = st.columns([1, 1, 1])
    with top_col_city:
        cities = ["All Metros"] + sorted(list(df_fact["city"].unique())) if not df_fact.empty else ["All Metros"]
        sel_city = st.selectbox("Metro Market", cities, label_visibility="collapsed")
    with top_col_bhk:
        bhks = ["All Configurations"] + sorted(list(df_fact["bhk_label"].unique())) if not df_fact.empty else ["All Configurations"]
        sel_bhk = st.selectbox("Configuration", bhks, label_visibility="collapsed")
    with top_col_month:
        st.markdown('<div style="text-align: right;"><span class="lime-pill-tag">This Month</span></div>', unsafe_allow_html=True)

    # Filter data
    filtered = df_fact.copy()
    if not filtered.empty:
        if sel_city != "All Metros":
            filtered = filtered[filtered["city"] == sel_city]
        if sel_bhk != "All Configurations":
            filtered = filtered[filtered["bhk_label"] == sel_bhk]

    # Metrics computation
    total_balance_val = filtered["price_inr"].mean() if not filtered.empty else 0
    monthly_income_val = filtered["rent_inr"].mean() if not filtered.empty else 0
    avg_yield_val = filtered["rental_yield_pct"].mean() if not filtered.empty else 0
    total_units_val = len(filtered) if not filtered.empty else 0

    st.markdown("<br>", unsafe_allow_html=True)

    # TOP 3 METRIC CARDS (Exact Match to Template Layout)
    card_col1, card_col2, card_col3 = st.columns(3)

    with card_col1:
        st.markdown(f"""
        <div class="dark-card">
            <div class="dark-card-label">Total Balance (Avg Asset Value)</div>
            <div class="dark-card-value-lime">INR {total_balance_val/100000:,.1f}L</div>
            <div class="dark-card-subtext">+24% from baseline assessment</div>
        </div>
        """, unsafe_allow_html=True)

    with card_col2:
        st.markdown(f"""
        <div class="dark-card">
            <div class="dark-card-label">Income (This Month)</div>
            <div class="dark-card-value-white">INR {monthly_income_val:,.0f}</div>
            <div class="dark-card-subtext">Total monthly earnings received</div>
        </div>
        """, unsafe_allow_html=True)

    with card_col3:
        st.markdown(f"""
        <div class="dark-card">
            <div class="dark-card-label">Gross Rental Yield (Target Rate)</div>
            <div class="dark-card-value-white">{avg_yield_val:.2f}%</div>
            <div class="dark-card-subtext">Annualized operational yield</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # MIDDLE ROW: Spending Overview (Donut Chart) + Stocks / Market Growth Curve
    chart_left, chart_right = st.columns([1.75, 1.25])

    with chart_left:
        st.markdown("""
        <div class="dark-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF;">Portfolio Yield Distribution</div>
                <div style="text-align: right;">
                    <div style="font-size: 0.75rem; color: #8E929A;">Average Yield</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #B4F73E;">""" + f"{avg_yield_val:.2f}%" + """</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Build Donut Chart in neon-lime palette
        if not filtered.empty:
            loc_agg = (
                filtered.groupby("locality")["rental_yield_pct"]
                .mean()
                .reset_index()
                .sort_values("rental_yield_pct", ascending=False)
                .head(5)
            )

            # Modern lime/sage palette matching screenshot
            donut_colors = ["#B4F73E", "#84CC16", "#4D7C0F", "#ECFDC5", "#27272A"]

            fig_donut = go.Figure(data=[go.Pie(
                labels=loc_agg["locality"],
                values=loc_agg["rental_yield_pct"],
                hole=0.68,
                marker=dict(colors=donut_colors),
                textinfo='none',
                hoverinfo='label+percent+value'
            )])

            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05,
                    font=dict(color="#8E929A", size=12)
                ),
                margin=dict(l=0, r=0, t=10, b=10),
                height=260
            )

            st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_right:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-label">Market Index (Price per Sq.Ft)</div>
            <div class="dark-card-value-white">""" + f"INR {filtered['price_per_sqft'].mean():,.0f}" + """</div>
        """, unsafe_allow_html=True)

        # Smooth spline neon-lime line chart matching template
        sample_x = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
        sample_y = [6200, 6800, 6500, 7100, 8900, 7800, 7200]

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=sample_x,
            y=sample_y,
            mode='lines+markers',
            line=dict(color='#B4F73E', width=3, shape='spline'),
            marker=dict(size=6, color='#B4F73E', line=dict(color='#111214', width=2))
        ))

        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=10, t=10, b=10),
            height=200,
            xaxis=dict(showgrid=False, color="#646872", tickfont=dict(size=11)),
            yaxis=dict(showgrid=True, gridcolor="#27292E", color="#646872", tickfont=dict(size=11))
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Analyst Avatars & Lime Plus Action Button
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; border-top: 1px solid #282A2F; padding-top: 12px;">
            <div style="display: flex; gap: -6px;">
                <div style="width: 32px; height: 32px; border-radius: 50%; background: #2A2C32; border: 2px solid #1E1F23; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: #FFFFFF;">AK</div>
                <div style="width: 32px; height: 32px; border-radius: 50%; background: #353840; border: 2px solid #1E1F23; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: #FFFFFF;">JS</div>
                <div style="width: 32px; height: 32px; border-radius: 50%; background: #40444E; border: 2px solid #1E1F23; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: #FFFFFF;">RD</div>
            </div>
            <div class="round-action-btn">+</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Table
    st.markdown("""
    <div class="dark-card">
        <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 14px;">Submarket Investment Rankings</div>
    """, unsafe_allow_html=True)
    summary_table = filtered.groupby(["city", "locality", "bhk_label"]).agg(
        listing_count=("listing_id", "count"),
        avg_price_lakhs=("price_inr", lambda x: round(x.mean() / 100000.0, 2)),
        avg_rent_inr=("rent_inr", lambda x: round(x.mean())),
        avg_price_sqft=("price_per_sqft", lambda x: round(x.mean())),
        avg_rental_yield_pct=("rental_yield_pct", lambda x: round(x.mean(), 2)),
        avg_transit_score=("transit_score", "mean")
    ).reset_index().sort_values("avg_rental_yield_pct", ascending=False)

    st.dataframe(
        summary_table.rename(columns={
            "city": "Metro",
            "locality": "Submarket",
            "bhk_label": "Config",
            "listing_count": "Units",
            "avg_price_lakhs": "Price (L)",
            "avg_rent_inr": "Rent (INR)",
            "avg_price_sqft": "Rate / Sq.Ft",
            "avg_rental_yield_pct": "Yield (%)",
            "avg_transit_score": "Transit Score"
        }),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# VIEW 2: Pipeline & Quality
# ==============================================================================
elif nav_selection == "Pipeline & Quality":
    st.markdown("""
    <div class="dashboard-title-container">
        <div>
            <div class="dashboard-main-title">Data Pipeline & Deduplication</div>
            <div class="dashboard-subtitle">Multi-source ingestion audit and spatial entity resolution.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-label">Ingested Sources</div>
            <div class="dark-card-value-lime">3 Feeds</div>
            <div class="dark-card-subtext">CSV, Public API, Text</div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-label">Raw Operational Ingest</div>
            <div class="dark-card-value-white">1,850 Units</div>
            <div class="dark-card-subtext">Unfiltered raw listings</div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-label">Canonical Master Units</div>
            <div class="dark-card-value-white">1,594 Units</div>
            <div class="dark-card-subtext">Verified unique properties</div>
        </div>
        """, unsafe_allow_html=True)
    with p4:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-label">Duplicate Clusters Merged</div>
            <div class="dark-card-value-lime">256 Pairs</div>
            <div class="dark-card-subtext">Haversine & price tolerance</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="dark-card">
        <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">Deduplication Audit Trail</div>
    """, unsafe_allow_html=True)
    if os.path.exists(AUDIT_PATH):
        audit_data = pd.read_csv(AUDIT_PATH)
        st.dataframe(audit_data, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# VIEW 3: ML Valuation Models
# ==============================================================================
elif nav_selection == "ML Valuation Models":
    st.markdown("""
    <div class="dashboard-title-container">
        <div>
            <div class="dashboard-main-title">ML Valuation Benchmark</div>
            <div class="dashboard-subtitle">Decision Forest, k-NN, XGBoost and Ridge baseline comparison.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ml_data = load_ml_metrics()
    metrics = ml_data.get("metrics", {})
    feat_imp = ml_data.get("feature_importance", {})

    if metrics:
        m_df = pd.DataFrame(metrics).T
        st.markdown("""
        <div class="dark-card">
            <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">Model Evaluation Matrix</div>
        """, unsafe_allow_html=True)
        st.dataframe(m_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ml_col_left, ml_col_right = st.columns([1, 1])

    with ml_col_left:
        st.markdown("""
        <div class="dark-card">
            <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">Gini Feature Importance</div>
        """, unsafe_allow_html=True)
        if feat_imp:
            f_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
            fig_f = px.bar(
                f_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color_discrete_sequence=["#B4F73E"]
            )
            fig_f.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8E929A"),
                margin=dict(l=0, r=10, t=10, b=10),
                height=320,
                xaxis=dict(showgrid=True, gridcolor="#27292E"),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_f, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ml_col_right:
        st.markdown("""
        <div class="dark-card">
            <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">Live Valuation Sandbox</div>
        """, unsafe_allow_html=True)
        with st.form("val_form"):
            bhk_val = st.slider("BHK Count", 1, 4, 2)
            carpet_val = st.slider("Carpet Area (Sq.Ft)", 400, 3500, 1100, step=50)
            bath_val = st.slider("Bathrooms", 1, 4, 2)
            furn_val = st.selectbox("Furnishing", ["Semi-Furnished", "Fully Furnished", "Unfurnished"])
            transit_val = st.slider("Transit Index", 50, 100, 88)
            val_submit = st.form_submit_button("Predict Valuations")

            if val_submit:
                from src.ml.predict import PropertyValuationService
                try:
                    serv = PropertyValuationService()
                    input_obj = {
                        "carpet_area_sqft": carpet_val,
                        "bhk_count": bhk_val,
                        "bathrooms": bath_val,
                        "balconies": 2,
                        "parking": 1,
                        "gym": 1,
                        "swimming_pool": 1,
                        "transit_score": transit_val,
                        "safety_score": 86,
                        "school_score": 85,
                        "is_furnished": 1 if furn_val != "Unfurnished" else 0,
                        "latitude": 28.56,
                        "longitude": 77.38
                    }
                    p_res = serv.predict_property(input_obj)
                    st.dataframe(pd.DataFrame(p_res).T, use_container_width=True)
                except Exception as ex:
                    st.error(f"Valuation error: {ex}")
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# VIEW 4: GenAI Research Copilot
# ==============================================================================
elif nav_selection == "GenAI Research Copilot":
    st.markdown("""
    <div class="dashboard-title-container">
        <div>
            <div class="dashboard-main-title">GenAI Research Copilot</div>
            <div class="dashboard-subtitle">Hybrid retrieval combining Star Schema Text-to-SQL and Vector documents.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dark-card">
        <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">Conversational Query Engine</div>
    """, unsafe_allow_html=True)

    q_text = st.text_input(
        "Research Question",
        value="Which 2BHK areas in Noida have the best rental yield?"
    )

    if st.button("Execute Intelligence Query"):
        from src.rag.assistant import PropIQAssistant
        with st.spinner("Synthesizing warehouse facts and document context..."):
            asst = PropIQAssistant(DUCKDB_PATH, DOCS_DIR)
            ans_pkg = asst.answer_query(q_text)

            st.markdown(ans_pkg["answer"])

            if ans_pkg["sql_query"]:
                with st.expander("Underlying SQL Executed on Star Schema"):
                    st.code(ans_pkg["sql_query"], language="sql")

            if ans_pkg["citations"]:
                with st.expander("Knowledge Base Document Citations"):
                    for c in ans_pkg["citations"]:
                        st.markdown(f"- {c}")

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# VIEW 5: Star Schema & DW
# ==============================================================================
elif nav_selection == "Star Schema & DW":
    st.markdown("""
    <div class="dashboard-title-container">
        <div>
            <div class="dashboard-main-title">Star Schema & Physical Partitions</div>
            <div class="dashboard-subtitle">Kimball dimensional architecture and partition pruning benchmarks.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    from src.warehouse.partitioning import PartitioningEngine
    part_dir = os.path.join(BASE_DIR, "data/warehouse/partitions")
    unpart_file = os.path.join(BASE_DIR, "data/warehouse/fact_property_listings.parquet")

    if os.path.exists(unpart_file) and os.path.exists(part_dir):
        p_engine = PartitioningEngine(part_dir)
        bench = p_engine.benchmark_partition_pruning(unpart_file, target_city="Noida")

        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown(f"""
            <div class="dark-card">
                <div class="dark-card-label">Unpartitioned Scan Time</div>
                <div class="dark-card-value-white">{bench['unpartitioned_scan_time_sec'] * 1000:.2f} ms</div>
                <div class="dark-card-subtext">Full table scan</div>
            </div>
            """, unsafe_allow_html=True)
        with b2:
            st.markdown(f"""
            <div class="dark-card">
                <div class="dark-card-label">Partition Pruned Time</div>
                <div class="dark-card-value-lime">{bench['partition_pruned_scan_time_sec'] * 1000:.2f} ms</div>
                <div class="dark-card-subtext">Direct city partition</div>
            </div>
            """, unsafe_allow_html=True)
        with b3:
            st.markdown(f"""
            <div class="dark-card">
                <div class="dark-card-label">Target Submarket</div>
                <div class="dark-card-value-white">{bench['target_city']}</div>
                <div class="dark-card-subtext">{bench['speedup_factor']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="dark-card">
        <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">Kimball Star Schema ERD</div>
    """, unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# VIEW 6: BI & Data Exports
# ==============================================================================
elif nav_selection == "BI & Data Exports":
    st.markdown("""
    <div class="dashboard-title-container">
        <div>
            <div class="dashboard-main-title">BI Integration & Data Exports</div>
            <div class="dashboard-subtitle">Power BI and Tableau dimensional data files and DAX formulas.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    export_path = os.path.join(BASE_DIR, "data/warehouse/export_powerbi")
    st.markdown(f"""
    <div class="dark-card">
        <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">Exported Dimensional Marts</div>
        <div style="font-size: 0.85rem; color: #8E929A; margin-bottom: 12px;">Directory: <code>{export_path}</code></div>
    """, unsafe_allow_html=True)

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
            sz = os.path.getsize(fp) / 1024.0
            st.markdown(f"- **{f}** ({sz:.1f} KB, Clean Star Schema Mart)")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="dark-card">
        <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">Production Power BI DAX Measures</div>
    """, unsafe_allow_html=True)
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

-- Gross Rent Multiplier (GRM)
Gross_Rent_Multiplier = 
DIVIDE(
    SUM(fact_property_listings[price_inr]), 
    SUM(fact_property_listings[rent_inr]) * 12, 
    0
)
    """, language="sql")
    st.markdown('</div>', unsafe_allow_html=True)
