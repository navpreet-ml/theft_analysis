import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. INSTITUTIONAL UI CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Strategic Risk & Capital Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional "FinTech" CSS
st.markdown("""
<style>
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #262730;
    }

    /* Metrics Cards */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 20px;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 700;
        color: #111;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
        border-bottom: 1px solid #ddd;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        font-size: 16px;
        font-weight: 600;
        color: #555;
        border-radius: 0;
        background-color: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #1a73e8; /* Professional Blue */
        border-bottom: 3px solid #1a73e8;
    }

    /* Remove Clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# COLOR PALETTE (Institutional)
CP_NAVY = "#0d47a1"
CP_BLUE = "#1976d2"
CP_SLATE = "#546e7a"
CP_RED = "#d32f2f"
CP_GREEN = "#388e3c"
CP_GREY = "#f5f5f5"


# ==========================================
# 2. DATA GENERATION ENGINE (Internal)
# ==========================================
@st.cache_data
def load_data():
    """Generates Granular Claims, FSA Aggregates, and Vehicle Aggregates."""

    # --- A. Setup ---
    np.random.seed(42)
    n_rows = 30000
    ontario_fsas = ['M5V', 'L4T', 'K1A', 'N2L', 'L5N', 'M1B', 'L6P', 'H3Z', 'L4W', 'M9V']
    vehicles = [('Toyota', 'RAV4'), ('Honda', 'CR-V'), ('Lexus', 'RX350'), ('Ford', 'F-150'),
                ('Dodge', 'Ram 1500'), ('Toyota', 'Highlander'), ('Honda', 'Civic'), ('Hyundai', 'Elantra')]
    veh_codes = [f"{m} {model}" for m, model in vehicles]

    # --- B. Granular Claims Data ---
    df = pd.DataFrame({
        'garaging_fsa': np.random.choice(ontario_fsas, n_rows),
        'vehicle_code': np.random.choice(veh_codes, n_rows),
        'exposure': np.random.uniform(0.1, 1.0, n_rows),
        'premium': np.random.uniform(800, 3500, n_rows),
    })

    # Risk Logic (Theft)
    base_prob = 0.015
    df['is_stolen'] = np.random.random(n_rows) < base_prob
    # Make L4T and Lexus Risky
    risky_mask = (df['garaging_fsa'] == 'L4T') | (df['vehicle_code'] == 'Lexus RX350')
    df.loc[risky_mask, 'is_stolen'] = np.random.random(risky_mask.sum()) < 0.05

    df['incurred'] = 0.0
    df.loc[df['is_stolen'], 'incurred'] = np.random.uniform(5000, 85000, df['is_stolen'].sum())

    # --- C. Market Data Generation (Derived) ---
    # 1. FSA Level
    fsa_agg = df.groupby('garaging_fsa')['exposure'].sum().reset_index()
    fsa_agg.rename(columns={'exposure': 'Company_Exposure'}, inplace=True)
    # Simulate Market: Company has ~5% share, with variance
    fsa_agg['Market_Vehicles'] = (fsa_agg['Company_Exposure'] / np.random.uniform(0.02, 0.08, len(fsa_agg))).astype(int)

    # 2. Vehicle Level
    veh_agg = df.groupby('vehicle_code')['exposure'].sum().reset_index()
    veh_agg.rename(columns={'exposure': 'Company_Exposure'}, inplace=True)
    veh_agg['Market_Vehicles'] = (veh_agg['Company_Exposure'] / np.random.uniform(0.02, 0.08, len(veh_agg))).astype(int)

    return df, fsa_agg, veh_agg


df_claims, df_fsa, df_veh = load_data()

# ==========================================
# 3. ADVANCED ANALYTICS LOGIC
# ==========================================

# Enrich FSA Data with Claims Info
fsa_risk = df_claims.groupby('garaging_fsa').agg(
    Theft_Count=('is_stolen', 'sum'),
    Theft_Loss=('incurred', 'sum'),
    Company_Exp_Actual=('exposure', 'sum')
).reset_index()

# Merge with Market Data
master_fsa = pd.merge(fsa_risk, df_fsa, left_on='garaging_fsa', right_on='garaging_fsa')

# METRICS CALCULATION
master_fsa['Theft_Freq'] = master_fsa['Theft_Count'] / master_fsa['Company_Exp_Actual']
master_fsa['Avg_Severity'] = master_fsa['Theft_Loss'] / master_fsa['Theft_Count']
master_fsa['Penetration_Rate'] = (master_fsa['Company_Exp_Actual'] / master_fsa['Market_Vehicles']) * 100

# QUADRANT CLASSIFICATION
# "Toxic": High Theft Freq + High Penetration
# "Growth": Low Theft Freq + Low Penetration
# "Watch": High Theft Freq + Low Penetration
# "Safe Cow": Low Theft Freq + High Penetration

avg_freq = master_fsa['Theft_Freq'].mean()
avg_pen = master_fsa['Penetration_Rate'].mean()


def classify_fsa(row):
    if row['Theft_Freq'] > avg_freq and row['Penetration_Rate'] > avg_pen:
        return "Toxic Concentration"
    elif row['Theft_Freq'] < avg_freq and row['Penetration_Rate'] < avg_pen:
        return "Growth Target"
    elif row['Theft_Freq'] > avg_freq:
        return "Avoidance Zone"
    else:
        return "Safe Stronghold"


master_fsa['Strategy_Zone'] = master_fsa.apply(classify_fsa, axis=1)

# ==========================================
# 4. DASHBOARD UI
# ==========================================

st.title("🛡️ Strategic Risk & Capital Allocation Platform")
st.markdown(f"**Reporting Period:** Q4 2025 | **Entity:** Ontario Auto Portfolio")

# --- GLOBAL KPI ROW ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total Market Exposure", f"{df_fsa['Market_Vehicles'].sum():,}", "Vehicles in Market")
with kpi2:
    global_pen = (df_claims['exposure'].sum() / df_fsa['Market_Vehicles'].sum()) * 100
    st.metric("Global Market Share", f"{global_pen:.2f}%", "Of Total ON Market")
with kpi3:
    loss_ratio = (df_claims['incurred'].sum() / df_claims['premium'].sum()) * 100
    st.metric("Portfolio Loss Ratio", f"{loss_ratio:.1f}%", "-1.2% vs Plan")
with kpi4:
    st.metric("Theft Frequency", f"{(df_claims['is_stolen'].sum() / df_claims['exposure'].sum()) * 100:.2f}%",
              "Claims / Exposure")

st.markdown("---")

# --- TABS ---
tab_strat, tab_map, tab_veh, tab_rsp = st.tabs([
    "📈 Strategic Positioning",
    "🗺️ Geospatial Intelligence",
    "🚗 Asset Selection",
    "🧪 RSP Optimization"
])

# --- TAB 1: STRATEGIC POSITIONING (The Creative Part) ---
with tab_strat:
    st.subheader("Market Positioning Matrix")
    st.markdown("""
    This matrix identifies where our portfolio is **over-exposed to risk** relative to our market share.
    * **Toxic Concentration (Top-Right):** We own too much of a bad neighborhood. **Action:** Increase Rates / Cede to RSP.
    * **Growth Target (Bottom-Left):** Safe areas where we are underweight. **Action:** Marketing Spend.
    """)

    col_chart, col_detail = st.columns([3, 1])

    with col_chart:
        fig_strat = px.scatter(
            master_fsa,
            x="Theft_Freq",
            y="Penetration_Rate",
            size="Market_Vehicles",
            color="Strategy_Zone",
            text="garaging_fsa",
            hover_data=["Theft_Loss", "Company_Exp_Actual"],
            color_discrete_map={
                "Toxic Concentration": CP_RED,
                "Safe Stronghold": CP_GREEN,
                "Growth Target": CP_BLUE,
                "Avoidance Zone": "orange"
            },
            title="Strategic Alignment: Risk vs. Market Share"
        )

        # Add quadrants
        fig_strat.add_hline(y=avg_pen, line_dash="dot", annotation_text="Avg Share")
        fig_strat.add_vline(x=avg_freq, line_dash="dot", annotation_text="Avg Risk")
        fig_strat.update_layout(template="plotly_white", height=600)
        st.plotly_chart(fig_strat, use_container_width=True)

    with col_detail:
        st.markdown("**Priority Action Items**")

        toxic = master_fsa[master_fsa['Strategy_Zone'] == "Toxic Concentration"].sort_values('Theft_Loss',
                                                                                             ascending=False)
        st.error(f"🚨 **De-Risk Needed:** {len(toxic)} FSAs")
        if not toxic.empty:
            st.dataframe(toxic[['garaging_fsa', 'Penetration_Rate', 'Theft_Loss']], hide_index=True)

        growth = master_fsa[master_fsa['Strategy_Zone'] == "Growth Target"].sort_values('Market_Vehicles',
                                                                                        ascending=False)
        st.info(f"🚀 **Growth Targets:** {len(growth)} FSAs")
        if not growth.empty:
            st.dataframe(growth[['garaging_fsa', 'Market_Vehicles', 'Theft_Freq']], hide_index=True)

# --- TAB 2: GEOSPATIAL INTELLIGENCE ---
with tab_map:
    st.subheader("Penetration & Risk Heatmap")

    # Toggle Map Type
    map_mode = st.radio("Select Layer:", ["Company Theft Risk", "Market Penetration Opportunity"], horizontal=True)

    # NOTE: In a real app, you would join this with a GeoJSON polygon file.
    # Here we simulate lat/lon for the bubble map.
    fsa_coords = {
        'M5V': [43.6426, -79.3871], 'L4T': [43.7081, -79.6291], 'K1A': [45.4215, -75.6972],
        'N2L': [43.4643, -80.5204], 'L5N': [43.5903, -79.7625], 'M1B': [43.8093, -79.2216],
        'L6P': [43.7788, -79.7183], 'H3Z': [45.4856, -73.5964], 'L4W': [43.6333, -79.6167],
        'M9V': [43.7431, -79.5858]
    }
    master_fsa['lat'] = master_fsa['garaging_fsa'].map(lambda x: fsa_coords.get(x, [0, 0])[0])
    master_fsa['lon'] = master_fsa['garaging_fsa'].map(lambda x: fsa_coords.get(x, [0, 0])[1])

    if map_mode == "Company Theft Risk":
        fig_map = px.scatter_mapbox(
            master_fsa, lat="lat", lon="lon", size="Theft_Loss", color="Theft_Freq",
            hover_name="garaging_fsa", zoom=6, mapbox_style="carto-positron",
            color_continuous_scale="Reds", title="High Loss Zones (Size = $ Loss, Color = Frequency)"
        )
    else:
        # Growth Map: High Market Size + Low Penetration
        fig_map = px.scatter_mapbox(
            master_fsa, lat="lat", lon="lon", size="Market_Vehicles", color="Penetration_Rate",
            hover_name="garaging_fsa", zoom=6, mapbox_style="carto-positron",
            color_continuous_scale="Tealgrn", title="Expansion Opportunities (Size = Market Volume, Color = Our Share)"
        )

    fig_map.update_layout(height=600, margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)

# --- TAB 3: ASSET SELECTION (Adverse Selection Analysis) ---
with tab_veh:
    st.subheader("Adverse Selection Analysis")
    st.markdown("""
    Are we attracting the **wrong** risks? 
    * **Adverse Selection:** When our share of a High-Theft Vehicle is *higher* than our average market share.
    """)

    # 1. Prepare Vehicle Data (Risk View)
    veh_risk = df_claims.groupby('vehicle_code').agg(
        Theft_Count=('is_stolen', 'sum'),
        Company_Exposure=('exposure', 'sum'),
        Loss=('incurred', 'sum')
    ).reset_index()

    # 2. Merge with Market Data
    # FIX: We only select ['vehicle_code', 'Market_Vehicles'] from df_veh
    # This prevents 'Company_Exposure' from appearing twice and becoming 'Company_Exposure_x'
    master_veh = pd.merge(veh_risk, df_veh[['vehicle_code', 'Market_Vehicles']], on='vehicle_code')

    # 3. Calculations
    master_veh['Penetration_Rate'] = (master_veh['Company_Exposure'] / master_veh['Market_Vehicles']) * 100
    master_veh['Theft_Freq'] = master_veh['Theft_Count'] / master_veh['Company_Exposure']

    # Calculate "Adverse Selection Index"
    # ASI = (This Vehicle Penetration) / (Portfolio Average Penetration)
    portfolio_avg_pen = master_veh['Penetration_Rate'].mean()
    master_veh['ASI'] = master_veh['Penetration_Rate'] / portfolio_avg_pen

    # 4. Visualization
    fig_asi = px.bar(
        master_veh.sort_values('ASI', ascending=False),
        x='vehicle_code',
        y='ASI',
        color='Theft_Freq',
        color_continuous_scale='Reds',
        title="Adverse Selection Index (Bars > 1.0 = Overweight vs. Avg Market Share)",
        hover_data=['Market_Vehicles', 'Company_Exposure']
    )
    # Add Threshold Line
    fig_asi.add_hline(y=1.0, line_color="black", line_dash="dash", annotation_text="Portfolio Avg")
    fig_asi.update_layout(template="plotly_white")
    st.plotly_chart(fig_asi, use_container_width=True)

# --- TAB 4: RSP OPTIMIZATION ---
with tab_rsp:
    st.subheader("RSP Transfer Simulator")

    col_ctrl, col_res = st.columns([1, 3])

    with col_ctrl:
        st.markdown("### Transfer Rules")
        with st.form("rsp_form"):
            min_asi = st.slider("Transfer if ASI > X", 0.5, 2.0, 1.2, help="Adverse Selection Index Threshold")
            min_theft = st.slider("Transfer if Theft Freq > X%", 0.0, 5.0, 2.0) / 100
            btn_run = st.form_submit_button("Simulate Transfer")

    with col_res:
        if btn_run:
            # Logic: Flag vehicles that meet criteria
            target_vehs = master_veh[
                (master_veh['ASI'] > min_asi) &
                (master_veh['Theft_Freq'] > min_theft)
                ]['vehicle_code'].tolist()

            # Calculate Savings
            mask = df_claims['vehicle_code'].isin(target_vehs)
            ceded_prem = df_claims.loc[mask, 'premium'].sum()
            ceded_loss = df_claims.loc[mask, 'incurred'].sum()
            net_benefit = ceded_loss - ceded_prem

            st.markdown(f"**Simulation Results for: {len(target_vehs)} Vehicle Models**")

            m1, m2, m3 = st.columns(3)
            m1.metric("Ceded Premium", f"${ceded_prem:,.0f}")
            m2.metric("Avoided Losses", f"${ceded_loss:,.0f}")
            m3.metric("Net Savings", f"${net_benefit:,.0f}", delta_color="normal")

            st.markdown("#### Impacted Vehicles")
            st.table(
                master_veh[master_veh['vehicle_code'].isin(target_vehs)][['vehicle_code', 'ASI', 'Theft_Freq', 'Loss']])