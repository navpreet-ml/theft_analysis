import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# -------------------------------
# 1. PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Theft Claims & Exposure Dashboard",
    layout="wide"
)

st.title("🛡️ Strategic Theft Analysis Dashboard")
st.markdown("Analyze theft claims and exposure across your Ontario auto portfolio.")


# -------------------------------
# 2. LOAD DATA FROM CSV
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/clean_ontario_theft_data.csv",
                     parse_dates=['episode start date', 'episode expiry date', 'loss date'])

    # Correct column names
    df.rename(columns={
        'underwriting company name': 'company',
        'garaging postal code FSA': 'FSA',
        'claim count theft': 'is_stolen',
        'incurred theft claim': 'incurred',
        'earned exposure': 'exposure',
        'vehice code': 'vehicle_code'
    }, inplace=True)

    # Create vehicle code if missing
    if 'vehicle_code' not in df.columns:
        df['vehicle_code'] = df['vehicle make'] + ' ' + df['model'] + ' ' + df['vehicle year'].astype(str)

    df['quarter'] = df['episode start date'].dt.to_period('Q').astype(str)
    df['year'] = df['episode start date'].dt.year
    df['month'] = df['episode start date'].dt.to_period('M').dt.to_timestamp()

    return df


df = load_data()

# -------------------------------
# 3. TAB SELECTION
# -------------------------------
tab1, tab2 = st.tabs(["Portfolio Overview", "Vehicle Insights"])

# -------------------------------
# 4. TAB 1: Portfolio Overview
# -------------------------------
with tab1:
    st.header("📊 Portfolio Overview")

    # GLOBAL KPI CARDS (All Data)
    total_claims = df['is_stolen'].sum()
    total_exposure = df['exposure'].sum()
    total_claim_amount = df['incurred'].sum()
    loss_ratio = total_claim_amount / df['purchase amount'].sum() if df['purchase amount'].sum() > 0 else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Claims", f"{total_claims:,}")
    kpi2.metric("Total Exposure", f"{total_exposure:,.2f}")
    kpi3.metric("Total Claim Amount", f"${total_claim_amount:,.0f}")
    kpi4.metric("Loss Ratio", f"{loss_ratio:.2%}")

    st.markdown("---")

    # CONTROLS
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col1:
        group_options = {"Underwriting Company": "company", "Vehicle Model": "vehicle_code", "FSA": "FSA"}
        group_choice = st.selectbox("Group By", list(group_options.keys()))
        group_col = group_options[group_choice]

    with col2:
        top_n = st.slider(f"Top N {group_choice}", 1, 20, 10)

    with col3:
        time_options = {"Monthly": "month", "Quarterly": "quarter", "Yearly": "year"}
        time_choice = st.selectbox("Time Aggregation", list(time_options.keys()))
        time_col = time_options[time_choice]

    with col4:
        show_exposure = st.checkbox("Show Exposure Bars", value=True)

    with col5:
        all_categories = df[group_col].value_counts().index.tolist()
        selected_categories = st.multiselect(f"Select {group_choice}(s)", all_categories,
                                             default=all_categories[:top_n])

    filtered_df = df[df[group_col].isin(selected_categories)]

    if filtered_df.empty:
        st.warning("No data to display for the selected filters.")
    else:
        agg = filtered_df.groupby([time_col, group_col]).agg(
            num_claims=('is_stolen', 'sum'),
            exposure=('exposure', 'sum'),
            claim_amount=('incurred', 'sum')
        ).reset_index().sort_values(time_col)

        # PLOTLY DUAL AXIS CHART
        fig = go.Figure()
        if show_exposure:
            for grp in agg[group_col].unique():
                subset = agg[agg[group_col] == grp]
                fig.add_trace(go.Bar(
                    x=subset[time_col],
                    y=subset['exposure'],
                    name=f"{grp} Exposure",
                    yaxis='y2',
                    opacity=0.4
                ))
        for grp in agg[group_col].unique():
            subset = agg[agg[group_col] == grp]
            fig.add_trace(go.Scatter(
                x=subset[time_col],
                y=subset['num_claims'],
                mode='lines+markers',
                name=f"{grp} Claims",
                line=dict(width=2)
            ))

        fig.update_layout(
            template="plotly_white",
            title=f"Theft Claims & Exposure by {group_choice} ({time_choice})",
            xaxis_title=time_choice,
            yaxis_title="Number of Claims",
            yaxis2=dict(title="Exposure", overlaying='y', side='right'),
            hovermode="closest",
            font=dict(family="Segoe UI", size=12),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.35,  # move lower to avoid overlapping x-axis
                xanchor="center",
                x=0.5
            ),
            margin=dict(b=80)  # add bottom margin for legend
        )
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------
        # KPI CARDS FOR FILTERED DATA
        # -------------------------------
        st.markdown("---")
        st.subheader("Filtered Metrics")

        filtered_claims = filtered_df['is_stolen'].sum()
        filtered_exposure = filtered_df['exposure'].sum()
        filtered_claim_amount = filtered_df['incurred'].sum()
        filtered_loss_ratio = filtered_claim_amount / filtered_df['purchase amount'].sum() if filtered_df[
                                                                                                  'purchase amount'].sum() > 0 else 0

        fkpi1, fkpi2, fkpi3, fkpi4 = st.columns(4)
        fkpi1.metric("Filtered Claims", f"{filtered_claims:,}")
        fkpi2.metric("Filtered Exposure", f"{filtered_exposure:,.2f}")
        fkpi3.metric("Filtered Claim Amount", f"${filtered_claim_amount:,.0f}")
        fkpi4.metric("Filtered Loss Ratio", f"{filtered_loss_ratio:.2%}")

# -------------------------------
# 5. TAB 2: Vehicle Insights
# -------------------------------
with tab2:
    st.header("🚗 Vehicle Insights")

    # Vehicle selection
    vehicle_list = df['vehicle_code'].unique().tolist()
    selected_vehicle = st.selectbox("Select Vehicle", vehicle_list)

    vehicle_df = df[df['vehicle_code'] == selected_vehicle]

    # KPIs for selected vehicle
    veh_claims = vehicle_df['is_stolen'].sum()
    veh_exposure = vehicle_df['exposure'].sum()
    veh_claim_amount = vehicle_df['incurred'].sum()
    veh_loss_ratio = veh_claim_amount / vehicle_df['purchase amount'].sum() if vehicle_df[
                                                                                   'purchase amount'].sum() > 0 else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Claims", f"{veh_claims:,}")
    kpi2.metric("Total Exposure", f"{veh_exposure:,.2f}")
    kpi3.metric("Total Claim Amount", f"${veh_claim_amount:,.0f}")
    kpi4.metric("Loss Ratio", f"{veh_loss_ratio:.2%}")

    st.markdown("---")

    # Chart: Monthly claims & exposure for vehicle
    agg = vehicle_df.groupby('month').agg(
        num_claims=('is_stolen', 'sum'),
        exposure=('exposure', 'sum')
    ).reset_index().sort_values('month')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg['month'],
        y=agg['num_claims'],
        mode='lines+markers',
        name="Claims",
        line=dict(width=2, color='blue')
    ))
    fig.add_trace(go.Bar(
        x=agg['month'],
        y=agg['exposure'],
        name="Exposure",
        yaxis='y2',
        opacity=0.4
    ))

    fig.update_layout(
        template="plotly_white",
        title=f"{selected_vehicle} Theft Claims & Exposure",
        xaxis_title="Month",
        yaxis_title="Number of Claims",
        yaxis2=dict(title="Exposure", overlaying='y', side='right'),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.35,  # move legend lower
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=80)
    )
    st.plotly_chart(fig, use_container_width=True)

    # KPI CARDS BELOW VEHICLE PLOT
    st.markdown("---")
    st.subheader("Selected Vehicle Metrics")

    fkpi1, fkpi2, fkpi3, fkpi4 = st.columns(4)
    fkpi1.metric("Claims", f"{veh_claims:,}")
    fkpi2.metric("Exposure", f"{veh_exposure:,.2f}")
    fkpi3.metric("Claim Amount", f"${veh_claim_amount:,.0f}")
    fkpi4.metric("Loss Ratio", f"{veh_loss_ratio:.2%}")
