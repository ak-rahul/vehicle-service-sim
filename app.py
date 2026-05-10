import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import io
import os
import numpy as np

from src import config
from src.des_model import run_des
from src.hybrid_model import run_hybrid
from src.analysis import (
    extract_kpis,
    run_optimization_sweep,
    plot_kpi_comparison,
    plot_financials,
    plot_wait_distributions,
    plot_hourly_throughput,
    plot_outcome_donut,
    plot_personality_breakdown,
    plot_optimization_surface,
    plot_optimal_config_radar
)

# ─── Streamlit Configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="VSC Hybrid Sim | Advanced Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Premium UI Styling ──────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
<style>
    /* Global Styles */
    * { font-family: 'Outfit', sans-serif; }
    .stApp { background-color: #0E1117; }
    
    /* Metrics Styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(145deg, #1e222d, #14171f);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #2d343f;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        flex: 1;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover { transform: translateY(-5px); }
    .metric-label { color: #8e94a0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.8rem; font-weight: 600; margin-top: 0.5rem; }
    .metric-delta { font-size: 0.9rem; margin-top: 0.25rem; }
    
    /* Model Badges */
    .badge { padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-des { background-color: rgba(79, 156, 249, 0.2); color: #4F9CF9; }
    .badge-hybrid { background-color: rgba(249, 123, 79, 0.2); color: #F97B4F; }
    
    /* Custom Headers */
    .main-title { font-size: 3rem; font-weight: 600; background: -webkit-linear-gradient(#fff, #8e94a0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0; }
    .sub-title { color: #8e94a0; font-size: 1.1rem; margin-bottom: 2rem; }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; padding: 0; }
    .stTabs [data-baseweb="tab"] {
        height: 48px; border-radius: 8px; border: 1px solid transparent;
        color: #8e94a0; padding: 0 24px; transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: #1e222d; color: #fff; }
    .stTabs [aria-selected="true"] { background-color: #1e222d !important; border-color: #4F9CF9 !important; color: #fff !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Initialization ──────────────────────────────────────────────
if 'des_logs' not in st.session_state: st.session_state.des_logs = []
if 'hybrid_logs' not in st.session_state: st.session_state.hybrid_logs = []
if 'des_kpis' not in st.session_state: st.session_state.des_kpis = {}
if 'hybrid_kpis' not in st.session_state: st.session_state.hybrid_kpis = {}
if 'optimization_results' not in st.session_state: st.session_state.optimization_results = None

# ─── Sidebar configuration ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='color: #4F9CF9; margin-bottom: 0;'>⚙️ Control</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8e94a0; font-size: 0.9rem;'>Simulation Parameters</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    sim_time = st.slider("Simulation Horizon (Min)", 120, 1440, 600, 60, help="Total minutes to simulate.")
    
    st.markdown("#### 🏢 Resource Allocation")
    c1, c2 = st.columns(2)
    with c1:
        advisors = st.number_input("👨‍💼 Advisors", 1, 10, config.NUM_SERVICE_ADVISORS)
        general = st.number_input("🔧 Gen Bays", 1, 20, config.NUM_GENERAL_BAYS)
    with c2:
        inspection = st.number_input("🔍 Insp Bays", 1, 10, config.NUM_INSPECTION_BAYS)
        express = st.number_input("⚡ Exp Bays", 0, 10, config.NUM_EXPRESS_BAYS)
        
    st.markdown("---")
    # Issue 11: Provide a reproducible random seed
    seed = st.number_input("🎲 Random Seed", value=42)
    
    if st.button("🚀 EXECUTE SIMULATION", use_container_width=True, type="primary"):
        with st.spinner("Processing Hybrid Engines..."):
            cfg = {"num_advisors": advisors, "num_inspection": inspection, "num_general": general, "num_express": express}
            st.session_state.des_logs = run_des(sim_time, cfg, seed)
            st.session_state.hybrid_logs = run_hybrid(sim_time, cfg, seed)
            st.session_state.des_kpis = extract_kpis(st.session_state.des_logs, "DES", cfg=cfg, sim_time=sim_time)
            st.session_state.hybrid_kpis = extract_kpis(st.session_state.hybrid_logs, "Hybrid", cfg=cfg, sim_time=sim_time)
            st.toast("Simulation complete!", icon="✅")

# ─── Main Content ────────────────────────────────────────────────────────────

st.markdown("<h1 class='main-title'>Vehicle Service Center</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Advanced Hybrid Simulation & Operational Analytics</p>", unsafe_allow_html=True)

if not st.session_state.des_kpis:
    st.info("👈 Configure resources in the sidebar and click **EXECUTE SIMULATION** to generate data.")
    
    # Showcase logic
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🔄 Hybrid Modelling")
        st.write("Fusing Discrete Event Simulation with Agent-Based Modelling for behavioral accuracy.")
    with col2:
        st.markdown("#### 🧠 Agent Psychology")
        st.write("Real-time emotional decay (Liu-Zhen Model) driving balking and reneging decisions.")
    with col3:
        st.markdown("#### 🎯 Resource Optimization")
        st.write("Parallelized parameter sweeps to discover optimal operational configurations.")
    st.stop()

# Helper for premium metrics
def premium_metric(label, des_val, hyb_val, unit="", invert=False):
    diff = hyb_val - des_val
    color = "#F87171" if (diff > 0 if not invert else diff < 0) else "#34D399"
    trend = "↑" if diff > 0 else "↓" if diff < 0 else ""
    
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div style="display:flex; justify-content:space-between; align-items:flex-end;">
            <div>
                <div class="metric-value">{hyb_val:.1f}{unit}</div>
                <div class="metric-delta" style="color: {color}">{trend} {abs(diff):.1f}{unit} vs DES</div>
            </div>
            <div style="text-align: right; color: #8e94a0; font-size: 0.8rem;">
                DES: {des_val:.1f}{unit}
            </div>
        </div>
    </div>
    """
    return html

des = st.session_state.des_kpis
hyb = st.session_state.hybrid_kpis

# Metric Row
c1, c2, c3, c4 = st.columns(4)
c1.markdown(premium_metric("Throughput Rate", des["throughput_rate"], hyb["throughput_rate"], "%", invert=True), unsafe_allow_html=True)
c2.markdown(premium_metric("Net Profit", des["net_profit"], hyb["net_profit"], "$", invert=True), unsafe_allow_html=True)
c3.markdown(premium_metric("Lost Revenue", des["lost_revenue"], hyb["lost_revenue"], "$"), unsafe_allow_html=True)
c4.markdown(premium_metric("Avg Total Wait", des["avg_wait_all_stages"], hyb["avg_wait_all_stages"], "m"), unsafe_allow_html=True)

st.markdown("---")

tab_comp, tab_deep, tab_fin, tab_opt = st.tabs(["📊 Performance Benchmarking", "🧠 Behavioral Analysis", "💵 Financial Analysis", "🎯 Resource Optimization"])

with tab_comp:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(plot_kpi_comparison(des, hyb), use_container_width=True)
        st.plotly_chart(plot_hourly_throughput(st.session_state.des_logs, st.session_state.hybrid_logs), use_container_width=True)
    with col2:
        st.plotly_chart(plot_wait_distributions(st.session_state.des_logs, st.session_state.hybrid_logs), use_container_width=True)
        st.markdown("#### 📝 Model Summary")
        st.write(f"**DES Model:** Processed {des['total_vehicles']} vehicles with {des['completed']} completions.")
        st.write(f"**Hybrid Model:** Processed {hyb['total_vehicles']} vehicles with {hyb['completed']} completions.")
        st.write("The Hybrid model accounts for psychological wait thresholds, often leading to higher renege rates but more realistic throughput projections.")

with tab_deep:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_outcome_donut(hyb, "Hybrid Context"), use_container_width=True)
    with col2:
        st.plotly_chart(plot_personality_breakdown(st.session_state.hybrid_logs), use_container_width=True)
    
    st.markdown("#### 📈 Emotional Decay Analysis")
    df_h = pd.DataFrame(st.session_state.hybrid_logs)
    if 'emotion' in df_h.columns and not df_h.empty:
        # Issue 9: Plot the final emotion state, not the initial one
        cust_data = df_h.drop_duplicates(subset=['vehicle'], keep='last')[['vehicle', 'personality', 'emotion', 'patience']]
        fig = px.histogram(cust_data, x="emotion", color="personality", barmode="overlay", 
                           title="Distribution of Final Agent Emotions", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

with tab_fin:
    st.markdown("### 💵 Financial Performance")
    st.write("Evaluating the economic impact of operational bottlenecks, balking, and reneging.")
    st.plotly_chart(plot_financials(des, hyb), use_container_width=True)

with tab_opt:
    st.markdown("### 🔍 Intelligent Parameter Sweep")
    st.write("Iteratively testing resource combinations to maximize profit or minimize wait times.")
    
    with st.form("opt_form"):
        st.markdown("**Range Selection:**")
        c1, c2, c3, c4, c5 = st.columns(5)
        a_r = c1.multiselect("Advisors", [1, 2, 3, 4], default=[1, 2])
        i_r = c2.multiselect("Inspection", [1, 2, 3, 4], default=[1, 2])
        g_r = c3.multiselect("Gen Bays", [4, 5, 6, 7, 8], default=[5, 6])
        e_r = c4.multiselect("Exp Bays", [1, 2, 3], default=[1])
        opt_target = c5.selectbox("Optimize For", ["Net Profit", "Wait Time"])
        
        if st.form_submit_button("Start Optimization", use_container_width=True):
            with st.spinner("Running parallel simulations..."):
                sort_col = "net_profit" if opt_target == "Net Profit" else "avg_wait_all_stages"
                ascending = False if opt_target == "Net Profit" else True
                st.session_state.optimization_results = run_optimization_sweep(
                    run_hybrid, sim_time, a_r, i_r, g_r, e_r, 
                    n_reps=2, sort_by=sort_col, ascending=ascending
                )
                st.session_state.opt_target = opt_target
                
    if st.session_state.optimization_results is not None:
        best = st.session_state.optimization_results.iloc[0]
        st.success("Analysis Complete: Optimal Configuration Identified")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### 🏆 Recommendation")
            st.info(f"👨‍💼 Advisors: **{int(best['num_advisors'])}**")
            st.info(f"🔍 Inspection: **{int(best['num_inspection'])}**")
            st.info(f"🔧 Gen Bays: **{int(best['num_general'])}**")
            st.info(f"⚡ Exp Bays: **{int(best['num_express'])}**")
            st.markdown(f"**Projected Wait:** {best['avg_wait_all_stages']:.1f}m")
            st.markdown(f"**Projected Profit:** ${best['net_profit']:,.2f}")
        with col2:
            st.plotly_chart(plot_optimal_config_radar(best, hyb), use_container_width=True)
        
        target_col = "net_profit" if st.session_state.get("opt_target") == "Net Profit" else "avg_wait_all_stages"
        st.plotly_chart(plot_optimization_surface(st.session_state.optimization_results, y_col=target_col), use_container_width=True)
