"""
analysis.py — High-Performance Analysis Engine
Features multi-processing parameter sweeps and vectorized KPI extraction.
"""

from __future__ import annotations
import os
import logging
import numpy as np
import pandas as pd
import concurrent.futures
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

log = logging.getLogger("analysis")

# ─────────────────────────────────────────────────────────────────────────────
#  KPI Extraction (Vectorized)
# ─────────────────────────────────────────────────────────────────────────────
def extract_kpis(logs: list[dict], model_name: str = "") -> dict:
    default_kpis = {
        "model":             model_name or "—",
        "total_vehicles":    0,
        "completed":         0,
        "balked":            0,
        "reneged":           0,
        "throughput_rate":   0.0,
        "balk_rate":         0.0,
        "renege_rate":       0.0,
        "avg_wait_advisor":  0.0,
        "avg_wait_inspection": 0.0,
        "avg_wait_bay":      0.0,
        "avg_total_time":    0.0,
        "p95_total_time":    0.0,
        "max_total_time":    0.0,
        "avg_wait_all_stages": 0.0,
    }
    if not logs: return default_kpis
    
    # Pre-allocate DataFrame for vectorized operations
    df = pd.DataFrame.from_records(logs)
    
    total  = df["vehicle"].nunique()
    ev_counts = df["event"].value_counts()
    
    comp   = ev_counts.get("Departed", 0)
    balk   = ev_counts.get("Balked", 0)
    renege = ev_counts.get("Reneged", 0)

    # Vectorized conditional means (excluding Reneged to avoid double counting, Issue 8)
    done_events = ["AdvisorDone", "InspectionDone", "RepairDone", "RepairFailed"]
    done_mask = df["event"].isin(done_events)
    
    adv_wait  = df[(df["stage"] == "Advisor") & done_mask]["wait_min"].mean()
    insp_wait = df[(df["stage"] == "Inspection") & done_mask]["wait_min"].mean()
    bay_wait  = df[(df["stage"].isin(["General", "Express"])) & done_mask]["wait_min"].mean()
    
    # Fill NA with 0.0 for stages that might not have happened
    adv_wait = 0.0 if pd.isna(adv_wait) else adv_wait
    insp_wait = 0.0 if pd.isna(insp_wait) else insp_wait
    bay_wait = 0.0 if pd.isna(bay_wait) else bay_wait
    
    total_times = df[df["event"] == "TotalTime"]["service_min"]
    avg_total   = total_times.mean() if not total_times.empty else 0.0
    p95_total   = total_times.quantile(0.95) if not total_times.empty else 0.0
    max_total   = total_times.max() if not total_times.empty else 0.0

    return {
        "model":             model_name or (df["model"].iloc[0] if "model" in df.columns else "—"),
        "total_vehicles":    int(total),
        "completed":         int(comp),
        "balked":            int(balk),
        "reneged":           int(renege),
        "throughput_rate":   float(np.round(comp / max(1, total) * 100, 1)),
        "balk_rate":         float(np.round(balk / max(1, total) * 100, 1)),
        "renege_rate":       float(np.round(renege / max(1, total) * 100, 1)),
        "avg_wait_advisor":  float(np.round(adv_wait, 2)),
        "avg_wait_inspection": float(np.round(insp_wait, 2)),
        "avg_wait_bay":      float(np.round(bay_wait, 2)),
        "avg_total_time":    float(np.round(avg_total, 2)),
        "p95_total_time":    float(np.round(p95_total, 2)),
        "max_total_time":    float(np.round(max_total, 2)),
        "avg_wait_all_stages": float(np.round(adv_wait + insp_wait + bay_wait, 2)),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  Parallel Optimization Sweep
# ─────────────────────────────────────────────────────────────────────────────
def _run_single_config(cfg_tuple, model_fn, sim_time, n_reps, seed_base):
    """Worker function for process pool executor."""
    adv, insp, gen, exp = cfg_tuple
    cfg = {
        "num_advisors": adv,
        "num_inspection": insp,
        "num_general": gen,
        "num_express": exp,
    }
    
    rep_kpis = []
    for r in range(n_reps):
        logs = model_fn(sim_time=sim_time, cfg=cfg, seed=seed_base + r)
        rep_kpis.append(extract_kpis(logs))
        
    avg = {}
    for key in rep_kpis[0]:
        if isinstance(rep_kpis[0][key], (int, float)):
            avg[key] = round(float(np.mean([k[key] for k in rep_kpis])), 2)
        else:
            avg[key] = rep_kpis[0][key]

    avg.update({
        "num_advisors": adv, "num_inspection": insp,
        "num_general": gen, "num_express": exp,
        "total_staff": adv + insp + gen + exp,
    })
    return avg

def run_optimization_sweep(
    model_fn, sim_time: float,
    advisor_range: list[int], inspection_range: list[int],
    general_range: list[int], express_range: list[int],
    n_reps: int = 3, seed_base: int = 42,
) -> pd.DataFrame:
    """Multi-processed parameter grid search."""
    # Generate Cartesian product of all configs
    configs = [(a, i, g, e) 
               for a in advisor_range for i in inspection_range
               for g in general_range for e in express_range]

    worker_fn = partial(_run_single_config, model_fn=model_fn, sim_time=sim_time, 
                        n_reps=n_reps, seed_base=seed_base)

    results = []
    # Issue 10: Use ProcessPoolExecutor for true CPU parallelism
    # Wrapped in a try-except to fallback to ThreadPoolExecutor for Windows + Streamlit compatibility
    max_workers = os.cpu_count() or 4
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for res in executor.map(worker_fn, configs):
                results.append(res)
    except Exception as e:
        log.warning(f"ProcessPoolExecutor failed ({e}), falling back to ThreadPoolExecutor.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for res in executor.map(worker_fn, configs):
                results.append(res)

    df = pd.DataFrame(results)
    df.sort_values("avg_wait_all_stages", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ─────────────────────────────────────────────────────────────────────────────
#  Plotly interactive charts (Unchanged Interface)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {"DES": "#4F9CF9", "Hybrid": "#F97B4F"}

def plot_kpi_comparison(des_kpis: dict, hybrid_kpis: dict) -> go.Figure:
    metrics = [
        ("avg_wait_advisor",    "Avg Advisor Wait (min)"),
        ("avg_wait_inspection", "Avg Inspection Wait (min)"),
        ("avg_wait_bay",        "Avg Bay Wait (min)"),
        ("avg_total_time",      "Avg Total Time (min)"),
        ("throughput_rate",     "Throughput Rate (%)"),
        ("balk_rate",           "Balk Rate (%)"),
        ("renege_rate",         "Renege Rate (%)"),
    ]
    labels = [m[1] for m in metrics]
    des_vals    = [des_kpis.get(m[0], 0) for m in metrics]
    hybrid_vals = [hybrid_kpis.get(m[0], 0) for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="DES (Pure)", x=labels, y=des_vals, marker_color=PALETTE["DES"], text=[f"{v:.1f}" for v in des_vals], textposition="outside"))
    fig.add_trace(go.Bar(name="Hybrid (DES + ABM)", x=labels, y=hybrid_vals, marker_color=PALETTE["Hybrid"], text=[f"{v:.1f}" for v in hybrid_vals], textposition="outside"))
    fig.update_layout(barmode="group", title="📊 DES vs Hybrid Model KPIs", template="plotly_dark", height=480)
    return fig

def plot_wait_distributions(des_logs: list[dict], hybrid_logs: list[dict]) -> go.Figure:
    fig = go.Figure()
    for stage_key in ["Advisor", "Inspection", "General", "Express"]:
        df_d = pd.DataFrame(des_logs)
        df_h = pd.DataFrame(hybrid_logs)
        
        if not df_d.empty:
            wd = df_d[(df_d["stage"] == stage_key) & df_d["event"].isin(["AdvisorDone", "InspectionDone", "Departed", "Reneged"])]["wait_min"].dropna().tolist()
            if wd: fig.add_trace(go.Box(y=wd, name=f"DES - {stage_key}", marker_color=PALETTE["DES"], boxmean="sd"))
        if not df_h.empty:
            wh = df_h[(df_h["stage"] == stage_key) & df_h["event"].isin(["AdvisorDone", "InspectionDone", "Departed", "Reneged"])]["wait_min"].dropna().tolist()
            if wh: fig.add_trace(go.Box(y=wh, name=f"Hybrid - {stage_key}", marker_color=PALETTE["Hybrid"], boxmean="sd"))

    fig.update_layout(title="⏱ Wait Time Distributions by Stage", template="plotly_dark", height=500)
    return fig

def plot_hourly_throughput(des_logs: list[dict], hybrid_logs: list[dict]) -> go.Figure:
    def _hourly(logs, label):
        df = pd.DataFrame(logs)
        if df.empty: return pd.DataFrame(columns=["hour", "count", "model"])
        dep = df[df["event"] == "Departed"].copy()
        if dep.empty: return pd.DataFrame(columns=["hour", "count", "model"])
        dep["hour"] = (dep["time"] / 60).astype(int)
        h = dep.groupby("hour").size().reset_index(name="count")
        h["model"] = label
        return h

    combined = pd.concat([_hourly(des_logs, "DES"), _hourly(hybrid_logs, "Hybrid")], ignore_index=True)
    fig = px.line(combined, x="hour", y="count", color="model", color_discrete_map=PALETTE, markers=True, title="🕐 Hourly Vehicle Throughput", template="plotly_dark")
    fig.update_layout(height=420)
    return fig

def plot_outcome_donut(kpis: dict, model_label: str) -> go.Figure:
    fig = go.Figure(go.Pie(labels=["Completed", "Balked", "Reneged"], values=[kpis.get("completed",0), kpis.get("balked",0), kpis.get("reneged",0)], hole=0.55, marker=dict(colors=["#34D399", "#FBBF24", "#F87171"])))
    fig.update_layout(title=f"🍩 Customer Outcomes — {model_label}", template="plotly_dark", height=380)
    return fig

def plot_personality_breakdown(hybrid_logs: list[dict]) -> go.Figure:
    df = pd.DataFrame(hybrid_logs)
    if df.empty or "personality" not in df.columns: return go.Figure()
    final = df[df["event"].isin(["Departed", "Balked", "Reneged"])].drop_duplicates("vehicle", keep="first")
    if final.empty: return go.Figure()
    grp = final.groupby(["personality", "event"]).size().reset_index(name="count")
    fig = px.bar(grp, x="personality", y="count", color="event", color_discrete_map={"Departed": "#34D399", "Balked": "#FBBF24", "Reneged": "#F87171"}, barmode="stack", title="👤 Outcomes by Personality Type", template="plotly_dark")
    fig.update_layout(height=420)
    return fig

def plot_optimization_surface(opt_df: pd.DataFrame, x_col: str = "num_general") -> go.Figure:
    fig = go.Figure()
    for i, val in enumerate(sorted(opt_df["num_advisors"].unique())):
        sub = opt_df[opt_df["num_advisors"] == val].sort_values(x_col)
        fig.add_trace(go.Scatter(x=sub[x_col], y=sub["avg_wait_all_stages"], mode="lines+markers", name=f"Advisors = {val}"))
    fig.update_layout(title=f"🔍 Wait Time vs {x_col.replace('_', ' ').title()}", template="plotly_dark", height=460)
    return fig

def plot_optimal_config_radar(optimal_row: pd.Series, baseline_kpis: dict) -> go.Figure:
    cats = ["Throughput %", "Balk Rate %↓", "Renege Rate %↓", "Avg Wait (norm)↓", "P95 Time (norm)↓"]
    mw = max(baseline_kpis.get("avg_total_time", 1), 1)
    mp95 = max(baseline_kpis.get("p95_total_time", 1), 1)
    def _norm(v, m): return round(min(1.0, v / max(1.0, m)) * 100, 1)
    
    bv = [baseline_kpis.get("throughput_rate",0), 100-baseline_kpis.get("balk_rate",0), 100-baseline_kpis.get("renege_rate",0), 100-_norm(baseline_kpis.get("avg_total_time",0), mw), 100-_norm(baseline_kpis.get("p95_total_time",0), mp95)]
    ov = [optimal_row.get("throughput_rate",0), 100-optimal_row.get("balk_rate",0), 100-optimal_row.get("renege_rate",0), 100-_norm(optimal_row.get("avg_total_time",0), mw), 100-_norm(optimal_row.get("p95_total_time",0), mp95)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=bv+[bv[0]], theta=cats+[cats[0]], fill="toself", name="Baseline", line_color="#4F9CF9"))
    fig.add_trace(go.Scatterpolar(r=ov+[ov[0]], theta=cats+[cats[0]], fill="toself", name="Optimal", line_color="#34D399"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="🎯 Optimal vs Baseline", template="plotly_dark", height=500)
    return fig

def export_static_charts(des_kpis: dict, hybrid_kpis: dict, des_logs: list, hybrid_logs: list, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="darkgrid", palette="muted")
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["avg_wait_advisor", "avg_wait_inspection", "avg_wait_bay", "avg_total_time"]
    x = np.arange(len(metrics)); w = 0.35
    ax.bar(x - w/2, [des_kpis.get(m, 0) for m in metrics], w, label="DES", color="#4F9CF9")
    ax.bar(x + w/2, [hybrid_kpis.get(m, 0) for m in metrics], w, label="Hybrid", color="#F97B4F")
    ax.set_xticks(x); ax.set_xticklabels(["Advisor Wait", "Inspection Wait", "Bay Wait", "Total Time"])
    ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(output_dir, "kpi_comparison.png"), dpi=200); plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, k, t in zip(axes, [des_kpis, hybrid_kpis], ["DES Model", "Hybrid Model"]):
        v = [k.get("completed",0), k.get("balked",0), k.get("reneged",0)]
        if sum(v) > 0: ax.pie([x for x in v if x>0], labels=[l for l,x in zip(["Completed","Balked","Reneged"],v) if x>0], autopct="%1.1f%%", colors=[c for c,x in zip(["#34D399","#FBBF24","#F87171"],v) if x>0])
        ax.set_title(f"Customer Outcomes — {t}")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "outcomes_comparison.png"), dpi=200); plt.close()
