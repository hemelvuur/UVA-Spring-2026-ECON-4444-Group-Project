"""
AI Labor Market Diffusion — Interactive Dashboard
UVA ECON 4444 Group Project
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model import run_simulation

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Labor Market Simulator",
    page_icon="🤖",
    layout="wide",
)

st.title("AI Labor Market Diffusion Simulator")
st.caption("UVA ECON 4444 — Artificial Intelligence and the Future of Work")

# ── Sidebar: parameters ───────────────────────────────────────────────────────
st.sidebar.header("Simulation Parameters")

T = st.sidebar.radio("Time horizon (periods)", [10, 20], index=1, horizontal=True)

st.sidebar.subheader("Initial Conditions")
I0 = st.sidebar.slider(
    "I₀ — Initial automation share",
    min_value=0.0, max_value=0.5, value=0.05, step=0.01,
    help="Share of tasks already automated at t = 0.",
)

st.sidebar.subheader("Bass Diffusion")
p = st.sidebar.slider(
    "p — Innovation parameter",
    min_value=0.001, max_value=0.1, value=0.03, step=0.001,
    format="%.3f",
    help="Rate of adoption driven by external factors.",
)
q = st.sidebar.slider(
    "q — Imitation parameter",
    min_value=0.0, max_value=0.5, value=0.20, step=0.01,
    help="Rate of adoption driven by word-of-mouth / network effects.",
)

st.sidebar.subheader("AI Productivity")
g = st.sidebar.slider(
    "g — AI productivity growth rate",
    min_value=0.0, max_value=0.2, value=0.05, step=0.005,
    format="%.3f",
    help="A_K(t) = 1 + g·t",
)

st.sidebar.subheader("Production Structure")
sigma = st.sidebar.slider(
    "σ — CES elasticity of substitution",
    min_value=0.2, max_value=5.0, value=1.5, step=0.1,
    help="Elasticity across tasks. Keep away from exactly 1.",
)

# ── Run simulation ─────────────────────────────────────────────────────────────
results = run_simulation(T=T, I0=I0, p=p, q=q, g=g, sigma=sigma)

t      = results["t"]
I      = results["I"]
A_K    = results["A_K"]
Y      = results["Y"]
z_star = results["z_star"]

start_year = 2025
x = [start_year + int(i) for i in t]

# ── Charts ────────────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "AI Adoption Path  I(t)",
        "Aggregate Output  Y(t)",
        "AI Productivity  A_K(t)",
        "Adoption vs. Frontier  [diagnostic]",
    ),
)

def line(fig, x, y, name, row, col, dash="solid"):
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", name=name,
                   line=dict(dash=dash), showlegend=(row == 2 and col == 2)),
        row=row, col=col,
    )

line(fig, x, I,      "Adoption I(t)",   1, 1)
line(fig, x, Y,      "Output Y(t)",     1, 2)
line(fig, x, A_K,    "AI productivity", 2, 1)
line(fig, x, I,      "I(t)",            2, 2)
line(fig, x, z_star, "z*(t) frontier",  2, 2, dash="dash")

fig.update_layout(height=620, margin=dict(t=40, b=20), legend=dict(x=0.75, y=0.15))
st.plotly_chart(fig, use_container_width=True)

# ── Diagnostic note ───────────────────────────────────────────────────────────
if I[-1] > z_star[-1]:
    st.warning(
        f"At t = {T}, adoption I = {I[-1]:.3f} exceeds the productivity frontier "
        f"z* = {z_star[-1]:.3f}. Some automated tasks have lower AI output than "
        f"labor would produce."
    )

# ── Summary metrics ───────────────────────────────────────────────────────────
st.subheader("End-of-period Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Automation share I(T)", f"{I[-1]:.3f}",  f"{I[-1]  - I[0]:+.3f}")
col2.metric("Aggregate output Y(T)", f"{Y[-1]:.3f}",  f"{Y[-1]  - Y[0]:+.3f}")
col3.metric("AI productivity A_K(T)",f"{A_K[-1]:.3f}",f"{A_K[-1]- A_K[0]:+.3f}")
col4.metric("Frontier z*(T)",        f"{z_star[-1]:.3f}", f"{z_star[-1] - z_star[0]:+.3f}")
