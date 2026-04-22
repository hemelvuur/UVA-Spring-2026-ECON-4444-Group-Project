"""
AI Labor Market Diffusion — Presentation Dashboard
UVA ECON 4444 Group Project
Paper: "Task Reallocation and the AI Productivity Paradox"
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model import run_simulation

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Productivity Paradox",
    page_icon="💽",
    layout="wide",
)

WIN98_FONT = '"Pixelated MS Sans Serif", "MS Sans Serif", "Microsoft Sans Serif", Tahoma, Geneva, sans-serif'

st.markdown(
    f"""
    <style>
      /* Desktop */
      [data-testid="stAppViewContainer"], body {{
          background: #008080 !important;
      }}

      /* Streamlit header = Win98 title bar carrying the project title */
      [data-testid="stHeader"] {{
          background: linear-gradient(90deg, #000080 0%, #1084d0 100%) !important;
          height: 38px !important;
          border-bottom: 2px solid #000000 !important;
          box-shadow: inset 1px 1px 0 #ffffff80 !important;
      }}
      [data-testid="stHeader"]::before {{
          content: "Task Reallocation and the AI Productivity Paradox";
          position: absolute;
          left: 0;
          right: 0;
          top: 10px;
          height: 38px;
          display: flex;
          align-items: center;
          justify-content: center;
          text-align: center;
          color: #ffffff;
          font-family: {WIN98_FONT};
          font-size: 20px;
          font-weight: 600;
          letter-spacing: 1px;
          pointer-events: none;
          z-index: 999;
      }}
      /* Keep the toolbar buttons visible above the gradient */
      [data-testid="stToolbar"] {{ z-index: 1000; }}
      [data-testid="stMainMenu"] button,
      [data-testid="stAppDeployButton"] button {{
          color: #ffffff !important;
      }}

      /* Main content = a Win98 window panel */
      .block-container {{
          background: #c0c0c0 !important;
          color: #000000 !important;
          font-family: {WIN98_FONT};
          font-size: 13px;
          padding: 0 0 1rem 0 !important;
          margin: 0 !important;
          max-width: 1180px;
          border: 2px solid;
          border-color: #dfdfdf #000000 #000000 #dfdfdf;
          box-shadow: inset 1px 1px 0 #ffffff, inset -1px -1px 0 #808080;
      }}

      /* H1 = the window's title bar */
      .block-container h1 {{
          background: linear-gradient(90deg, #000080 0%, #1084d0 100%);
          color: #ffffff !important;
          font-family: {WIN98_FONT};
          font-size: 1.25rem !important;
          font-weight: bold !important;
          letter-spacing: 1px !important;
          padding: 4px 8px !important;
          margin: 0 0 0.8rem 0 !important;
      }}

      /* H2 = section title bars */
      .block-container h2 {{
          background: linear-gradient(90deg, #000080 0%, #1084d0 100%);
          color: #ffffff !important;
          font-family: {WIN98_FONT};
          font-size: 1rem !important;
          font-weight: bold !important;
		  text-align: center;
          letter-spacing: 1px !important;
          padding: 3px 8px !important;
		  margin-bottom: 1rem !important;
          border: none !important;
      }}

      /* H3 = bold labels */
      .block-container h3 {{
          font-family: {WIN98_FONT};
          font-size: 0.9rem !important;
          font-weight: bold !important;
          color: #000000 !important;
          letter-spacing: 0 !important;
      }}

      /* Content gutters — simulates title bar spanning full width, prose inset */
      .block-container > div:first-child > div > div {{ padding: 0 1rem; }}

      /* Body text */
      .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown strong,
      [data-testid="stCaptionContainer"] {{
          font-family: {WIN98_FONT};
          font-size: 13px;
          color: #000000 !important;
      }}
      .stMarkdown a {{ color: #0000ff; }}

      /* Blockquote = sunken callout */
      .block-container blockquote,
      .paper-box {{
          background: #ffffff !important;
          border: 2px solid !important;
          border-color: #808080 #ffffff #ffffff #808080 !important;
          box-shadow: inset 1px 1px 0 #000000, inset -1px -1px 0 #dfdfdf !important;
          padding: 10px 14px !important;
          border-radius: 0 !important;
          color: #000000 !important;
          font-family: {WIN98_FONT};
		  margin: 0 0 1em 0;
      }}
      .paper-box {{
          margin: 0.6rem 1rem 0.8rem 1rem !important;
          display: block;
          overflow: hidden;
      }}
      .paper-box p, .paper-box li, .paper-box strong {{
          color: #000000 !important;
          font-family: {WIN98_FONT};
          font-size: 13px;
      }}

      /* Part banners = audience handoff cues between presenters */
      .part-banner {{
          background: repeating-linear-gradient(
              45deg, #000080 0 12px, #1084d0 12px 24px
          );
          color: #ffffff;
          font-family: {WIN98_FONT};
          font-size: 1.25rem;
          font-weight: bold;
          text-transform: uppercase;
          letter-spacing: 1px;
          padding: 8px 14px !important;
          margin: 2.2rem 0 0.4rem 0 !important;
          border-top: 2px solid #ffffff;
          border-bottom: 2px solid #000000;
          box-shadow: 0 2px 0 #00000033;
          text-align: center;
      }}
      .part-banner .part-label {{
          opacity: 0.75;
          margin-right: 10px;
      }}
	  .stAlert {{
		padding: 0 !important;
	  }}
      /* Alerts (info/warning/success) = dialog boxes */
      [data-testid="stAlert"], [data-testid="stNotification"] {{
          background: #c0c0c0 !important;
          color: #000000 !important;
          border: 2px solid !important;
          border-color: #ffffff #808080 #808080 #ffffff !important;
          border-radius: 0 !important;
          box-shadow: 1px 1px 0 #000000;
          font-family: {WIN98_FONT};
          font-size: 13px !important;
      }}
      [data-testid="stAlert"] * {{ color: #000000 !important; font-family: {WIN98_FONT}; }}

      /* Metrics = sunken inset panels */
      [data-testid="stMetric"] {{
          background: #ffffff;
          border: 2px solid;
          border-color: #808080 #ffffff #ffffff #808080;
          box-shadow: inset 1px 1px 0 #000000, inset -1px -1px 0 #dfdfdf;
          padding: 8px 10px;
          border-radius: 0;
          margin: 4px;
      }}
      [data-testid="stMetricLabel"] {{
          font-family: {WIN98_FONT} !important; font-size: 12px !important;
          color: #000000 !important; font-weight: normal !important;
      }}
      [data-testid="stMetricValue"] {{
          font-family: {WIN98_FONT} !important; font-size: 1.25rem !important;
          color: #000080 !important; font-weight: bold !important;
      }}
      [data-testid="stMetricDelta"] {{
          font-family: {WIN98_FONT} !important; font-size: 11px !important;
          color: #404040 !important;
      }}

      /* Bordered containers = white-inset paper boxes (equations, lesson cards).
         st.container(border=True) renders:
           <div data-testid="stVerticalBlockBorderWrapper">      <-- we paint this
             <div data-testid="stVerticalBlock">                 <-- and this
               ...content...
             </div>
           </div>
         Both need the white background or grey bleeds through. */
      .block-container [data-testid="stVerticalBlockBorderWrapper"] {{
          background: #ffffff !important;
          border: 2px solid !important;
          border-color: #808080 #ffffff #ffffff #808080 !important;
          border-radius: 0 !important;
          box-shadow: inset 1px 1px 0 #000000, inset -1px -1px 0 #dfdfdf !important;
          padding: 10px 14px !important;
        #   margin: 0.6rem 1rem 0.8rem 1rem !important;
      }}
      .block-container [data-testid="stVerticalBlockBorderWrapper"]
        > [data-testid="stVerticalBlock"],
      .block-container [data-testid="stVerticalBlockBorderWrapper"]
        [data-testid="stHorizontalBlock"],
      .block-container [data-testid="stVerticalBlockBorderWrapper"]
        [data-testid="stColumn"] {{
          background: #ffffff !important;
      }}
      /* Make sure KaTeX / markdown inside paper boxes stays transparent
         (let the white parent show through) */
      .block-container [data-testid="stVerticalBlockBorderWrapper"]
        [data-testid="stMarkdownContainer"],
      .block-container [data-testid="stVerticalBlockBorderWrapper"]
        .katex,
      .block-container [data-testid="stVerticalBlockBorderWrapper"]
        .katex-display {{
          background: transparent !important;
      }}

      /* Expander */
      [data-testid="stExpander"] {{
          background: #c0c0c0 !important;
          border: 2px solid !important;
          border-color: #ffffff #808080 #808080 #ffffff !important;
          border-radius: 0 !important;
          box-shadow: 1px 1px 0 #000000;
          font-family: {WIN98_FONT};
      }}
      [data-testid="stExpander"] summary {{
          background: #c0c0c0 !important;
          color: #000000 !important;
          font-family: {WIN98_FONT};
          font-weight: bold;
      }}

      /* Tables */
      table {{
          font-family: {WIN98_FONT} !important;
          font-size: 12px !important;
          background: #ffffff !important;
          border: 2px solid !important;
          border-color: #808080 #ffffff #ffffff #808080 !important;
          border-collapse: collapse !important;
      }}
      table thead th {{
          background: #c0c0c0 !important;
          color: #000000 !important;
          border: 1px solid;
          border-color: #ffffff #808080 #808080 #ffffff !important;
          font-weight: bold !important;
          padding: 3px 8px !important;
      }}
      table tbody td, table tbody th {{
          border: 1px solid #dfdfdf !important;
          padding: 3px 8px !important;
          color: #000000 !important;
      }}

      /* Divider */
      hr {{
          border: none !important;
          border-top: 1px solid #808080 !important;
          border-bottom: 1px solid #ffffff !important;
          margin: 0 1rem !important;
      }}

      /* Sidebar = Win98 side panel */
      [data-testid="stSidebar"] {{
          background: #c0c0c0 !important;
          border-right: 2px solid #000000;
          font-family: {WIN98_FONT};
      }}
      [data-testid="stSidebar"] * {{
          font-family: {WIN98_FONT} !important;
          color: #000000 !important;
      }}
      /* Restore Material Symbols/Icons font on icon glyphs — our blanket
         [data-testid="stSidebar"] * rule above clobbers the icon ligature
         font and leaves raw text like "keyboard_double_arrow_left". */
      [data-testid*="Icon"],
      [data-testid*="icon"],
      [class*="Material"],
      [class*="material-"],
      span.material-icons,
      span.material-icons-outlined,
      span.material-symbols-rounded,
      span.material-symbols-outlined,
      [data-testid="stSidebarCollapseButton"] span,
      [data-testid="stSidebarCollapsedControl"] span,
      [data-testid="stIconMaterial"],
      [data-testid="stExpander"] summary svg + span,
      button[kind="headerNoPadding"] span {{
          font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                       "Material Icons", "Material Icons Outlined" !important;
          font-weight: normal !important;
          font-style: normal !important;
          font-feature-settings: "liga" !important;
          -webkit-font-feature-settings: "liga" !important;
          text-transform: none !important;
          letter-spacing: normal !important;
          word-wrap: normal !important;
          white-space: nowrap !important;
          direction: ltr !important;
          -webkit-font-smoothing: antialiased !important;
      }}
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3 {{
          background: linear-gradient(90deg, #000080 0%, #1084d0 100%) !important;
          color: #ffffff !important;
          font-size: 0.85rem !important;
          font-weight: bold !important;
          padding: 3px 8px !important;
          margin: 0.8rem -0.5rem 0.5rem -0.5rem !important;
          letter-spacing: 0 !important;
      }}
      [data-testid="stSidebar"] label {{
          font-size: 12px !important;
          font-weight: normal !important;
      }}
      /* Sidebar sliders: thumb becomes a chunky raised button */
      [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {{
          background: #c0c0c0 !important;
          border: 2px solid !important;
          border-color: #ffffff #000000 #000000 #ffffff !important;
          box-shadow: inset 1px 1px 0 #dfdfdf, inset -1px -1px 0 #808080 !important;
          border-radius: 0 !important;
          width: 14px !important;
          height: 20px !important;
      }}
      [data-testid="stSidebar"] [data-baseweb="slider"] > div > div {{
          background: #808080 !important;
          border-radius: 0 !important;
          height: 4px !important;
      }}
      /* Radio buttons as retro toggles */
      [data-testid="stSidebar"] [role="radiogroup"] label {{
          background: #c0c0c0;
          border: 2px solid;
          border-color: #ffffff #808080 #808080 #ffffff;
          padding: 3px 10px !important;
          margin: 2px !important;
      }}

      /* Muted caption */
      .caption-muted {{
          color: #404040 !important;
          font-family: {WIN98_FONT};
          font-size: 12px;
          margin: 0.2rem 1rem 0.4rem 1rem;
      }}

      /* LaTeX gets a slightly larger, serif render for contrast */
      .katex {{ font-size: 1.05em !important; }}

      /* Remove Streamlit top bar rounded corners */
      .stApp {{ font-family: {WIN98_FONT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ══ Sidebar: simulation parameters ════════════════════════════════════════════
st.sidebar.header("Simulation Parameters")
st.sidebar.caption("Adjust, then scroll to the *Interactive Simulator* section.")

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
    help="Below 1: gross complements. Above 1: substitutes. Keep away from exactly 1.",
)


# ══ Run simulation ════════════════════════════════════════════════════════════
results = run_simulation(T=T, I0=I0, p=p, q=q, g=g, sigma=sigma)
t      = results["t"]
I      = results["I"]
A_K    = results["A_K"]
Y      = results["Y"]
z_star = results["z_star"]

start_year = 2025
years = [start_year + int(i) for i in t]


# ══ 1. HERO ═══════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='caption-muted' style='margin-top:0.6rem;'>A diffusion-driven mechanism for the AI J-curve "
    "&nbsp;·&nbsp; Artificial Intelligence and the Future of Work</div>",
    unsafe_allow_html=True,
)


# ══ PART 1 ════════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='part-banner'><span class='part-label'>Part 1 of 3</span>"
    "Motivation &amp; Literature</div>",
    unsafe_allow_html=True,
)

# ══ 2. THE PUZZLE ═════════════════════════════════════════════════════════════
st.header("The \"Puzzle\"")
st.markdown(
    "Forty years ago, Robert Solow quipped that the computer age was visible everywhere "
    "except in the productivity statistics. A similar paradox is re-emerging around "
    "generative AI: firms adopt it faster than ever, yet the measured productivity "
    "gains remain strikingly small."
)

c1, c2, c3 = st.columns(3)
c1.metric("Executives surveyed", "≈ 6,000", "Feb 2026 NBER study", delta_color="off")
c2.metric("Acemoglu TFP estimate", "≤ 0.66%", "cumulative over 10 years", delta_color="off")
c3.metric("U.S. firm AI adoption", "3.7% → ~10%", "2023 → 2025 (BTOS)", delta_color="off")

st.markdown(
    "<div class='caption-muted'>Adoption is racing ahead. Measured productivity is not. "
    "Why?</div>",
    unsafe_allow_html=True,
)


# ══ 3. RESEARCH QUESTION & CONTRIBUTION ═══════════════════════════════════════
st.header("Research Question & Contribution")
st.markdown(
    "Most explanations for the paradox invoke **measurement problems**, "
    "**complementary intangibles** (Brynjolfsson, Rock & Syverson 2021), or "
    "**organizational adjustment costs** (McElheran et al. 2025). "
)
st.markdown(
    "> **Can changes in task composition during AI diffusion, on their own, generate a "
    "temporary stagnation or decline in aggregate output?**"
)
st.markdown(
    "When adoption is driven by diffusion dynamics rather than realized productivity, it "
    "can mechanically push the automation frontier past the point where reallocating tasks "
    "to AI raises output. A CES aggregator with task complementarities then converts that "
    "overshoot into a decline in $Y$, even as AI's own productivity keeps rising."
)

st.info(
    "**To isolate the mechanism, the model deliberately assumes:**\n"
    "- Labor productivity is constant ($A_L = 1$) — no labor-side gains are doing the work.\n"
    "- AI adoption is **exogenous** — not optimized in response to realized productivity.\n"
    "- The task set is **fixed** — no new tasks, no reorganization.\n\n"
    "These assumptions strip away every channel the existing literature leans on. "
    "If a J-curve still appears, the mechanism stands on its own.",
	width="stretch"
)


# ══ PART 2 ════════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='part-banner'><span class='part-label'>Part 2 of 3</span>"
    "The Model &amp; Simulator</div>",
    unsafe_allow_html=True,
)

# ══ 4. THE MODEL ══════════════════════════════════════════════════════════════
st.header("The Model")
st.markdown(
    "Time is discrete, $t \\in \\{0, 1, \\ldots, T\\}$. Tasks are indexed by "
    "$z \\in [0, 1]$, where lower $z$ denotes tasks for which AI has a larger intrinsic "
    "comparative advantage. At each date an adoption threshold $I_t \\in [0, 1]$ "
    "partitions the continuum: tasks with $z \\le I_t$ are performed by AI capital, "
    "tasks with $z > I_t$ by labor."
)

st.subheader("1. Task-level output")
st.markdown(
    r"""
	> $$
	> y(z, t) = \begin{cases}
	> A_K(t)\,(1 - z)^2 & \text{if } z \le I_t \\[4pt]
	> A_L\, z^2 & \text{if } z > I_t
	> \end{cases}
	> $$
	"""
	"AI is most productive at **low** $z$, labor at **high** $z$."
	"The quadratic heterogeneity terms $\phi_K(z) = (1-z)^2$ and $\phi_L(z) = z^2$ give each task a meaningful comparative advantage, not just a marginal one."
)

st.subheader("2. AI productivity grows linearly")
st.markdown(
    r"""
	> $$
	> A_K(t) = 1 + g\,t
	> $$
	"""
	"A deliberately modest functional form. " 
	"Keeping AI's own productivity on a linear path ensures it **cannot** be what generates the downturn (any decline in $Y$ is caused by something else)."
)

st.subheader("3. Adoption follows a Bass diffusion process")
st.markdown(
    r"""
	> $$
	> I_{t+1} = I_t + (p + q\,I_t)(1 - I_t)
	> $$
	"""
	"Innovation ($p$) plus imitation ($q$) drive adoption in a logistic S-curve. " 
	"Crucially, this process is **independent of realized output,** so it captures " 
	"adoption driven by institutional pressure, investor expectations, and competitive imitation rather than productivity."
)

st.subheader("4. Tasks aggregate via a CES function")
st.markdown(
    r"""
	> $$
	> Y_t = \left[\int_0^1 y(z, t)^{\tfrac{\sigma - 1}{\sigma}}\,dz\right]^{\tfrac{\sigma}{\sigma - 1}}
	> $$
	"""
)
st.markdown(
	"The elasticity $\sigma$ controls how forgiving the aggregate is to compositional imbalance: \n"
	"- $\sigma < 1$: **gross complements** — a shortfall on any one task hurts the whole. \n"
	"- $\sigma > 1$: **substitutes** — gains elsewhere can paper over weak tasks."
)

st.subheader("5. The productivity-consistent frontier (diagnostic)")
st.markdown(
    r"""
	> $$
	> z^*(t) = \frac{\sqrt{1 + g\,t}}{1 + \sqrt{1 + g\,t}}
	> $$
	"""
	"Solves $A_K(t)(1-z)^2 = z^2$ — the threshold at which AI and labor deliver identical task output. "
	"A planner optimizing realized productivity would set $I_t = z^*(t)$. "
	"**When Bass diffusion pushes $I_t$ past $z^*(t)$, output has overshot.**"
)


# ══ 5. CALIBRATION ════════════════════════════════════════════════════════════
st.header("Calibration")
st.markdown(
    "The parameter ranges below are based on empirical or "
    "theoretical literature, so the simulator stays inside economically credible territory."
)

calib = pd.DataFrame(
    [
        ["p",  "Bass innovation",        "0.001 - 0.03", "Sultan, Farley & Lehmann (1990)"],
        ["q",  "Bass imitation",         "0.20 - 0.50",  "Sultan, Farley & Lehmann (1990)"],
        ["σ",  "CES elasticity",         "0.5 - 2.0",    "Oberfield & Raval (NBER w20626, 2014)"],
        ["I₀", "Initial AI adoption",    "0.05 - 0.10",  "BTOS — Bonney et al. (2024); Goldschlag (2025)"],
        ["g",  "AI productivity growth", "0.02 - 0.10",  "Set weak to isolate the mechanism"],
    ],
    columns=["Parameter", "Significance", "Plausible range", "Source"],
)
st.table(calib.set_index("Parameter"))


# ══ 6. INTERACTIVE SIMULATOR ══════════════════════════════════════════════════
st.header("Interactive Simulator")
st.markdown(
    "Adjust the sliders in the sidebar and watch $Y_t$ re-shape in real time. "
    "At the default calibration, the J-curve emerges within the 20-year horizon."
)

# Diagnostic markers
peak_idx = int(np.argmax(Y))
peak_year = years[peak_idx]
overshoot_mask = I > z_star
first_crossing = int(np.argmax(overshoot_mask)) if overshoot_mask.any() else None
crossing_year = years[first_crossing] if first_crossing is not None else None

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Adoption path  I(t)",
        "Aggregate output  Y(t)",
        "AI productivity  A_K(t)",
        "Adoption vs. productivity frontier",
    ),
    vertical_spacing=0.28,
    horizontal_spacing=0.14,
    x_title=None,
    y_title=None,
)

fig.add_trace(go.Scatter(x=years, y=I,   mode="lines", name="I(t)",
                         line=dict(color="#3b82f6", width=2.6)), row=1, col=1)
fig.add_trace(go.Scatter(x=years, y=Y,   mode="lines", name="Y(t)",
                         line=dict(color="#10b981", width=2.6)), row=1, col=2)
fig.add_trace(go.Scatter(x=years, y=A_K, mode="lines", name="A_K(t)",
                         line=dict(color="#a855f7", width=2.6)), row=2, col=1)
fig.add_trace(go.Scatter(x=years, y=I,   mode="lines", name="I(t)",
                         line=dict(color="#3b82f6", width=2.6),
                         showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=years, y=z_star, mode="lines", name="z*(t) frontier",
                         line=dict(color="#ef4444", width=2.6, dash="dash")), row=2, col=2)

# Peak marker on Y(t)
fig.add_trace(
    go.Scatter(
        x=[peak_year], y=[Y[peak_idx]],
        mode="markers",
        marker=dict(size=11, color="#10b981", line=dict(color="white", width=2)),
        name="peak", showlegend=False,
        hovertemplate=f"Peak: {peak_year}<br>Y = {Y[peak_idx]:.3f}<extra></extra>",
    ),
    row=1, col=2,
)
# Peak label — pinned below the marker so it never collides with the subplot title
fig.add_annotation(
    x=peak_year, y=Y[peak_idx],
    xref="x2", yref="y2",
    text=f"peak · {peak_year}",
    showarrow=False,
    yshift=-16,
    font=dict(family="MS Sans Serif, Tahoma, sans-serif", size=10, color="#065f46"),
    bgcolor="rgba(255,255,255,0.85)",
    borderpad=2,
)

# Crossing annotation on I vs z*  (drawn as scatter to keep subplot-scoped)
if crossing_year is not None:
    ymax_panel = float(max(I.max(), z_star.max())) * 1.05
    fig.add_trace(
        go.Scatter(
            x=[crossing_year, crossing_year],
            y=[0, ymax_panel],
            mode="lines",
            line=dict(color="#6b7280", width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ),
        row=2, col=2,
    )
    fig.add_annotation(
        x=crossing_year, y=ymax_panel * 0.95,
        xref="x4", yref="y4",
        text=f"crossing · {crossing_year}",
        showarrow=False,
        xshift=4, xanchor="left",
        font=dict(family="MS Sans Serif, Tahoma, sans-serif", size=10, color="#374151"),
        bgcolor="rgba(255,255,255,0.85)",
        borderpad=2,
    )

# Axis titles (per-panel) — added before layout-level font overrides so they inherit cleanly
fig.update_xaxes(title_text="Year", row=1, col=1)
fig.update_xaxes(title_text="Year", row=1, col=2)
fig.update_xaxes(title_text="Year", row=2, col=1)
fig.update_xaxes(title_text="Year", row=2, col=2)
fig.update_yaxes(title_text="Automation share", row=1, col=1)
fig.update_yaxes(title_text="Output Y", row=1, col=2)
fig.update_yaxes(title_text="Productivity A_K", row=2, col=1)
fig.update_yaxes(title_text="I(t)  &  z*(t)", row=2, col=2)

fig.update_layout(
    height=740,
    paper_bgcolor="#c0c0c0",
    plot_bgcolor="#ffffff",
    margin=dict(t=60, b=80, l=60, r=30),
    legend=dict(
        orientation="h", y=-0.12, x=0.5, xanchor="center",
        bgcolor="#c0c0c0", bordercolor="#808080", borderwidth=1,
        font=dict(family="MS Sans Serif, Tahoma, sans-serif", size=11, color="#000000"),
    ),
    font=dict(family="MS Sans Serif, Tahoma, sans-serif", size=11, color="#000000"),
)
_axis_tick_font = dict(family="MS Sans Serif, Tahoma, sans-serif", size=10, color="#000000")
_axis_title_font = dict(family="MS Sans Serif, Tahoma, sans-serif", size=11, color="#000000")
fig.update_xaxes(
    gridcolor="#dfdfdf", linecolor="#808080", zerolinecolor="#808080",
    tickfont=_axis_tick_font, title_font=_axis_title_font, title_standoff=8,
)
fig.update_yaxes(
    gridcolor="#dfdfdf", linecolor="#808080", zerolinecolor="#808080",
    tickfont=_axis_tick_font, title_font=_axis_title_font, title_standoff=6,
)
# Style only the subplot titles (not axis titles — keep those black/smaller)
_subplot_titles = {
    "Adoption path  I(t)",
    "Aggregate output  Y(t)",
    "AI productivity  A_K(t)",
    "Adoption vs. productivity frontier",
}
for ann in fig.layout.annotations:
    if ann.text in _subplot_titles:
        ann.font = dict(family="MS Sans Serif, Tahoma, sans-serif", size=13, color="#000080")
st.plotly_chart(fig, use_container_width=True)

# Diagnostic callout
if crossing_year is not None:
    st.warning(
        f"**Overshoot detected at {crossing_year}.** Adoption "
        f"$I = {I[first_crossing]:.3f}$ exceeds the productivity-consistent frontier "
        f"$z^* = {z_star[first_crossing]:.3f}$. From here, the economy slides down the "
        f"descending arm of the J-curve, even though $A_K(t)$ is still rising."
    )
else:
    st.success(
        "No overshoot in this run: adoption stays at or below the productivity-consistent "
        "frontier across the whole horizon. Try raising $q$ or $T$ to trigger the paradox."
    )

# End-of-period summary
st.subheader("End-of-period Summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Automation share I(T)",  f"{I[-1]:.3f}",      f"{I[-1]   - I[0]:+.3f}")
m2.metric("Aggregate output Y(T)",  f"{Y[-1]:.3f}",      f"{Y[-1]   - Y[0]:+.3f}")
m3.metric("AI productivity A_K(T)", f"{A_K[-1]:.3f}",    f"{A_K[-1] - A_K[0]:+.3f}")
m4.metric("Frontier z*(T)",         f"{z_star[-1]:.3f}", f"{z_star[-1] - z_star[0]:+.3f}")


# ══ PART 3 ════════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='part-banner'><span class='part-label'>Part 3 of 3</span>"
    "Results &amp; Implications</div>",
    unsafe_allow_html=True,
)

# ══ 7. COMPARATIVE STATICS ════════════════════════════════════════════════════
st.header("What the Sliders Teach")
st.markdown(
    "The J-curve is **generic** within the literature's parameter ranges — under any "
    "plausible $(\\sigma, q, g)$, $Y_t$ rises, peaks, and falls. Three dials shape "
    "*how* it falls."
)

with st.container(border=True):
    st.markdown("### σ — depth of the decline")
    st.markdown(
        "When tasks are gross complements ($\\sigma < 1$), shortfalls on high-$z$ tasks "
        "cannot be papered over by AI gains elsewhere, so the post-peak fall is sharp. "
        "When tasks are substitutes ($\\sigma > 1$), the decline is muted but does not "
        "disappear."
    )

with st.container(border=True):
    st.markdown("### q — timing of the peak")
    st.markdown(
        "Faster imitation pulls the frontier-crossing forward in time, shortens the "
        "rising arm, and deepens the fall within the fixed horizon. High-$q$ diffusion "
        "is what gets the economy into trouble early."
    )

with st.container(border=False):
    st.markdown("### g — level, not shape")
    st.markdown(
        "Higher AI-productivity growth lifts the whole $Y_t$ path and pushes $z^*(t)$ "
        "slightly outward, but because $g$ enters the frontier only under a square root "
        "while Bass diffusion drives $I_t$ toward $1$, faster AI cannot save you from the "
        "crossing."
    )


# ══ 8. SIGNIFICANCE & IMPACT ══════════════════════════════════════════════════
st.header("Significance & Impact")
st.markdown(
    "The headline is straightforward: **The AI productivity paradox does not require any "
    "of the channels emphasized in the existing literature.** No measurement error. "
    "No unmeasured intangibles. No organizational adjustment costs. Not even a slowing "
    "of AI's own productivity growth."
)
st.markdown(
    "The mechanism — exogenous diffusion crosses the "
    "productivity-consistent frontier, filtered through a CES aggregator with task "
    "complementarities — is enough to produce a temporary decline in $Y$."
)
st.markdown(
    "**Policy implication.** If observed firm-level adoption is already running ahead of "
    "plausible task-level productivity gains (as the BTOS data and the NBER CEO survey "
    "both suggest), further exogenous diffusion without complementary gains in $A_K(t)$ "
    "moves the economy along the **downward**, not the upward, arm of the curve. "
    "Encouraging adoption for its own sake may be precisely the wrong prescription."
)
st.markdown(
    "<div class='caption-muted'>Disclaimer: The model should be read as isolating a theoretical "
    "mechanism, not as forecasting aggregate dynamics. The same compositional logic, "
    "combined with the intangibles and adjustment-cost channels from the existing "
    "literature, gives a richer picture of what a real AI J-curve might look like.</div>",
    unsafe_allow_html=True,
)


# ══ 9. REFERENCES ═════════════════════════════════════════════════════════════
with st.expander("References", expanded=False):
    st.markdown(
        """
- **Acemoglu, D.** (2024). *The Simple Macroeconomics of AI.* NBER Working Paper 32487. Published (2025) in *Economic Policy* 40(121): 13-58.
- **Acemoglu, D., & Johnson, S.** (2023). *Rebalancing AI.* Finance & Development, IMF, December.
- **Bass, F. M.** (1969). *A New Product Growth Model for Consumer Durables.* Management Science 15(5): 215-227.
- **Bonney, K., Breaux, C., Buffington, C., Dinlersoz, E., Foster, L., Goldschlag, N., Haltiwanger, J., Kroff, Z., & Savage, K.** (2024). *Tracking Firm Use of AI in Real Time: A Snapshot from the Business Trends and Outlook Survey.* U.S. Census Bureau, CES Working Paper 24-16.
- **Brynjolfsson, E., Rock, D., & Syverson, C.** (2021). *The Productivity J-Curve: How Intangibles Complement General Purpose Technologies.* American Economic Journal: Macroeconomics 13(1): 333-372.
- **Federal Reserve** (2026, April 3). *Monitoring AI Adoption in the U.S. Economy.* FEDS Notes.
- **Fortune** (2026, February 17). *Thousands of CEOs just admitted AI had no impact on employment or productivity.*
- **Goldman Sachs** (2023). *Generative AI could raise global GDP by 7%.* Goldman Sachs Research.
- **Goldschlag, N.** (2025, December 19). *How Many Businesses Are Using AI?* Agglomerations (Substack).
- **McElheran, K., Yang, M.-J., Kroff, Z., & Brynjolfsson, E.** (2025). *The Rise of Industrial AI in America: Microfoundations of the Productivity J-curve(s).* U.S. Census Bureau, CES Working Paper 25-27.
- **Oberfield, E., & Raval, D.** (2014). *Micro Data and Macro Technology.* NBER Working Paper 20626.
- **Sultan, F., Farley, J. U., & Lehmann, D. R.** (1990). *A Meta-Analysis of Applications of Diffusion Models.* Journal of Marketing Research 27(1): 70-77.
        """
    )
