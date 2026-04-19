# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

UVA ECON 4444 ("Artificial Intelligence and the Future of Work") group project. A Streamlit web app that simulates AI diffusion into the labor market over 10–20 years. Users adjust parameters via sliders and see results as interactive Plotly charts.

## Setup and Running

```bash
# First time only — create the virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run the app (every time)
.venv/bin/streamlit run app.py
```

The app opens automatically in the browser at `http://localhost:8501`.

## Architecture

Two files:

- **`model.py`** — simulation logic only. `run_simulation()` takes parameters and returns a dict of NumPy arrays (one value per year). All economics equations go here.
- **`app.py`** — Streamlit UI. Reads sliders from the sidebar, calls `run_simulation()`, and renders Plotly charts and summary metrics.

## Adding the Real Equations

The placeholder equations in `model.py` are clearly marked with `# TODO`. Replace them with the project leader's equations. The function signature and return dict shape should stay the same unless new output variables are added — in which case, add matching charts in `app.py`.

## Parameters

Current sliders: diffusion speed, substitution intensity, complementarity intensity, time horizon (10 or 20 years). Add new parameters to the `run_simulation()` signature in `model.py` and add a matching `st.sidebar.slider()` in `app.py`.
