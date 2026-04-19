"""
AI Labor Market Diffusion Model
Discrete-time task-allocation model with Bass diffusion and CES aggregation.
"""

import numpy as np


def run_simulation(
    T: int,
    I0: float,
    p: float,
    q: float,
    g: float,
    sigma: float,
    N: int = 1000,
) -> dict:
    """
    Parameters
    ----------
    T     : time horizon (number of periods)
    I0    : initial share of tasks automated, in [0, 1]
    p     : Bass innovation parameter
    q     : Bass imitation parameter
    g     : linear AI productivity growth rate
    sigma : CES elasticity of substitution across tasks
    N     : task grid resolution (default 1000)

    Returns
    -------
    dict with keys:
        "t"      : array of time indices 0..T
        "I"      : adoption path I_t
        "A_K"    : AI productivity path A_K(t) = 1 + g*t
        "Y"      : aggregate output path Y_t
        "z_star" : productivity-consistent frontier z*(t)  [diagnostic]
    """
    # ── Task grid (midpoint rule) ────────────────────────────────────────────
    i_idx = np.arange(N)
    z = (i_idx + 0.5) / N          # z_i = (i + 0.5) / N
    phi_K = (1 - z) ** 2           # AI task heterogeneity
    phi_L = z ** 2                  # Labor task heterogeneity

    rho = (sigma - 1) / sigma       # CES exponent; 1/rho = sigma/(sigma-1)

    # ── Pre-allocate output arrays ───────────────────────────────────────────
    t_arr   = np.arange(T + 1, dtype=float)
    I_arr   = np.empty(T + 1)
    A_K_arr = np.empty(T + 1)
    Y_arr   = np.empty(T + 1)

    # ── Simulate period by period ────────────────────────────────────────────
    I_t = float(I0)
    for t in range(T + 1):
        I_arr[t]   = I_t
        A_K_t      = 1.0 + g * t
        A_K_arr[t] = A_K_t

        # Task-level output
        mask = z <= I_t
        y_t = np.where(mask, A_K_t * phi_K, phi_L)

        # CES aggregate  Y = (mean(y^rho))^(1/rho)
        Y_arr[t] = (np.mean(y_t ** rho)) ** (1.0 / rho)

        # Bass diffusion step (don't update I on the last period)
        if t < T:
            I_t = I_t + (p + q * I_t) * (1.0 - I_t)
            I_t = float(np.clip(I_t, 0.0, 1.0))

    # ── Diagnostic: productivity-consistent frontier ─────────────────────────
    # Solves (1 + g*t)*(1-z)^2 = z^2  =>  z*(t) = sqrt(1+g*t) / (1 + sqrt(1+g*t))
    z_star = np.sqrt(1.0 + g * t_arr) / (1.0 + np.sqrt(1.0 + g * t_arr))

    return {
        "t":      t_arr,
        "I":      I_arr,
        "A_K":    A_K_arr,
        "Y":      Y_arr,
        "z_star": z_star,
    }
