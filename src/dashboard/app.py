"""Streamlit dashboard — Interactive explorer for the FEM Surrogate model.

Features
--------
- Sidebar: input form for geometry and loading parameters
- Predict button: calls the FastAPI /predict endpoint
- Results panel: numerical predictions + colour-coded gauges
- Physical context: analytical estimate vs model prediction
- Parameter sweep: 1-D sensitivity graph for any parameter
- Model info: version and feature count from the /version endpoint

Local execution (with the API running on :8000)
------------------------------------------------
    streamlit run src/dashboard/app.py

Or set the API_URL variable:
    API_URL=http://api:8000 streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL  = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
_api_username = os.environ.get("API_USERNAME")
_api_password = os.environ.get("API_PASSWORD")
_API_AUTH = (_api_username or "", _api_password or "")

# Yield strength defaults per material (MPa)
_YIELD_DEFAULTS = {"steel": 250.0, "aluminum": 270.0, "titanium": 880.0}

# Max traction seen during training per material (MPa) — model predicts Kt_eff (normalised),
# so extrapolation is physically valid in the elastic regime, but flagged for transparency.
_TRACTION_TRAIN_MAX = {"steel": 2.2, "aluminum": 1.2, "titanium": 1.8}

# Material property ranges from training (MATERIAL_CATEGORIES in simulations)
# E in GPa, ν dimensionless — sliders are constrained to these physically correct ranges.
_MAT_PROPS = {
    "steel":    {"E_min": 190.0, "E_max": 230.0, "E_def": 210.0, "nu_min": 0.27, "nu_max": 0.31, "nu_def": 0.29},
    "aluminum": {"E_min":  65.0, "E_max":  78.0, "E_def":  70.0, "nu_min": 0.31, "nu_max": 0.35, "nu_def": 0.33},
    "titanium": {"E_min": 105.0, "E_max": 125.0, "E_def": 114.0, "nu_min": 0.30, "nu_max": 0.35, "nu_def": 0.32},
}

# Page configuration
st.set_page_config(
    page_title="FEM Surrogate Explorer",
    page_icon="⚙️",
    layout="wide",
)

# Guard — credentials must be set via environment variables
if not _api_username or not _api_password:
    st.error(
        "**Configuration manquante** : les variables d'environnement "
        "`API_USERNAME` et `API_PASSWORD` doivent être définies. "
        "Copiez `.env.example` vers `.env` et renseignez ces valeurs."
    )
    st.stop()

# Header
st.title("⚙️ FEM Surrogate Model — Interactive Explorer")
st.caption(
    "Predicts **maximum displacement** and **maximum von Mises stress** "
    "for traction-loaded plates using a LightGBM surrogate trained on FEM data."
)


# API utility functions

def _get_version() -> dict:
    try:
        r = requests.get(f"{API_URL}/version", auth=_API_AUTH, timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _get_health() -> dict:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"status": "unreachable", "model_loaded": False}


def _predict(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, auth=_API_AUTH, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None
    except requests.ConnectionError:
        st.error(f"Cannot reach API at `{API_URL}`. Is the API container running?")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# Sidebar: API status
with st.sidebar:
    st.subheader("🔌 API Status")
    health = _get_health()
    version_info = _get_version()
    if health.get("status") == "ok":
        st.success(f"API Online — model `{health.get('model_version', 'unknown')}`")
    elif health.get("status") == "degraded":
        st.warning("API degraded — model not loaded")
    else:
        st.error("API unreachable")

    if version_info:
        st.caption(
            f"API v{version_info.get('api_version', '?')} · "
            f"{version_info.get('feature_count', '?')} features"
        )

    st.divider()

    # Input form
    st.subheader("🔧 Simulation Parameters")

    geometry_type = st.selectbox(
        "Geometry type",
        ["with_hole", "without_hole", "with_hole_moving"],
        help="Plate geometry variant",
    )

    material_category = st.selectbox(
        "Material", ["steel", "aluminum", "titanium"],
        help="Used for model encoding (categorical feature)"
    )
    dimension_category = st.selectbox(
        "Dimension class", ["small", "medium", "large"],
        help="Used for model encoding (categorical feature)"
    )

    st.markdown("**Plate dimensions**")
    col1, col2 = st.columns(2)
    with col1:
        length_m = st.number_input("Length (m)", value=1.2, min_value=0.1, max_value=5.0, step=0.05)
    with col2:
        height_m = st.number_input("Height (m)", value=0.30, min_value=0.05, max_value=2.0, step=0.01)

    st.markdown("**Material properties**")
    _mp = _MAT_PROPS[material_category]
    young_gpa = st.slider(
        "Young's modulus (GPa)",
        _mp["E_min"], _mp["E_max"], _mp["E_def"], 1.0,
        help=f"Plage physique {material_category} : {_mp['E_min']}–{_mp['E_max']} GPa",
    )
    young_modulus_pa = young_gpa * 1e9
    poisson_ratio = st.slider(
        "Poisson ratio",
        _mp["nu_min"], _mp["nu_max"], _mp["nu_def"], 0.01,
        help=f"Plage physique {material_category} : {_mp['nu_min']}–{_mp['nu_max']}",
    )

    default_yield = _YIELD_DEFAULTS.get(material_category, 250.0)
    yield_mpa = st.number_input(
        "Yield strength (MPa)",
        value=default_yield,
        min_value=50.0, max_value=2000.0, step=10.0,
        help="Used to compute safety factor. Default set per material.",
    )
    yield_pa = yield_mpa * 1e6

    st.markdown("**Loading**")
    traction_mpa = st.number_input(
        "Applied traction (MPa)", value=1.5, min_value=0.01, step=0.1,
        help="No upper limit — model predicts Kt_eff (normalised ratio), valid for any elastic traction.",
    )
    traction_pa = traction_mpa * 1e6
    _train_max = _TRACTION_TRAIN_MAX.get(material_category, 2.2)
    if traction_mpa > _train_max:
        st.warning(
            f"⚠️ {traction_mpa:.2f} MPa dépasse la plage d'entraînement "
            f"({material_category} : max {_train_max} MPa). "
            "Valide en régime élastique linéaire (Kt_eff normalisé).",
            icon=None,
        )

    hole_radius_ratio = None
    hole_cx_ratio = None
    hole_cy_ratio = None

    if geometry_type in ("with_hole", "with_hole_moving"):
        st.markdown("**Hole parameters**")
        hole_radius_ratio = st.slider("Hole radius ratio", 0.01, 0.45, 0.10, 0.01,
                                      help="Fraction of min(L, H)")
    if geometry_type == "with_hole_moving":
        col_cx, col_cy = st.columns(2)
        with col_cx:
            hole_cx_ratio = st.slider("Hole cx ratio", 0.1, 0.9, 0.5, 0.05,
                                      help="Hole centre x as fraction of plate length")
        with col_cy:
            hole_cy_ratio = st.slider("Hole cy ratio", 0.1, 0.9, 0.5, 0.05,
                                      help="Hole centre y as fraction of plate height")

    predict_btn = st.button("▶ Predict", type="primary", use_container_width=True)


# Build request payload
payload: dict = {
    "geometry_type":    geometry_type,
    "length_m":         length_m,
    "height_m":         height_m,
    "young_modulus_pa": young_modulus_pa,
    "poisson_ratio":    poisson_ratio,
    "traction_pa":      traction_pa,
    "material_category": material_category,
    "dimension_category": dimension_category,
    "mesh_nx": 120,
    "mesh_ny": 24,
}
if hole_radius_ratio is not None:
    payload["hole_radius_ratio"] = hole_radius_ratio
if hole_cx_ratio is not None:
    payload["hole_cx_ratio"] = hole_cx_ratio
if hole_cy_ratio is not None:
    payload["hole_cy_ratio"] = hole_cy_ratio


# Analytical reference (physics)
def _analytical_estimate(p: dict) -> tuple[float, float, float]:
    """Quick physics-based estimate for sanity checking. Returns (delta, vm_estimate, Kt)."""
    E   = p["young_modulus_pa"]
    sig = p["traction_pa"]
    L   = p["length_m"]
    r   = p.get("hole_radius_ratio", 0.0) or 0.0
    H   = p["height_m"]

    eps   = sig / E
    delta = eps * L

    d_over_W = min((2.0 * r * min(L, H)) / H, 0.95) if r > 0 else 0.0
    Kt = (
        max(3.0 - 3.13 * d_over_W + 3.66 * d_over_W**2 - 1.53 * d_over_W**3, 1.0)
        if r > 0 else 1.0
    )
    net_section = max(1.0 - d_over_W, 0.05)
    sigma_net   = sig / net_section
    vm_estimate = Kt * sigma_net

    return delta, vm_estimate, Kt


def _safety_factor_gauge(sf: float) -> go.Figure:
    color = "green" if sf >= 2.0 else ("orange" if sf >= 1.0 else "red")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(sf, 2),
        number={"font": {"size": 36}},
        title={"text": "Safety Factor  (σ_y / σ_vm)", "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, max(5.0, sf * 1.2)], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0, 1.0],                        "color": "#ffcccc"},
                {"range": [1.0, 2.0],                      "color": "#ffe5b4"},
                {"range": [2.0, max(5.0, sf * 1.2)],       "color": "#d4edda"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 1.0,
            },
        },
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def _plate_sketch(L: float, H: float, r_ratio: float | None,
                  cx_ratio: float | None, cy_ratio: float | None) -> go.Figure:
    """Simple 2-D plate outline with hole and load arrows."""
    W, Hd = 6.0, 6.0 * H / L  # normalised display dimensions

    shapes = [
        dict(type="rect", x0=0, y0=0, x1=W, y1=Hd,
             line=dict(color="#1f77b4", width=2), fillcolor="rgba(31,119,180,0.08)"),
    ]
    annotations = []

    if r_ratio:
        cx = (cx_ratio or 0.5) * W
        cy = (cy_ratio or 0.5) * Hd
        radius = r_ratio * min(W, Hd)
        shapes.append(dict(
            type="circle",
            x0=cx - radius, y0=cy - radius,
            x1=cx + radius, y1=cy + radius,
            line=dict(color="#d62728", width=2),
            fillcolor="rgba(255,255,255,0.9)",
        ))

    # Load arrows (left & right)
    annotations += [
        dict(x=W + 0.7, y=Hd / 2, ax=W + 0.1, ay=Hd / 2,
             axref="x", ayref="y", text="σ",
             showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#e377c2"),
        dict(x=-0.7, y=Hd / 2, ax=-0.1, ay=Hd / 2,
             axref="x", ayref="y", text="σ",
             showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#e377c2"),
    ]

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(range=[-1, W + 1], visible=False, scaleanchor="y"),
        yaxis=dict(range=[-0.5, Hd + 0.5], visible=False),
        margin=dict(t=10, b=10, l=10, r=10),
        height=200,
        plot_bgcolor="white",
    )
    return fig


# Main content
tab_predict, tab_sweep, tab_about = st.tabs(["🔮 Prediction", "📊 Parameter Sweep", "ℹ️ About"])

with tab_predict:
    # Always show plate sketch
    with st.expander("Plate geometry", expanded=False):
        st.plotly_chart(
            _plate_sketch(length_m, height_m, hole_radius_ratio, hole_cx_ratio, hole_cy_ratio),
            use_container_width=True,
        )

    if predict_btn:
        with st.spinner("Running surrogate model ..."):
            t0 = time.perf_counter()
            result = _predict(payload)
            elapsed = (time.perf_counter() - t0) * 1000

        if result:
            preds   = result["predictions"]
            disp    = preds["max_displacement_m"]
            vm      = preds["max_von_mises_pa"]
            vm_mpa  = vm / 1e6
            disp_mm = disp * 1e3
            ana_disp, ana_vm, Kt = _analytical_estimate(payload)
            sf = yield_pa / vm if vm > 0 else float("inf")
            amp_ratio = vm_mpa / traction_mpa  # stress amplification

            st.success(f"Prediction completed in **{elapsed:.0f} ms**  |  Model: `{result['model_version']}`")

            # Row 1 — displacement
            st.markdown("#### Displacement")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Max Displacement", f"{disp:.4e} m",
                          delta=f"Analytical: {ana_disp:.3e} m", delta_color="off")
            with c2:
                st.metric("Max Displacement (mm)", f"{disp_mm:.4f} mm")
            with c3:
                ratio_d = disp / ana_disp if ana_disp > 0 else float("nan")
                st.metric("Model / Analytical", f"{ratio_d:.3f}")

            st.markdown("#### Stress")
            c4, c5, c6 = st.columns(3)
            with c4:
                st.metric("Max von Mises", f"{vm_mpa:.3f} MPa",
                          delta=f"Analytical: {ana_vm/1e6:.3f} MPa", delta_color="off")
            with c5:
                st.metric("Stress amplification  (σ_vm / σ_applied)", f"{amp_ratio:.2f}×")
            with c6:
                st.metric("Stress concentration Kt  (analytical)", f"{Kt:.3f}")

            # Row 3 — safety factor gauge + regime badge
            st.markdown("#### Safety assessment")
            g_col, b_col = st.columns([2, 1])
            with g_col:
                st.plotly_chart(_safety_factor_gauge(sf), use_container_width=True)
            with b_col:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if vm < yield_pa:
                    st.success(f"**Elastic regime**  \nσ_vm = {vm_mpa:.2f} MPa  \nσ_y = {yield_mpa:.0f} MPa")
                else:
                    st.error(f"**Plastic regime ⚠️**  \nσ_vm = {vm_mpa:.2f} MPa  \nσ_y = {yield_mpa:.0f} MPa  \n(surrogate trained in elastic range)")

            # Physics summary
            st.divider()
            st.subheader("Physics context")
            eps = payload["traction_pa"] / payload["young_modulus_pa"]
            st.info(
                f"**Elastic strain** ε = σ/E = {eps:.3e}  |  "
                f"**Theoretical elongation** = ε×L = {eps * length_m:.4e} m  |  "
                f"**Applied traction** = {traction_mpa:.2f} MPa"
            )

            with st.expander("Input parameters sent to API"):
                st.json(result["input_summary"])
    else:
        st.info("Configure parameters in the sidebar and click **▶ Predict**.")


with tab_sweep:
    st.subheader("1-D Parameter Sensitivity")
    sweep_param = st.selectbox(
        "Sweep parameter",
        ["traction_pa", "young_modulus_pa", "hole_radius_ratio", "length_m", "height_m"],
        format_func=lambda x: {
            "traction_pa": "Traction (Pa)",
            "young_modulus_pa": "Young's modulus (Pa)",
            "hole_radius_ratio": "Hole radius ratio",
            "length_m": "Plate length (m)",
            "height_m": "Plate height (m)",
        }.get(x, x),
    )

    show_sf = st.checkbox("Show safety factor curve", value=True)

    _mp_sweep = _MAT_PROPS[material_category]
    _ranges = {
        "traction_pa":      (2e5, 10e6, 12),
        "young_modulus_pa": (_mp_sweep["E_min"] * 1e9, _mp_sweep["E_max"] * 1e9, 12),
        "hole_radius_ratio": (0.02, 0.45, 12),
        "length_m":         (0.3, 3.0, 12),
        "height_m":         (0.05, 1.0, 12),
    }
    lo, hi, n_pts = _ranges.get(sweep_param, (0.1, 1.0, 10))

    run_sweep = st.button("Run sweep", use_container_width=False)
    if run_sweep:
        if sweep_param == "hole_radius_ratio" and geometry_type == "without_hole":
            st.warning("Cannot sweep hole_radius_ratio for 'without_hole' geometry.")
        else:
            sweep_values = np.linspace(lo, hi, n_pts)
            disps, vms = [], []
            bar = st.progress(0, text="Running sweep...")
            for i, v in enumerate(sweep_values):
                p = dict(payload)
                p[sweep_param] = float(v)
                if sweep_param == "hole_radius_ratio":
                    p["hole_radius_ratio"] = float(v)
                r = _predict(p)
                if r:
                    disps.append(r["predictions"]["max_displacement_m"])
                    vms.append(r["predictions"]["max_von_mises_pa"])
                else:
                    disps.append(float("nan"))
                    vms.append(float("nan"))
                bar.progress((i + 1) / n_pts, text=f"Point {i+1}/{n_pts}")
            bar.empty()

            valid = [not np.isnan(d) for d in disps]
            x_vals = sweep_values[valid]
            d_vals = [v for v, ok in zip(disps, valid) if ok]
            v_vals = [v for v, ok in zip(vms,   valid) if ok]
            sf_vals = [yield_pa / vm if vm > 0 else float("nan") for vm in v_vals]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=[d * 1e3 for d in d_vals],
                                     mode="lines+markers", name="Max Displacement (mm)", yaxis="y1"))
            fig.add_trace(go.Scatter(x=x_vals, y=[v / 1e6 for v in v_vals],
                                     mode="lines+markers", name="Max von Mises (MPa)", yaxis="y2"))
            if show_sf:
                fig.add_trace(go.Scatter(x=x_vals, y=sf_vals,
                                         mode="lines+markers", name="Safety Factor",
                                         line=dict(dash="dot", color="green"), yaxis="y3"))
                # Danger line at SF=1
                fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                              annotation_text="SF = 1 (yield)", yref="y3")

            yaxis3_cfg = dict(
                title="Safety Factor", overlaying="y", side="right",
                anchor="free", position=1.0, showgrid=False
            ) if show_sf else {}

            layout_kwargs: dict = dict(
                title=f"Sensitivity to {sweep_param}",
                xaxis_title=sweep_param,
                yaxis=dict(title="Max Displacement (mm)"),
                yaxis2=dict(title="Max von Mises (MPa)", overlaying="y", side="right", showgrid=False),
                legend=dict(x=0.01, y=0.99),
                height=450,
            )
            if show_sf:
                layout_kwargs["yaxis3"] = yaxis3_cfg

            fig.update_layout(**layout_kwargs)
            st.plotly_chart(fig, use_container_width=True)


with tab_about:
    st.markdown("""
## About this dashboard

This Streamlit interface connects to the **FEM Surrogate API** (FastAPI) to interactively
predict the structural response of traction-loaded plates.

### Model
- **Algorithm**: LightGBM with Optuna hyperparameter search
- **Targets**: max_displacement_m · max_von_mises_pa (both in log₁₀ space)
- **Features**: 42 physics-informed features (Peterson Kt, net section stress, ligament distances, ...)
- **Geometries**: with_hole · without_hole · with_hole_moving

### Displayed metrics
| Metric | Description |
|---|---|
| Max Displacement (m / mm) | Surrogate prediction |
| Max von Mises (MPa) | Surrogate prediction |
| Stress amplification | σ_vm / σ_applied — how much the geometry amplifies the load |
| Kt (analytical) | Peterson stress concentration factor |
| Safety Factor | σ_yield / σ_vm — distance from yielding |
| Regime (elastic / plastic) | Based on user-supplied yield strength |

### Architecture
```
Streamlit UI  ──►  FastAPI /predict  ──►  LightGBM model (loaded from registry)
                          │
                    /metrics  ──►  Prometheus  ──►  Grafana
```

### API Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Availability + model status |
| `/version` | GET | API and model version |
| `/predict` | POST | Inference on a single case |
| `/metrics` | GET | Prometheus metrics |
""")
