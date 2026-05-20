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

API_URL  = os.environ.get("API_URL",      "http://localhost:8000").rstrip("/")
_API_AUTH = (
    os.environ.get("API_USERNAME", "admin"),
    os.environ.get("API_PASSWORD", "mdp123"),
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FEM Surrogate Explorer",
    page_icon="⚙️",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚙️ FEM Surrogate Model — Interactive Explorer")
st.caption(
    "Predicts **maximum displacement** and **maximum von Mises stress** "
    "for traction-loaded plates using a LightGBM surrogate trained on FEM data."
)


# ── API utility functions ─────────────────────────────────────────────────────

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


# ── Sidebar: API status ───────────────────────────────────────────────────────
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

    # ── Input form ────────────────────────────────────────────────────────────
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
    young_gpa = st.slider("Young's modulus (GPa)", 60.0, 250.0, 210.0, 5.0)
    young_modulus_pa = young_gpa * 1e9
    poisson_ratio = st.slider("Poisson ratio", 0.20, 0.45, 0.30, 0.01)

    st.markdown("**Loading**")
    traction_mpa = st.slider("Applied traction (MPa)", 0.1, 2.2, 1.5, 0.1,
                             help="Max 2.2 MPa — upper bound of training data (steel)")
    traction_pa = traction_mpa * 1e6

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


# ── Build request payload ─────────────────────────────────────────────────────
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


# ── Analytical reference (physics) ───────────────────────────────────────────
def _analytical_estimate(p: dict) -> tuple[float, float]:
    """Quick physics-based estimate for sanity checking."""
    E = p["young_modulus_pa"]
    sig = p["traction_pa"]
    L   = p["length_m"]
    r   = p.get("hole_radius_ratio", 0.0) or 0.0
    H   = p["height_m"]

    eps = sig / E
    delta = eps * L

    d_over_W = min((2.0 * r * min(L, H)) / H, 0.95) if r > 0 else 0.0
    Kt = max(3.0 - 3.13 * d_over_W + 3.66 * d_over_W**2 - 1.53 * d_over_W**3, 1.0) if r > 0 else 1.0
    net_section = max(1.0 - d_over_W, 0.05)
    sigma_net = sig / net_section
    vm_estimate = Kt * sigma_net

    return delta, vm_estimate


# ── Main content ──────────────────────────────────────────────────────────────
tab_predict, tab_sweep, tab_about = st.tabs(["🔮 Prediction", "📊 Parameter Sweep", "ℹ️ About"])

with tab_predict:
    if predict_btn:
        with st.spinner("Running surrogate model ..."):
            t0 = time.perf_counter()
            result = _predict(payload)
            elapsed = (time.perf_counter() - t0) * 1000

        if result:
            preds = result["predictions"]
            disp  = preds["max_displacement_m"]
            vm    = preds["max_von_mises_pa"]
            ana_disp, ana_vm = _analytical_estimate(payload)

            st.success(f"Prediction completed in **{elapsed:.0f} ms**  |  Model: `{result['model_version']}`")

            col_disp, col_vm = st.columns(2)

            with col_disp:
                st.metric(
                    label="Max Displacement",
                    value=f"{disp:.4e} m",
                    delta=f"Analytical est: {ana_disp:.3e} m",
                    delta_color="off",
                )
                ratio_d = disp / ana_disp if ana_disp > 0 else float("nan")
                st.caption(f"Model / Analytical = {ratio_d:.3f}")

            with col_vm:
                st.metric(
                    label="Max von Mises Stress",
                    value=f"{vm:.4e} Pa  ({vm/1e6:.2f} MPa)",
                    delta=f"Analytical est: {ana_vm:.3e} Pa",
                    delta_color="off",
                )
                ratio_v = vm / ana_vm if ana_vm > 0 else float("nan")
                st.caption(f"Model / Analytical = {ratio_v:.3f}")

            # Physics summary
            st.divider()
            st.subheader("Physics context")
            eps = payload["traction_pa"] / payload["young_modulus_pa"]
            st.info(
                f"**Elastic strain** ε = σ/E = {eps:.3e}  |  "
                f"**Theoretical elongation** = ε×L = {eps * length_m:.4e} m"
            )

            # Show echo of input parameters
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

    _ranges = {
        "traction_pa":      (2e5, 2.2e6, 12),
        "young_modulus_pa": (6e10, 2.5e11, 12),
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
            v_vals = [v for v, ok in zip(vms, valid) if ok]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=d_vals, mode="lines+markers",
                                     name="Max Displacement (m)", yaxis="y1"))
            fig.add_trace(go.Scatter(x=x_vals, y=v_vals, mode="lines+markers",
                                     name="Max von Mises (Pa)", yaxis="y2"))
            fig.update_layout(
                title=f"Sensitivity to {sweep_param}",
                xaxis_title=sweep_param,
                yaxis=dict(title="Max Displacement (m)"),
                yaxis2=dict(title="Max von Mises (Pa)", overlaying="y", side="right"),
                legend=dict(x=0.01, y=0.99),
                height=450,
            )
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
