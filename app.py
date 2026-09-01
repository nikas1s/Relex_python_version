"""Streamlit frontend for py-relex."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from relex.core import run_relex
from relex.io import normalize_input


ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "examples"


st.set_page_config(page_title="py-relex", layout="wide")
st.title("py-relex — Relativistic Coulomb Excitation")
st.caption(
    "Python implementation of C. A. Bertulani's RELEX model, maintained by Stylianos Nikas; "
    "earlier Python adaptation work by Emma Rice is acknowledged."
)

st.sidebar.header("Input")
example_files = sorted([p for p in EXAMPLES_DIR.glob("*.json")])
example_names = [p.name for p in example_files]

selection_mode = st.sidebar.radio("Load input", ["Example", "Upload", "Paste"], index=0)

input_json_text = ""
base_dir = ROOT

if selection_mode == "Example" and example_files:
    selected = st.sidebar.selectbox("Example file", example_names)
    example_path = EXAMPLES_DIR / selected
    input_json_text = example_path.read_text(encoding="utf-8")
    base_dir = example_path.parent
elif selection_mode == "Upload":
    uploaded = st.sidebar.file_uploader("Upload JSON", type=["json"])
    if uploaded is not None:
        input_json_text = uploaded.read().decode("utf-8")
        base_dir = ROOT
else:
    input_json_text = st.sidebar.text_area("Paste JSON", height=200, value="")
    base_dir = ROOT

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run RELEX")
plot_scale = st.sidebar.selectbox("Plot scale", ["Linear", "Log (y)"], index=0)

col_left, col_right = st.columns([3, 1], gap="large")

with col_left:
    st.subheader("Input JSON")
    input_json_text = st.text_area("", value=input_json_text, height=320)

with col_right:
    st.subheader("ℹ️ Input Guide")
    st.markdown(
        "\n".join(
            [
                "**What this solves**",
                "Relativistic Coulomb + nuclear excitation in heavy-ion collisions.",
                "It solves coupled-channels time‑dependent equations for E1, E2, and M1 modes",
                "(E0 only via nuclear excitation), as supported by the published RELEX model.",
                "",
                "**Key inputs**",
                "- `system`: projectile/target masses & charges, energy per nucleon, excitation side.",
                "- `integration`: impact-parameter mesh and ODE accuracy.",
                "- `options`: optical potential (`iopw`) and nuclear excitation (`iopnuc`).",
                "- `states`: level energies and spins.",
                "- `matrix_elements`: reduced E1/E2/M1 matrix elements for transitions.",
                "",
                "**Optional**",
                "- `optical_potential`: file or arrays when `iopw=1`.",
                "- `nuclear_deformation`: `delte0/1/2` when `iopnuc=1`.",
            ]
        )
    )

if run_button:
    with st.spinner("Running solver..."):
        try:
            data = json.loads(input_json_text)
            inputs = normalize_input(data, base_dir=base_dir)
            result = run_relex(inputs)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error: {exc}")
        else:
            st.session_state["last_result"] = result
            st.session_state["last_inputs"] = inputs
            st.success("Done")

result = st.session_state.get("last_result")
inputs = st.session_state.get("last_inputs")
if result is not None and inputs is not None:
    # Cross sections table
    st.subheader("Cross sections (mb)")
    cross_df = pd.DataFrame(
        [{"state": int(k), "cross_section_mb": float(v)} for k, v in result.cross_sections.items()]
    )
    st.dataframe(cross_df, use_container_width=True)

    # Spin-averaged probabilities vs b
    st.subheader("Spin-averaged probabilities vs impact parameter")
    b = result.b
    spinave = result.spinave
    chart_df = pd.DataFrame({"b": b})
    for j in range(spinave.shape[0]):
        chart_df[f"state_{j+1}"] = spinave[j, :]
    chart_df = chart_df.set_index("b")
    if plot_scale == "Log (y)":
        chart_df_plot = chart_df.replace(0.0, np.nan).reset_index().melt("b", var_name="state", value_name="prob")
        chart = (
            alt.Chart(chart_df_plot)
            .mark_line()
            .encode(
                x=alt.X("b:Q", title="b"),
                y=alt.Y("prob:Q", title="Probability", scale=alt.Scale(type="log")),
                color=alt.Color("state:N", title="State"),
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(chart_df, use_container_width=True)

    # Quick summary
    st.subheader("Summary")
    st.write(
        {
            "nb": int(inputs["integration"]["nb"]),
            "ngrid": int(inputs.get("grid", {}).get("ngrid", 200)),
            "nst": len(inputs["states"]),
            "iopw": int(inputs["options"].get("iopw", 0)),
            "iopnuc": int(inputs["options"].get("iopnuc", 0)),
        }
    )

    st.subheader("Download results")
    results_json = {
        "cross_sections": {int(k): float(v) for k, v in result.cross_sections.items()},
        "b": result.b.tolist(),
        "spinave": result.spinave.tolist(),
    }
    st.download_button(
        "Download results.json",
        data=json.dumps(results_json, indent=2),
        file_name="results.json",
        mime="application/json",
    )
    st.download_button(
        "Download cross_sections.csv",
        data=cross_df.to_csv(index=False),
        file_name="cross_sections.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download spinave.csv",
        data=chart_df.reset_index().to_csv(index=False),
        file_name="spinave.csv",
        mime="text/csv",
    )
