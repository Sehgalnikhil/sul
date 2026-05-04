"""
Comparison Dashboard — Side-by-side convergence analysis
Compares Logistic Regression, Linear Regression, and K-Means.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import make_classification, make_blobs

from modules.logistic_regression import train_logistic_regression
from modules.linear_regression import (generate_linear_data,
                                        train_linear_regression)
from modules.kmeans import kmeans_from_scratch


def render():
    st.markdown("## 📊 Comparison Dashboard")
    st.markdown("Compare convergence behaviour across all three algorithms "
                "under different hyper-parameter settings.")

    # ---- Sidebar ----
    st.sidebar.markdown("### 📊 Comparison")
    lr_val = st.sidebar.slider("Shared Learning Rate", 0.001, 5.0, 0.1,
                               0.01, key="lr_cmp")
    iters_val = st.sidebar.slider("Max Iterations", 50, 1000, 300, 50,
                                  key="it_cmp")
    scale = st.sidebar.checkbox("Feature Scaling", True, key="sc_cmp")

    # ===== Logistic Regression =====
    X_cls, y_cls = make_classification(n_samples=200, n_features=2,
                                       n_redundant=0, n_informative=2,
                                       n_clusters_per_class=1,
                                       random_state=42)
    if scale:
        X_cls = (X_cls - X_cls.mean(0)) / (X_cls.std(0) + 1e-8)
    _, _, hist_log = train_logistic_regression(X_cls, y_cls, lr_val,
                                               iters_val)

    # ===== Linear Regression =====
    X_reg, y_reg = generate_linear_data(200, 3.0)
    if scale:
        X_reg = (X_reg - X_reg.mean(0)) / (X_reg.std(0) + 1e-8)
    _, _, hist_lin = train_linear_regression(X_reg, y_reg, lr_val, iters_val)

    # ===== K-Means =====
    X_km, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.2,
                         random_state=42)
    if scale:
        X_km = (X_km - X_km.mean(0)) / (X_km.std(0) + 1e-8)
    _, _, hist_km = kmeans_from_scratch(X_km, 3, iters_val)

    # ---- Normalise losses to [0, 1] for comparison ----
    def norm(vals):
        arr = np.array(vals, dtype=float)
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-12)

    # ---- Side-by-side convergence ----
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Logistic — Log-Loss",
                                        "Linear — MSE",
                                        "K-Means — Inertia"))
    fig.add_trace(go.Scatter(y=hist_log["loss"], mode="lines",
                             line=dict(color="#636EFA", width=2),
                             name="Log-Loss"), row=1, col=1)
    fig.add_trace(go.Scatter(y=hist_lin["mse"], mode="lines",
                             line=dict(color="#00CC96", width=2),
                             name="MSE"), row=1, col=2)
    fig.add_trace(go.Scatter(y=hist_km["inertia"], mode="lines",
                             line=dict(color="#FFA15A", width=2),
                             name="Inertia"), row=1, col=3)
    fig.update_layout(height=380, template="plotly_dark",
                      margin=dict(t=50, b=30))
    fig.update_xaxes(title_text="Iteration")
    st.plotly_chart(fig, use_container_width=True)

    # ---- Normalised overlay ----
    st.markdown("### Normalised Convergence Overlay")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=norm(hist_log["loss"]), mode="lines",
                              line=dict(color="#636EFA", width=2),
                              name="Logistic"))
    fig2.add_trace(go.Scatter(y=norm(hist_lin["mse"]), mode="lines",
                              line=dict(color="#00CC96", width=2),
                              name="Linear"))
    fig2.add_trace(go.Scatter(y=norm(hist_km["inertia"]), mode="lines",
                              line=dict(color="#FFA15A", width=2),
                              name="K-Means"))
    fig2.update_layout(height=380, template="plotly_dark",
                       xaxis_title="Iteration",
                       yaxis_title="Normalised Cost (0–1)",
                       margin=dict(t=20, b=30))
    st.plotly_chart(fig2, use_container_width=True)

    # ---- Learning Rate Sensitivity ----
    st.markdown("### 🔥 Learning Rate Sensitivity")
    st.markdown("Watch how different learning rates affect convergence speed "
                "and stability for **Logistic Regression**.")

    test_lrs = [0.01, 0.1, 0.5, 1.0, 5.0]
    fig3 = go.Figure()
    for alr in test_lrs:
        _, _, h = train_logistic_regression(X_cls, y_cls, alr, 500)
        fig3.add_trace(go.Scatter(y=h["loss"], mode="lines", name=f"α={alr}"))
    fig3.update_layout(height=380, template="plotly_dark",
                       xaxis_title="Iteration", yaxis_title="Log-Loss",
                       title="Effect of Learning Rate on Convergence",
                       margin=dict(t=50, b=30))
    st.plotly_chart(fig3, use_container_width=True)

    # ---- Feature Scaling Impact ----
    st.markdown("### 📏 Feature Scaling Impact")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**With Scaling**")
        X_s = (X_cls - X_cls.mean(0)) / (X_cls.std(0) + 1e-8)
        _, _, hs = train_logistic_regression(X_s, y_cls, 0.5, 300)
        st.metric("Final Loss", f"{hs['loss'][-1]:.6f}")
        st.metric("Iters to converge", len(hs["loss"]))
    with col2:
        st.markdown("**Without Scaling**")
        X_raw, y_raw = make_classification(n_samples=200, n_features=2,
                                            n_redundant=0, n_informative=2,
                                            n_clusters_per_class=1,
                                            random_state=42)
        _, _, hr = train_logistic_regression(X_raw, y_raw, 0.5, 300)
        st.metric("Final Loss", f"{hr['loss'][-1]:.6f}")
        st.metric("Iters to converge", len(hr["loss"]))

    # ---- Summary table ----
    st.markdown("### 📋 Summary")
    st.table({
        "Algorithm": ["Logistic Reg", "Linear Reg", "K-Means"],
        "Cost Function": ["Log-Loss", "MSE", "Inertia (WCSS)"],
        "Iterations Run": [len(hist_log["loss"]), len(hist_lin["mse"]),
                           len(hist_km["inertia"])],
        "Final Cost": [f"{hist_log['loss'][-1]:.4f}",
                       f"{hist_lin['mse'][-1]:.4f}",
                       f"{hist_km['inertia'][-1]:.2f}"],
        "Converged": [
            hist_log["grad_norm"][-1] < 1e-4,
            hist_lin["grad_norm"][-1] < 1e-4,
            hist_km["centroid_shift"][-1] < 1e-4],
    })
