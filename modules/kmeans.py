"""
K-Means Clustering Module — From-Scratch Implementation
Demonstrates centroid convergence in unsupervised learning.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import make_blobs


# ──────────────────────────────────────────────
# Core K-Means (from scratch)
# ──────────────────────────────────────────────

def kmeans_from_scratch(X, K, max_iters=100, tol=1e-6, seed=42):
    """
    K-Means clustering implemented from scratch.
    Returns labels, final centroids, and per-iteration history.
    """
    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]

    # Random initialisation: pick K data points
    indices = rng.choice(n_samples, K, replace=False)
    centroids = X[indices].copy()

    history = {
        "centroids": [centroids.copy()],
        "inertia": [],
        "centroid_shift": [],
    }

    for _ in range(max_iters):
        # Assign each point to nearest centroid
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)

        # Compute inertia (sum of squared distances to assigned centroid)
        inertia = sum(np.sum((X[labels == k] - centroids[k]) ** 2)
                      for k in range(K))
        history["inertia"].append(inertia)

        # Update centroids
        new_centroids = np.array([X[labels == k].mean(axis=0)
                                  if np.any(labels == k) else centroids[k]
                                  for k in range(K)])

        shift = np.linalg.norm(new_centroids - centroids)
        history["centroid_shift"].append(shift)
        history["centroids"].append(new_centroids.copy())

        centroids = new_centroids

        if shift < tol:
            break

    return labels, centroids, history


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────

PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA",
           "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"]


def render():
    st.markdown("## 🟡 K-Means Clustering — Centroid Convergence")

    with st.expander("📖 Theory — Convergence in K-Means", expanded=False):
        st.markdown(r"""
**K-Means** minimises the Within-Cluster Sum of Squares (inertia):

$$J = \sum_{k=1}^{K}\sum_{x \in C_k} \|x - \mu_k\|^2$$

**Algorithm:**
1. Initialise K centroids randomly.
2. **Assign** each point to the nearest centroid.
3. **Update** each centroid to the mean of its assigned points.
4. Repeat until centroids stop moving (shift → 0).

**Convergence is guaranteed** because inertia decreases monotonically at every step.
However, the algorithm may converge to a **local minimum** depending on initialisation.
""")

    # ---- Sidebar ----
    st.sidebar.markdown("### 🟡 K-Means")
    K = st.sidebar.slider("Number of Clusters (K)", 2, 8, 3, key="k_km")
    dataset = st.sidebar.selectbox("Dataset", ["Blobs", "Random"],
                                   key="ds_km")
    n_samples = st.sidebar.slider("Samples", 100, 1000, 300, 50,
                                  key="ns_km")
    max_iters = st.sidebar.slider("Max Iterations", 5, 100, 30, 5,
                                  key="it_km")

    # ---- Generate data ----
    if dataset == "Blobs":
        X, _ = make_blobs(n_samples=n_samples, centers=K,
                          cluster_std=1.2, random_state=42)
    else:
        rng = np.random.RandomState(42)
        X = rng.randn(n_samples, 2) * 5

    # ---- Run K-Means ----
    labels, centroids, hist = kmeans_from_scratch(X, K, max_iters)

    # ---- Metrics ----
    c1, c2, c3 = st.columns(3)
    c1.metric("Final Inertia", f"{hist['inertia'][-1]:.2f}")
    c2.metric("Iterations", len(hist["inertia"]))
    c3.metric("Final Shift", f"{hist['centroid_shift'][-1]:.2e}")

    # ---- Cluster scatter + centroid trail ----
    fig = go.Figure()
    for k in range(K):
        mask = labels == k
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1], mode="markers",
            marker=dict(color=PALETTE[k % len(PALETTE)], size=5, opacity=0.6),
            name=f"Cluster {k+1}"))

    # Centroid movement trail
    all_c = np.array(hist["centroids"])  # shape (steps+1, K, 2)
    for k in range(K):
        trail = all_c[:, k, :]
        fig.add_trace(go.Scatter(
            x=trail[:, 0], y=trail[:, 1], mode="lines+markers",
            line=dict(color=PALETTE[k % len(PALETTE)], width=2, dash="dot"),
            marker=dict(size=8, symbol="x"),
            name=f"Centroid {k+1} path"))

    # Final centroids
    fig.add_trace(go.Scatter(
        x=centroids[:, 0], y=centroids[:, 1], mode="markers",
        marker=dict(color="white", size=14, symbol="star",
                    line=dict(width=2, color="black")),
        name="Final Centroids"))

    fig.update_layout(height=500, template="plotly_dark",
                      xaxis_title="X₁", yaxis_title="X₂",
                      margin=dict(t=20, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # ---- Convergence curves ----
    from plotly.subplots import make_subplots
    fig2 = make_subplots(rows=1, cols=2,
                         subplot_titles=("Inertia vs Iteration",
                                         "Centroid Shift vs Iteration"))
    fig2.add_trace(go.Scatter(y=hist["inertia"], mode="lines+markers",
                              line=dict(color="#FFA15A", width=2),
                              name="Inertia"), row=1, col=1)
    fig2.add_trace(go.Scatter(y=hist["centroid_shift"], mode="lines+markers",
                              line=dict(color="#AB63FA", width=2),
                              name="Shift"), row=1, col=2)
    fig2.update_layout(height=350, template="plotly_dark",
                       margin=dict(t=40, b=30))
    fig2.update_xaxes(title_text="Iteration")
    st.plotly_chart(fig2, use_container_width=True)

    # ---- Step-by-step animation slider ----
    st.markdown("### 🎞️ Step-by-Step Centroid Animation")
    step = st.slider("Iteration Step", 0, len(hist["centroids"]) - 1, 0,
                     key="step_km")

    fig3 = go.Figure()
    # Show all data in grey
    fig3.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1], mode="markers",
        marker=dict(color="grey", size=4, opacity=0.4), name="Data"))

    step_centroids = hist["centroids"][step]
    for k in range(K):
        fig3.add_trace(go.Scatter(
            x=[step_centroids[k, 0]], y=[step_centroids[k, 1]],
            mode="markers",
            marker=dict(color=PALETTE[k % len(PALETTE)], size=16,
                        symbol="diamond", line=dict(width=2, color="white")),
            name=f"C{k+1} @ step {step}"))

    fig3.update_layout(height=400, template="plotly_dark",
                       title=f"Centroids at Iteration {step}",
                       margin=dict(t=40, b=30))
    st.plotly_chart(fig3, use_container_width=True)
