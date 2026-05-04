"""
Logistic Regression Module — Manual Gradient Descent Implementation
Demonstrates convergence of log-loss during supervised classification training.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import make_classification


# ──────────────────────────────────────────────
# Core Logistic Regression (from scratch)
# ──────────────────────────────────────────────

def sigmoid(z):
    """Numerically-stable sigmoid activation."""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


def compute_log_loss(X, y, weights, bias):
    """Binary cross-entropy loss."""
    m = len(y)
    z = X @ weights + bias
    h = sigmoid(z)
    eps = 1e-15  # avoid log(0)
    loss = -np.mean(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))
    return loss


def train_logistic_regression(X, y, lr, n_iters, early_stop_tol=1e-7):
    """
    Train logistic regression via gradient descent.
    Returns history of weights, biases, losses and gradient norms.
    """
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0.0

    history = {"loss": [], "grad_norm": [], "weights": [], "bias": []}

    for i in range(n_iters):
        z = X @ weights + bias
        h = sigmoid(z)
        error = h - y

        dw = (1 / m) * (X.T @ error)
        db = (1 / m) * np.sum(error)

        weights -= lr * dw
        bias -= lr * db

        loss = compute_log_loss(X, y, weights, bias)
        grad_norm = np.linalg.norm(dw)

        history["loss"].append(loss)
        history["grad_norm"].append(grad_norm)
        history["weights"].append(weights.copy())
        history["bias"].append(bias)

        # Early stopping
        if grad_norm < early_stop_tol:
            break

    return weights, bias, history


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────

def render():
    st.markdown("## 🔵 Logistic Regression — Gradient Descent Convergence")

    # ---- Theory ----
    with st.expander("📖 Theory — Convergence in Logistic Regression", expanded=False):
        st.markdown(r"""
**Logistic Regression** models the probability that a sample belongs to class 1:

$$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1+e^{-\theta^T x}}$$

**Log-Loss (Binary Cross-Entropy):**

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log h_\theta(x^{(i)}) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))\right]$$

**Gradient Descent Update:**

$$\theta := \theta - \alpha \nabla J(\theta)$$

**Convergence Conditions:**
1. The gradient norm ‖∇J‖ → 0
2. The loss stabilises between consecutive iterations
3. If learning rate α is too large, the loss **diverges** instead

> A well-chosen learning rate produces a smooth, monotonically decreasing loss curve.
""")

    # ---- Sidebar controls ----
    st.sidebar.markdown("### 🔵 Logistic Regression")
    lr = st.sidebar.slider("Learning Rate (α)", 0.001, 10.0, 0.5, 0.01,
                           key="lr_log")
    n_iters = st.sidebar.slider("Iterations", 10, 2000, 300, 10,
                                key="it_log")
    n_samples = st.sidebar.slider("Samples", 50, 500, 200, 50,
                                  key="ns_log")
    early_stop = st.sidebar.checkbox("Enable Early Stopping", True,
                                     key="es_log")
    tol = 1e-7 if early_stop else 0.0

    # ---- Generate data ----
    X, y = make_classification(n_samples=n_samples, n_features=2,
                               n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, random_state=42)
    # Standardise
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # ---- Train ----
    weights, bias, history = train_logistic_regression(X, y, lr, n_iters, tol)

    # ---- Metrics row ----
    c1, c2, c3 = st.columns(3)
    c1.metric("Final Loss", f"{history['loss'][-1]:.6f}")
    c2.metric("Final ‖∇‖", f"{history['grad_norm'][-1]:.2e}")
    c3.metric("Iters Run", len(history["loss"]))

    # ---- Loss & Gradient Norm plot ----
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Log-Loss vs Iterations",
                                        "Gradient Norm vs Iterations"))
    fig.add_trace(go.Scatter(y=history["loss"], mode="lines",
                             line=dict(color="#636EFA", width=2),
                             name="Loss"), row=1, col=1)
    fig.add_trace(go.Scatter(y=history["grad_norm"], mode="lines",
                             line=dict(color="#EF553B", width=2),
                             name="‖∇J‖"), row=1, col=2)
    fig.update_layout(height=380, template="plotly_dark",
                      margin=dict(t=40, b=30))
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Decision Boundary ----
    st.markdown("### Decision Boundary")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sigmoid(grid @ weights + bias).reshape(xx.shape)

    fig2 = go.Figure()
    fig2.add_trace(go.Contour(x=np.linspace(x_min, x_max, 200),
                              y=np.linspace(y_min, y_max, 200),
                              z=probs, colorscale="RdBu", opacity=0.6,
                              showscale=False,
                              contours=dict(start=0.5, end=0.5, size=0.1)))
    colors = ["#EF553B" if yi == 0 else "#636EFA" for yi in y]
    fig2.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                              marker=dict(color=colors, size=6,
                                          line=dict(width=0.5, color="white")),
                              name="Data"))
    fig2.update_layout(height=420, template="plotly_dark",
                       xaxis_title="Feature 1", yaxis_title="Feature 2",
                       margin=dict(t=20, b=30))
    st.plotly_chart(fig2, use_container_width=True)

    # Divergence warning
    if history["loss"][-1] > history["loss"][0] * 2:
        st.warning("⚠️ The loss is **diverging**! Try reducing the learning rate.")
