"""
Linear Regression Module — Gradient Descent vs Normal Equation
Demonstrates MSE convergence and regression-line evolution.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ──────────────────────────────────────────────
# Core Linear Regression (from scratch)
# ──────────────────────────────────────────────

def generate_linear_data(n_samples, noise, seed=42):
    """Generate y = 3x + 7 + noise."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-5, 5, n_samples)
    y = 3 * X + 7 + rng.normal(0, noise, n_samples)
    return X.reshape(-1, 1), y


def compute_mse(X, y, w, b):
    preds = X @ w + b
    return np.mean((preds - y) ** 2)


def train_linear_regression(X, y, lr, n_iters, early_stop_tol=1e-8):
    """
    Gradient descent for linear regression.
    Returns weight/bias history, MSE history, and gradient norms.
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    history = {"mse": [], "grad_norm": [], "w": [], "b": []}

    for _ in range(n_iters):
        preds = X @ w + b
        error = preds - y

        dw = (2 / m) * (X.T @ error)
        db = (2 / m) * np.sum(error)

        w -= lr * dw
        b -= lr * db

        mse = compute_mse(X, y, w, b)
        grad_norm = np.linalg.norm(dw)

        history["mse"].append(mse)
        history["grad_norm"].append(grad_norm)
        history["w"].append(w.copy())
        history["b"].append(b)

        if grad_norm < early_stop_tol:
            break

    return w, b, history


def normal_equation(X, y):
    """Closed-form solution: θ = (XᵀX)⁻¹Xᵀy."""
    X_b = np.c_[np.ones(len(X)), X]
    theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    return theta[1:], theta[0]  # weights, bias


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────

def render():
    st.markdown("## 🟢 Linear Regression — Gradient Descent Convergence")

    with st.expander("📖 Theory — Convergence in Linear Regression", expanded=False):
        st.markdown(r"""
**Mean Squared Error:**

$$J(\theta) = \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2$$

**Gradient Descent Update:**

$$\theta := \theta - \alpha \frac{2}{m} X^T (X\theta - y)$$

**Normal Equation (closed-form):**

$$\theta = (X^T X)^{-1} X^T y$$

- GD converges when the learning rate α < 2/λ_max (largest eigenvalue of XᵀX/m).
- The Normal Equation gives the **exact** optimum in one step but is O(n³).
""")

    # ---- Sidebar ----
    st.sidebar.markdown("### 🟢 Linear Regression")
    lr = st.sidebar.slider("Learning Rate (α)", 0.0001, 2.0, 0.05, 0.001,
                           key="lr_lin")
    n_iters = st.sidebar.slider("Iterations", 10, 2000, 300, 10,
                                key="it_lin")
    n_samples = st.sidebar.slider("Samples", 30, 500, 150, 10,
                                  key="ns_lin")
    noise = st.sidebar.slider("Noise σ", 0.5, 10.0, 3.0, 0.5,
                              key="noise_lin")
    early_stop = st.sidebar.checkbox("Enable Early Stopping", True,
                                     key="es_lin")
    tol = 1e-8 if early_stop else 0.0

    # ---- Data ----
    X, y = generate_linear_data(n_samples, noise)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # ---- Train ----
    w_gd, b_gd, hist = train_linear_regression(X_norm, y, lr, n_iters, tol)
    w_ne, b_ne = normal_equation(X_norm, y)
    mse_ne = compute_mse(X_norm, y, w_ne, b_ne)

    # ---- Metrics ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GD Final MSE", f"{hist['mse'][-1]:.4f}")
    c2.metric("Normal Eq MSE", f"{mse_ne:.4f}")
    c3.metric("GD Iters", len(hist["mse"]))
    c4.metric("‖∇‖ final", f"{hist['grad_norm'][-1]:.2e}")

    # ---- MSE Curve ----
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("MSE vs Iterations",
                                        "Regression Line Evolution"))
    fig.add_trace(go.Scatter(y=hist["mse"], mode="lines",
                             line=dict(color="#00CC96", width=2),
                             name="GD MSE"), row=1, col=1)
    fig.add_hline(y=mse_ne, line_dash="dash", line_color="#FFA15A",
                  annotation_text="Normal Eq", row=1, col=1)

    # ---- Regression line snapshots ----
    x_plot = np.linspace(X_norm.min(), X_norm.max(), 100).reshape(-1, 1)
    snapshots = [0, len(hist["w"]) // 4, len(hist["w"]) // 2, -1]
    palette = ["#EF553B", "#FFA15A", "#AB63FA", "#00CC96"]
    for idx, s in enumerate(snapshots):
        wi, bi = hist["w"][s], hist["b"][s]
        y_line = x_plot @ wi + bi
        fig.add_trace(go.Scatter(x=x_plot.ravel(), y=y_line.ravel(),
                                 mode="lines",
                                 line=dict(color=palette[idx], width=2),
                                 name=f"Iter {s if s >= 0 else len(hist['w'])}"),
                      row=1, col=2)
    fig.add_trace(go.Scatter(x=X_norm.ravel(), y=y, mode="markers",
                             marker=dict(color="white", size=4, opacity=0.5),
                             name="Data"), row=1, col=2)
    fig.update_layout(height=400, template="plotly_dark",
                      margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

    if hist["mse"][-1] > hist["mse"][0] * 2:
        st.warning("⚠️ MSE is **diverging**! Lower the learning rate.")
