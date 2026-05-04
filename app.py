"""
╔══════════════════════════════════════════════════════════════════╗
║  Analysis and Visualization of Convergence in Supervised        ║
║  and Unsupervised Learning Algorithms                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Run with:  streamlit run app.py                                ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Convergence Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for Premium Look ─────────────────────────────────
st.markdown("""
<style>
/* ---- Global ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a78bfa;
    border-bottom: 1px solid #4c3f8f;
    padding-bottom: 6px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px 16px;
}

/* Expander header */
details summary { font-weight: 600; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
}
.hero h1 { margin: 0 0 0.5rem 0; font-size: 1.75rem; }
.hero p  { margin: 0; opacity: 0.9; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─── Hero Banner ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📈 ML Convergence Visualizer</h1>
    <p>Interactive exploration of how Supervised &amp; Unsupervised
       learning algorithms converge during training.</p>
</div>
""", unsafe_allow_html=True)

# ─── Navigation ──────────────────────────────────────────────────
PAGES = {
    "🔵 Logistic Regression": "logistic",
    "🟢 Linear Regression": "linear",
    "🟡 K-Means Clustering": "kmeans",
    "📊 Comparison Dashboard": "comparison",
}

st.sidebar.markdown("## 🧭 Navigation")
selection = st.sidebar.radio("Choose Module", list(PAGES.keys()),
                             label_visibility="collapsed")
page = PAGES[selection]

st.sidebar.markdown("---")

# ─── Route to Module ─────────────────────────────────────────────
if page == "logistic":
    from modules.logistic_regression import render
    render()
elif page == "linear":
    from modules.linear_regression import render
    render()
elif page == "kmeans":
    from modules.kmeans import render
    render()
elif page == "comparison":
    from modules.comparison import render
    render()

# ─── Footer ──────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='opacity:0.5'>Built for educational purposes · "
    "ML Convergence Visualizer</small>",
    unsafe_allow_html=True,
)
