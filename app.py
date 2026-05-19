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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif; 
}

/* App Background */
.stApp {
    background-color: #0F0720;
    color: #eaddff;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #eaddff !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #170f28;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #bb86fc; /* Electric Violet */
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Metric cards & Expander */
div[data-testid="stMetric"], div[data-testid="stExpander"] {
    background: rgba(26, 11, 59, 0.6) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
}

div[data-testid="stMetric"] {
    padding: 16px 24px;
}

div[data-testid="stMetricValue"] {
    color: #46f5e0 !important; /* Cyan */
}

/* Expander header */
details summary { 
    font-weight: 600; 
    color: #dab9ff; 
}

/* Hero banner */
.hero {
    background: rgba(26, 11, 59, 0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: #eaddff;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
}
.hero h1 { 
    margin: 0 0 0.5rem 0; 
    font-size: 1.75rem; 
    color: #bb86fc !important; 
}
.hero p  { 
    margin: 0; 
    opacity: 0.9; 
    font-size: 1rem; 
    color: #cdc3d4;
}

/* Customizing inputs for the 'Control Center' feel */
.stNumberInput input {
    font-family: 'JetBrains Mono', monospace !important;
}

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
