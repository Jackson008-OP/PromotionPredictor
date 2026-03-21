import streamlit as st

SIDEBAR_CSS = """
<style>
[data-testid="stSidebarNav"] { display: none; }

[data-testid="stSidebarContent"] a {
    display: flex;
    align-items: center;
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 500;
    color: var(--text-color);
    text-decoration: none;
    margin-bottom: 4px;
    transition: background-color 0.2s;
}

[data-testid="stSidebarContent"] a:hover {
    background-color: rgba(255, 255, 255, 0.1);
}

[data-testid="stSidebarContent"] a[aria-current="page"] {
    background-color: #1f6feb;
    color: white !important;
}
</style>
"""

def render_sidebar():
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
    st.sidebar.markdown("## 🏆 Promotion Predictor")
    st.sidebar.markdown("*Data-Driven Promotion Prediction*")
    st.sidebar.markdown("---")
    st.sidebar.page_link("app.py",                         label="🏠  App")
    st.sidebar.page_link("pages/1_EDA_Visualizations.py",  label="📊  EDA Visualizations")
    st.sidebar.page_link("pages/2_Preprocessing.py",       label="🔧  Preprocessing")
    st.sidebar.page_link("pages/3_Model_Training.py",      label="🤖  Model Training")
    st.sidebar.page_link("pages/4_Model_Results.py",       label="📈  Model Results")
    st.sidebar.page_link("pages/5_Predict.py",             label="🔮  Predict")
    st.sidebar.markdown("---")