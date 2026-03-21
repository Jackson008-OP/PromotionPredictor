import streamlit as st

st.set_page_config(
    page_title="Promotion Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
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
""", unsafe_allow_html=True)

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

st.title("🏆 Data-Driven Promotion Prediction")
st.markdown("### In Organizational Hierarchies")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**54,808** Training records")
with col2:
    st.info("**23,490** Test records")
with col3:
    st.info("**12** Features analyzed")

st.markdown("---")

st.markdown("""
### About This Project

This application predicts whether an employee will be promoted based on their
performance metrics and demographic data. Built using real HR data, it combines
machine learning with an interactive interface to help HR teams make data-informed
promotion decisions.
""")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### What Was Done")
    st.markdown("""
- Exploratory data analysis on 54,808 employee records
- Handled missing values in `education` and `previous_year_rating`
- Removed outliers in `length_of_service`
- Engineered new features — KPI metric, sum metric, total score
- Applied **SMOTE** to fix severe class imbalance (only ~8.5% promoted)
- Trained three models and compared performance
""")

with col_b:
    st.markdown("#### Models Used")
    st.markdown("""
| Model | Type |
|---|---|
| Decision Tree | Interpretable baseline |
| Random Forest | Ensemble — 100 trees |
| XGBoost | Gradient boosting |

All models trained with **StandardScaler** normalization and
**80/20 train-validation** split after SMOTE resampling.
""")

st.markdown("---")
st.markdown("#### Navigate using the sidebar to explore EDA, train models, and predict promotions.")
st.caption("Project by — Nov 2024 to Dec 2024 | Python · Scikit-learn · XGBoost · Streamlit")