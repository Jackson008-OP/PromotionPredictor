import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.train_models import load_artifacts, models_exist
from utils.resume_parser import parse_resume
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Predict Promotion", page_icon="🔮", layout="wide")

from utils.sidebar import render_sidebar
render_sidebar()

st.title("🔮 Predict Employee Promotion")
st.markdown("Upload a resume **or** fill in the form manually to predict promotion.")
st.markdown("---")

if not models_exist():
    st.warning("⚠️ No trained models found. Please go to **Model Training** page first.")
    st.stop()

@st.cache_resource
def get_models():
    return load_artifacts()

models, scaler, feature_names = get_models()

DEPARTMENTS = [
    "Sales & Marketing", "Operations", "Technology", "Analytics & Reporting",
    "Finance", "HR", "Legal", "Procurement", "R&D"
]
DEPT_ENCODE = {d: i for i, d in enumerate(sorted(DEPARTMENTS))}
EDU_ENCODE = {"Below Secondary": 1, "Bachelor's": 2, "Master's & above": 3}
GENDER_ENCODE = {"Male": 1, "Female": 0}
RATING_OPTIONS = [1.0, 2.0, 3.0, 4.0, 5.0]


def build_feature_vector(dept, edu, gender, n_trainings, age,
                          prev_rating, los, awards, avg_score):
    dept_enc = DEPT_ENCODE.get(dept, 0)
    edu_enc = EDU_ENCODE.get(edu, 2)
    gender_enc = GENDER_ENCODE.get(gender, 1)
    kpis = 1 if avg_score > 80 else 0
    sum_metric = awards + kpis + prev_rating
    total_score = avg_score * n_trainings

    row = {
        "department": dept_enc,
        "education": edu_enc,
        "gender": gender_enc,
        "no_of_trainings": n_trainings,
        "age": age,
        "previous_year_rating": prev_rating,
        "length_of_service": los,
        "awards_won?": awards,
        "avg_training_score": avg_score,
        "KPIs_met_80": kpis,
        "sum_metric": sum_metric,
        "total_score": total_score,
    }

    df_row = pd.DataFrame([row])
    if feature_names:
        df_row = df_row.reindex(columns=feature_names, fill_value=0)

    scaled = scaler.transform(df_row)
    return scaled, row


def show_result(model_name, model, scaled_input, raw_row):
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0]
    confidence = proba[prediction] * 100

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"## ✅ Promoted")
        st.markdown(f"**Confidence:** {confidence:.1f}%")
    else:
        st.error(f"## ❌ Not Promoted")
        st.markdown(f"**Confidence:** {confidence:.1f}%")

    col1, col2, col3 = st.columns(3)
    col1.metric("Promotion probability", f"{proba[1]*100:.1f}%")
    col2.metric("Not promoted probability", f"{proba[0]*100:.1f}%")
    col3.metric("Model used", model_name)

    # Probability bar
    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.barh(["Not Promoted", "Promoted"], [proba[0]*100, proba[1]*100],
            color=["#E24B4A", "#1D9E75"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    for i, v in enumerate([proba[0]*100, proba[1]*100]):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Feature importance contribution
    if hasattr(model, "feature_importances_"):
        st.markdown("---")
        st.subheader("Top Features Influencing This Prediction")

        fi = model.feature_importances_
        feat_vals = list(raw_row.values())
        feat_keys = list(raw_row.keys())

        fi_df = pd.DataFrame({
            "Feature": feat_keys[:len(fi)],
            "Importance": fi[:len(feat_keys)],
            "Your Value": feat_vals[:len(fi)],
        }).sort_values("Importance", ascending=False).head(6)

        fig, ax = plt.subplots(figsize=(10, 4))
        colors_bar = ["#1D9E75" if v > fi.mean() else "#E24B4A" for v in fi_df["Importance"]]
        ax.barh(fi_df["Feature"], fi_df["Importance"], color=colors_bar)
        ax.set_xlabel("Importance Score")
        ax.set_title("Features that matter most for your prediction")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Improvement suggestions
    if prediction == 0:
        st.markdown("---")
        st.subheader("💡 Suggestions to Improve Promotion Chances")
        tips = []
        if raw_row["avg_training_score"] < 80:
            tips.append(f"📚 Improve your average training score (currently **{raw_row['avg_training_score']}**). Aim for **above 80** to meet KPI threshold.")
        if raw_row["awards_won?"] == 0:
            tips.append("🏆 Try to win at least one award or recognition — it significantly impacts the sum metric.")
        if raw_row["previous_year_rating"] < 4:
            tips.append(f"⭐ Improve your previous year performance rating (currently **{raw_row['previous_year_rating']}**). Aim for **4 or 5**.")
        if raw_row["no_of_trainings"] < 3:
            tips.append(f"🎓 Attend more training programs (currently **{raw_row['no_of_trainings']}**). More trainings increase total score.")
        if not tips:
            tips.append("Your profile is close to promotion threshold. Keep maintaining your current performance levels.")
        for tip in tips:
            st.markdown(f"- {tip}")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📄 Upload Resume", "✏️ Fill Manually"])


# ── TAB 1: Resume Upload ──────────────────────────────────────────────────────
with tab1:
    st.markdown("### Upload your resume (PDF)")
    st.markdown("The app will extract relevant features automatically. You can review and edit before predicting.")

    uploaded = st.file_uploader("Upload resume PDF", type=["pdf"], key="resume_upload")

    if uploaded:
        with st.spinner("Extracting information from resume..."):
            try:
                extracted, confidence, preview_text = parse_resume(uploaded)
            except Exception as e:
                st.error(f"Could not parse resume: {e}")
                st.stop()

        st.success("✅ Resume parsed! Review and correct the extracted values below.")

        with st.expander("Raw text extracted from resume (first 2000 chars)"):
            st.text(preview_text)

        st.markdown("#### Extracted Values — Review & Edit")
        st.caption("Fields marked 🟡 are estimated. Fields marked 🟢 were found in the resume.")

        col1, col2, col3 = st.columns(3)

        with col1:
            dept_idx = DEPARTMENTS.index(extracted["department"]) if extracted["department"] in DEPARTMENTS else 0
            dept = st.selectbox("Department", DEPARTMENTS, index=dept_idx,
                                help=f"Confidence: {confidence.get('department','?')}")
            edu_options = list(EDU_ENCODE.keys())
            edu_idx = edu_options.index(
                next((k for k, v in EDU_ENCODE.items() if v == extracted["education"]), "Bachelor's")
            )
            edu = st.selectbox("Education", edu_options, index=edu_idx,
                               help=f"Confidence: {confidence.get('education','?')}")
            gender = st.selectbox("Gender", ["Male", "Female"],
                                  index=0 if extracted["gender"] == "m" else 1)

        with col2:
            age = st.slider("Age", 20, 60, int(extracted["age"]),
                            help=f"Confidence: {confidence.get('age','?')}")
            los = st.slider("Length of Service (years)", 1, 30, int(extracted["length_of_service"]),
                            help=f"Confidence: {confidence.get('length_of_service','?')}")
            n_trainings = st.slider("Number of Trainings", 1, 10, int(extracted["no_of_trainings"]))

        with col3:
            avg_score = st.slider("Avg Training Score", 40, 100, int(extracted["avg_training_score"]),
                                  help=f"Confidence: {confidence.get('avg_training_score','?')}")
            prev_rating = st.select_slider("Previous Year Rating", RATING_OPTIONS,
                                           value=extracted["previous_year_rating"])
            awards = st.radio("Awards Won", [0, 1],
                              index=int(extracted["awards_won"]),
                              format_func=lambda x: "Yes" if x == 1 else "No",
                              horizontal=True)

        model_name_r = st.selectbox("Select Model", list(models.keys()), key="model_resume")

        if st.button("🔮 Predict Promotion", type="primary", key="predict_resume"):
            scaled, raw = build_feature_vector(
                dept, edu, gender, n_trainings, age,
                prev_rating, los, awards, avg_score
            )
            show_result(model_name_r, models[model_name_r], scaled, raw)


# ── TAB 2: Manual Form ────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Fill in employee details")

    col1, col2, col3 = st.columns(3)

    with col1:
        m_dept = st.selectbox("Department", DEPARTMENTS, key="m_dept")
        m_edu = st.selectbox("Education", list(EDU_ENCODE.keys()), key="m_edu")
        m_gender = st.selectbox("Gender", ["Male", "Female"], key="m_gender")

    with col2:
        m_age = st.slider("Age", 20, 60, 30, key="m_age")
        m_los = st.slider("Length of Service (years)", 1, 30, 3, key="m_los")
        m_trainings = st.slider("Number of Trainings", 1, 10, 2, key="m_trainings")

    with col3:
        m_score = st.slider("Avg Training Score", 40, 100, 65, key="m_score")
        m_rating = st.select_slider("Previous Year Rating", RATING_OPTIONS, value=3.0, key="m_rating")
        m_awards = st.radio("Awards Won", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No",
                             horizontal=True, key="m_awards")

    model_name_m = st.selectbox("Select Model", list(models.keys()), key="model_manual")

    if st.button("🔮 Predict Promotion", type="primary", key="predict_manual"):
        scaled, raw = build_feature_vector(
            m_dept, m_edu, m_gender, m_trainings, m_age,
            m_rating, m_los, m_awards, m_score
        )
        show_result(model_name_m, models[model_name_m], scaled, raw)
