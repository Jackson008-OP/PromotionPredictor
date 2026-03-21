import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.train_models import load_artifacts, models_exist
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Model Results", page_icon="📈", layout="wide")

from utils.sidebar import render_sidebar
render_sidebar()

st.title("📈 Model Results & Comparison")
st.markdown("Detailed performance analysis of all trained models.")
st.markdown("---")

if not models_exist():
    st.warning("⚠️ No trained models found. Please go to **Model Training** page first.")
    st.stop()

@st.cache_resource
def get_models():
    return load_artifacts()

models, scaler, feature_names = get_models()

from utils.preprocess import load_data, full_preprocess, prepare_xy
from utils.train_models import apply_smote, split_and_scale

@st.cache_data
def get_eval_data():
    train_df, test_df = load_data("data/train.csv", "data/test.csv")
    train_processed, _ = full_preprocess(train_df, test_df)
    X, y = prepare_xy(train_processed)
    X_res, y_res = apply_smote(X, y)
    X_train, X_valid, y_train, y_valid, _ = split_and_scale(X_res, y_res)
    return X_train, X_valid, y_train, y_valid, list(X.columns)

with st.spinner("Preparing evaluation data..."):
    X_train, X_valid, y_train, y_valid, feat_names = get_eval_data()

# ── Accuracy summary ─────────────────────────────────────────────────────────
st.subheader("Accuracy Summary")

summary_rows = []
for name, model in models.items():
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = model.predict(X_valid)
    report = classification_report(y_valid, y_pred, output_dict=True)
    summary_rows.append({
        "Model": name,
        "Train Accuracy (%)": round(model.score(X_train, y_train.values.ravel()) * 100, 2),
        "Val Accuracy (%)": round(model.score(X_valid, y_valid.values.ravel()) * 100, 2),
        "Precision (Promoted)": round(report["1"]["precision"], 4),
        "Recall (Promoted)": round(report["1"]["recall"], 4),
        "F1 (Promoted)": round(report["1"]["f1-score"], 4),
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(
    summary_df.style.highlight_max(
        subset=["Val Accuracy (%)", "F1 (Promoted)"], color="#c0dd97"
    ).highlight_min(
        subset=["Val Accuracy (%)", "F1 (Promoted)"], color="#f7c1c1"
    ),
    use_container_width=True
)

st.markdown("---")

# ── Confusion matrices ────────────────────────────────────────────────────────
st.subheader("Confusion Matrices")

cols = st.columns(len(models))
for i, (name, model) in enumerate(models.items()):
    with cols[i]:
        st.markdown(f"**{name}**")
        from sklearn.metrics import confusion_matrix
        y_pred = model.predict(X_valid)
        cm = confusion_matrix(y_valid, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Wistia", ax=ax,
                    xticklabels=["Not Promoted", "Promoted"],
                    yticklabels=["Not Promoted", "Promoted"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.markdown("---")

# ── Feature importance ────────────────────────────────────────────────────────
st.subheader("Feature Importance")

model_choice = st.selectbox("Select model for feature importance", list(models.keys()))
selected_model = models[model_choice]

if hasattr(selected_model, "feature_importances_"):
    importances = selected_model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feat_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title(f"Feature Importance — {model_choice}", fontsize=14)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("Feature importance table"):
        fi_df["Importance (%)"] = (fi_df["Importance"] * 100).round(2)
        st.dataframe(fi_df[["Feature", "Importance (%)"]], use_container_width=True)
else:
    st.info("Feature importance not available for this model.")

st.markdown("---")

# ── Model comparison bar chart ────────────────────────────────────────────────
st.subheader("Side-by-Side Model Comparison")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = ["Train Accuracy (%)", "Val Accuracy (%)", "F1 (Promoted)"]
colors = ["#378ADD", "#1D9E75", "#EF9F27"]

for ax, metric, color in zip(axes, metrics, colors):
    vals = summary_df[metric].values
    bars = ax.bar(summary_df["Model"], vals, color=color)
    ax.set_title(metric, fontsize=12)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.tick_params(axis="x", rotation=15)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.2f}", ha="center", fontsize=10)

plt.suptitle("Model Performance Comparison", fontsize=14)
plt.tight_layout()
st.pyplot(fig)
plt.close()
