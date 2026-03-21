import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import load_data, full_preprocess, prepare_xy
from utils.train_models import (
    apply_smote, split_and_scale, train_all_models,
    evaluate_model, save_artifacts, models_exist
)
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

from utils.sidebar import render_sidebar
render_sidebar()

st.title("🤖 Model Training")
st.markdown("Train all three models on the preprocessed dataset with SMOTE balancing.")
st.markdown("---")

st.subheader("Pipeline Overview")
col1, col2, col3, col4, col5 = st.columns(5)
for col, label in zip(
    [col1, col2, col3, col4, col5],
    ["Load data", "Preprocess", "SMOTE", "Split + Scale", "Train models"]
):
    col.info(label)

st.markdown("---")

if models_exist():
    st.success("✅ Models are already trained and saved. You can retrain below or go to Model Results.")

col_btn, col_info = st.columns([1, 3])
with col_btn:
    train_btn = st.button("🚀 Train All Models", type="primary", use_container_width=True)
with col_info:
    st.markdown("Trains Decision Tree, Random Forest, and XGBoost — then saves all models to disk.")

if train_btn:
    results = {}

    with st.spinner("Loading and preprocessing data..."):
        train_df, test_df = load_data("data/train.csv", "data/test.csv")
        train_processed, test_processed = full_preprocess(train_df, test_df)
        X, y = prepare_xy(train_processed)
        feature_names = list(X.columns)

    st.success(f"Data ready — {X.shape[0]:,} rows × {X.shape[1]} features after preprocessing")

    with st.spinner("Applying SMOTE..."):
        X_res, y_res = apply_smote(X, y)

    col1, col2 = st.columns(2)
    col1.metric("Samples before SMOTE", f"{len(X):,}")
    col2.metric("Samples after SMOTE", f"{len(X_res):,}")

    with st.spinner("Splitting and scaling..."):
        X_train, X_valid, y_train, y_valid, scaler = split_and_scale(X_res, y_res)

    col1, col2, col3 = st.columns(3)
    col1.metric("Train samples", f"{len(X_train):,}")
    col2.metric("Validation samples", f"{len(X_valid):,}")
    col3.metric("Features", X_train.shape[1])

    with st.spinner("Training Decision Tree, Random Forest, XGBoost..."):
        trained_models = train_all_models(X_train, y_train)

    for name, model in trained_models.items():
        results[name] = evaluate_model(model, X_train, y_train, X_valid, y_valid)

    save_artifacts(trained_models, scaler, feature_names)
    st.success("✅ All models trained and saved successfully!")
    st.markdown("---")

    # ── Accuracy comparison ──────────────────────────────────────────────────
    st.subheader("Model Accuracy Comparison")

    acc_data = {
        "Model": list(results.keys()),
        "Training Accuracy (%)": [r["train_accuracy"] for r in results.values()],
        "Validation Accuracy (%)": [r["valid_accuracy"] for r in results.values()],
    }
    acc_df = pd.DataFrame(acc_data)
    st.dataframe(acc_df.style.highlight_max(
        subset=["Validation Accuracy (%)"], color="#c0dd97"
    ), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(acc_df))
    w = 0.35
    bars1 = ax.bar(x - w/2, acc_df["Training Accuracy (%)"], w, label="Training", color="#378ADD")
    bars2 = ax.bar(x + w/2, acc_df["Validation Accuracy (%)"], w, label="Validation", color="#1D9E75")
    ax.set_xticks(x)
    ax.set_xticklabels(acc_df["Model"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training vs Validation Accuracy")
    ax.legend()
    ax.set_ylim(0, 110)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Confusion matrices ───────────────────────────────────────────────────
    st.subheader("Confusion Matrices")

    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(
                res["confusion_matrix"], annot=True, fmt="d",
                cmap="Wistia", ax=ax,
                xticklabels=["Not Promoted", "Promoted"],
                yticklabels=["Not Promoted", "Promoted"],
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # ── Classification reports ───────────────────────────────────────────────
    st.subheader("Classification Reports")

    tabs = st.tabs(list(results.keys()))
    for tab, (name, res) in zip(tabs, results.items()):
        with tab:
            report = res["classification_report"]
            rows = []
            for label in ["0", "1"]:
                rows.append({
                    "Class": "Not Promoted" if label == "0" else "Promoted",
                    "Precision": round(report[label]["precision"], 4),
                    "Recall": round(report[label]["recall"], 4),
                    "F1-Score": round(report[label]["f1-score"], 4),
                    "Support": int(report[label]["support"]),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            col1, col2 = st.columns(2)
            col1.metric("Overall Accuracy", f"{res['valid_accuracy']}%")
            col2.metric("Macro F1", f"{round(report['macro avg']['f1-score'] * 100, 2)}%")

    st.balloons()
