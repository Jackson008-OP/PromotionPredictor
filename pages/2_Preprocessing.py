import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Preprocessing", page_icon="🔧", layout="wide")

from utils.sidebar import render_sidebar
render_sidebar()

@st.cache_data
def load():
    return pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")

train_raw, test_raw = load()

st.title("🔧 Data Preprocessing Pipeline")
st.markdown("Step-by-step walkthrough of every preprocessing operation applied to the dataset.")
st.markdown("---")

# ── Step 1: Missing Values ────────────────────────────────────────────────────
st.subheader("Step 1 — Missing Value Imputation")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Before imputation — training data**")
    st.dataframe(
        pd.DataFrame({
            "Column": ["education", "previous_year_rating"],
            "Missing Count": [
                train_raw["education"].isnull().sum(),
                train_raw["previous_year_rating"].isnull().sum(),
            ],
            "Missing %": [
                f"{(train_raw['education'].isnull().sum()/len(train_raw)*100):.2f}%",
                f"{(train_raw['previous_year_rating'].isnull().sum()/len(train_raw)*100):.2f}%",
            ],
        }), use_container_width=True
    )

train_imp = train_raw.copy()
train_imp["education"] = train_imp["education"].fillna(train_imp["education"].mode()[0])
train_imp["previous_year_rating"] = train_imp["previous_year_rating"].fillna(
    train_imp["previous_year_rating"].mode()[0]
)

with col2:
    st.markdown("**After imputation — training data**")
    st.dataframe(
        pd.DataFrame({
            "Column": ["education", "previous_year_rating"],
            "Missing Count": [0, 0],
            "Strategy": ["Mode fill", "Mode fill"],
        }), use_container_width=True
    )

st.success("Missing values filled using mode (most frequent value) for both columns.")
st.markdown("---")

# ── Step 2: Outlier Removal ───────────────────────────────────────────────────
st.subheader("Step 2 — Outlier Removal")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=train_raw, x="length_of_service", color="red", ax=axes[0])
axes[0].set_title("Before — length_of_service (with outliers)", fontsize=12)

train_clean = train_imp[train_imp["length_of_service"] <= 13].copy()
sns.boxplot(data=train_clean, x="length_of_service", color="green", ax=axes[1])
axes[1].set_title("After — length_of_service (outliers removed > 13)", fontsize=12)

plt.tight_layout()
st.pyplot(fig)
plt.close()

col_a, col_b = st.columns(2)
col_a.metric("Rows before", f"{len(train_imp):,}")
col_b.metric("Rows after", f"{len(train_clean):,}", delta=f"-{len(train_imp)-len(train_clean)}")
st.markdown("---")

# ── Step 3: Feature Engineering ──────────────────────────────────────────────
st.subheader("Step 3 — Feature Engineering")

st.markdown("Three new features were created from existing columns:")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**KPIs_met_80**\n\n`avg_training_score > 80` → 1 else 0\n\nBinary flag for high performers")
with col2:
    st.info("**sum_metric**\n\n`awards_won + KPIs_met_80 + previous_year_rating`\n\nOverall performance score")
with col3:
    st.info("**total_score**\n\n`avg_training_score × no_of_trainings`\n\nWeighted learning effort")

train_fe = train_clean.copy()
train_fe["KPIs_met_80"] = train_fe["avg_training_score"].apply(lambda x: 1 if x > 80 else 0)
train_fe["sum_metric"] = train_fe["awards_won?"] + train_fe["KPIs_met_80"] + train_fe["previous_year_rating"]
train_fe["total_score"] = train_fe["avg_training_score"] * train_fe["no_of_trainings"]

with st.expander("Preview engineered features"):
    st.dataframe(
        train_fe[["avg_training_score", "KPIs_met_80", "awards_won?",
                   "previous_year_rating", "sum_metric", "no_of_trainings", "total_score"]].head(10),
        use_container_width=True
    )

st.markdown("---")

# ── Step 4: Noise Row Removal ─────────────────────────────────────────────────
st.subheader("Step 4 — Noise Row Removal")

noise_rows = train_fe[
    (train_fe["KPIs_met_80"] == 0) &
    (train_fe["previous_year_rating"] == 1.0) &
    (train_fe["awards_won?"] == 0) &
    (train_fe["avg_training_score"] < 60) &
    (train_fe["is_promoted"] == 1)
]

st.markdown(f"Employees with **zero KPIs**, **rating=1**, **no awards**, **low score** but still marked as promoted — likely labeling noise.")
st.metric("Noise rows removed", len(noise_rows))
if len(noise_rows) > 0:
    with st.expander("Show noise rows"):
        st.dataframe(noise_rows, use_container_width=True)

st.markdown("---")

# ── Step 5: Drop Columns ──────────────────────────────────────────────────────
st.subheader("Step 5 — Drop Low-Value Columns")

st.markdown("""
Three columns are dropped as they contribute very little to promotion prediction:

| Column | Reason |
|---|---|
| `employee_id` | Unique identifier — not a feature |
| `region` | Low correlation with promotion outcome |
| `recruitment_channel` | Weak predictor of promotion |
""")

st.markdown("---")

# ── Step 6: Encoding ───────────────────────────────────────────────────────────
st.subheader("Step 6 — Feature Encoding")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Education — ordinal encoding**")
    st.dataframe(pd.DataFrame({
        "Original Value": ["Master's & above", "Bachelor's", "Below Secondary"],
        "Encoded Value": [3, 2, 1],
        "Reason": ["Highest", "Mid", "Lowest"],
    }), use_container_width=True)

with col2:
    st.markdown("**Gender & Department — label encoding**")
    st.markdown("""
    `sklearn.LabelEncoder` applied to:
    - `gender` → integer codes
    - `department` → integer codes
    """)

st.markdown("---")

# ── Step 7: SMOTE ──────────────────────────────────────────────────────────────
st.subheader("Step 7 — SMOTE Oversampling")

before = train_raw["is_promoted"].value_counts()
promoted_count = int(before[1])
not_promoted_count = int(before[0])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

pd.Series({"Not Promoted": not_promoted_count, "Promoted": promoted_count}).plot(
    kind="bar", ax=axes[0], color=["#E24B4A", "#1D9E75"]
)
axes[0].set_title("Before SMOTE", fontsize=13)
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=0)
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f"{int(bar.get_height()):,}", ha="center", fontsize=11)

pd.Series({"Not Promoted": not_promoted_count, "Promoted": not_promoted_count}).plot(
    kind="bar", ax=axes[1], color=["#E24B4A", "#1D9E75"]
)
axes[1].set_title("After SMOTE (Balanced)", fontsize=13)
axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=0)
for bar in axes[1].patches:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f"{int(bar.get_height()):,}", ha="center", fontsize=11)

plt.suptitle("SMOTE Oversampling Effect on Class Distribution", fontsize=14)
plt.tight_layout()
st.pyplot(fig)
plt.close()

col1, col2, col3 = st.columns(3)
col1.metric("Not Promoted (original)", f"{not_promoted_count:,}")
col2.metric("Promoted (original)", f"{promoted_count:,}")
col3.metric("Promoted (after SMOTE)", f"{not_promoted_count:,}", delta=f"+{not_promoted_count - promoted_count:,}")

st.caption("SMOTE (Synthetic Minority Oversampling Technique) generates synthetic samples for the minority class to balance training data.")
