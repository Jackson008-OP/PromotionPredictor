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

st.set_page_config(page_title="EDA & Visualizations", page_icon="📊", layout="wide")

from utils.sidebar import render_sidebar
render_sidebar()

plt.rcParams["figure.figsize"] = (14, 5)
plt.style.use("fivethirtyeight")

@st.cache_data
def load():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    return train, test

train_raw, test_raw = load()

st.title("📊 Exploratory Data Analysis")
st.markdown("All visualizations reproduced from the original notebook — using the raw dataset before any preprocessing.")
st.markdown("---")


# ── 1. Dataset Overview ──────────────────────────────────────────────────────
st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Train rows", f"{train_raw.shape[0]:,}")
col2.metric("Test rows", f"{test_raw.shape[0]:,}")
col3.metric("Features", train_raw.shape[1] - 1)
col4.metric("Target", "is_promoted")

with st.expander("Show first 5 rows of training data"):
    st.dataframe(train_raw.head(), use_container_width=True)

with st.expander("Descriptive statistics — numerical columns"):
    st.dataframe(train_raw.describe(), use_container_width=True)

with st.expander("Descriptive statistics — categorical columns"):
    st.dataframe(train_raw.describe(include="object"), use_container_width=True)

st.markdown("---")


# ── 2. Missing Values ────────────────────────────────────────────────────────
st.subheader("Missing Values")

col_tr, col_te = st.columns(2)

with col_tr:
    st.markdown("**Training data**")
    train_miss = pd.DataFrame({
        "Total Missing": train_raw.isnull().sum(),
        "Percentage (%)": ((train_raw.isnull().sum() / train_raw.shape[0]) * 100).round(2)
    })
    st.dataframe(train_miss[train_miss["Total Missing"] > 0], use_container_width=True)

with col_te:
    st.markdown("**Test data**")
    test_miss = pd.DataFrame({
        "Total Missing": test_raw.isnull().sum(),
        "Percentage (%)": ((test_raw.isnull().sum() / test_raw.shape[0]) * 100).round(2)
    })
    st.dataframe(test_miss[test_miss["Total Missing"] > 0], use_container_width=True)

st.markdown("---")


# ── 3. Target Class Balance ──────────────────────────────────────────────────
st.subheader("Target Class Balance")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plt.style.use("fivethirtyeight")

sns.countplot(data=train_raw, x="is_promoted", ax=axes[0])
axes[0].set_xlabel("Promoted or Not?", fontsize=10)
axes[0].set_title("Count of Promoted vs Not Promoted")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Not Promoted (0)", "Promoted (1)"])

train_raw["is_promoted"].value_counts().plot(
    kind="pie",
    explode=[0, 0.1],
    autopct="%.2f%%",
    startangle=90,
    labels=["Not Promoted", "Promoted"],
    shadow=True,
    pctdistance=0.5,
    ax=axes[1],
)
axes[1].set_title("Target Class Balance")
axes[1].axis("off")

plt.suptitle("Target Class Balance", fontsize=14)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")


# ── 4. Feature Distributions ─────────────────────────────────────────────────
st.subheader("Feature Distributions")

# No. of trainings
fig, ax = plt.subplots(figsize=(14, 4))
sns.countplot(data=train_raw, x="no_of_trainings", palette="spring", ax=ax)
ax.set_xlabel(" ", fontsize=14)
ax.set_title("Distribution of Trainings Undertaken by the Employees")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Age distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(train_raw["age"], color="red", bins=20)
ax.set_title("Distribution of Age Among the Employees", fontsize=15)
ax.set_xlabel("Age of the Employees")
ax.grid(True)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Department distribution
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(data=train_raw, y="department", palette="cividis", ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Department Name")
ax.set_title("Distribution of Employees in Different Departments", fontsize=15)
ax.grid(True)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Region distribution
fig, ax = plt.subplots(figsize=(12, 8))
sns.countplot(data=train_raw, y="region", palette="inferno", ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Region")
ax.set_title("Different Regions", fontsize=15)
ax.grid(True)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Education + Gender + Recruitment pies
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, col in enumerate(["education", "gender", "recruitment_channel"]):
    labels = train_raw[col].value_counts().index
    sizes = train_raw[col].value_counts()
    colors = plt.cm.copper(np.linspace(0, 1, len(labels)))
    axes[i].pie(sizes, labels=labels, colors=colors, shadow=True,
                startangle=90, autopct="%.1f%%")
    axes[i].set_title(col.replace("_", " ").title())

plt.suptitle("Education, Gender & Recruitment Channel", fontsize=14)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")


# ── 5. Outlier Detection ──────────────────────────────────────────────────────
st.subheader("Outlier Detection — Boxplots")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=train_raw, x="avg_training_score", color="red", ax=axes[0])
axes[0].set_xlabel("Average Training Score", fontsize=12)
axes[0].set_ylabel("Range", fontsize=12)
axes[0].set_title("Avg Training Score — Outliers")

sns.boxplot(data=train_raw, x="length_of_service", color="red", ax=axes[1])
axes[1].set_xlabel("Length of Service", fontsize=12)
axes[1].set_ylabel("Range", fontsize=12)
axes[1].set_title("Length of Service — Outliers")

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")


# ── 6. Effect on Promotion ───────────────────────────────────────────────────
st.subheader("Effect of Features on Promotion")

# Gender gap
fig, ax = plt.subplots(figsize=(14, 3))
x = pd.crosstab(train_raw["gender"], train_raw["is_promoted"])
colors = plt.cm.Wistia(np.linspace(0, 1, 5))
x.div(x.sum(1).astype(float), axis=0).plot(
    kind="bar", stacked=False, color=colors, ax=ax
)
ax.set_title("Effect of Gender on Promotion", fontsize=15)
ax.set_xlabel(" ")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Department effect
fig, ax = plt.subplots(figsize=(14, 4))
x = pd.crosstab(train_raw["department"], train_raw["is_promoted"])
colors = plt.cm.copper(np.linspace(0, 1, 3))
x.div(x.sum(1).astype(float), axis=0).plot(
    kind="area", stacked=False, color=colors, ax=ax
)
ax.set_title("Effect of Department on Promotion", fontsize=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
ax.set_xlabel(" ")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Age vs promotion
fig, ax = plt.subplots(figsize=(14, 4))
sns.boxenplot(data=train_raw, x="is_promoted", y="age", palette="PuRd", ax=ax)
ax.set_title("Effect of Age on Promotion", fontsize=15)
ax.set_xlabel("Is the Employee Promoted?", fontsize=10)
ax.set_ylabel("Age of the Employee", fontsize=10)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Dept vs avg training score
fig, ax = plt.subplots(figsize=(16, 7))
sns.boxplot(data=train_raw, x="department", y="avg_training_score", palette="autumn", ax=ax)
ax.set_title("Average Training Scores from Each Department", fontsize=15)
ax.set_ylabel("Avg Training Score", fontsize=10)
ax.set_xlabel("Departments", fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")


# ── 7. Dept + Gender Barplot ─────────────────────────────────────────────────
st.subheader("Average Training Score by Department and Gender")

fig, ax = plt.subplots(figsize=(16, 7))
sns.barplot(
    data=train_raw,
    x="department",
    y="avg_training_score",
    hue="gender",
    palette="autumn",
    ax=ax,
)
ax.set_title(
    "Avg Training Score per Department by Gender", fontsize=15
)
ax.set_ylabel("Avg Training Score", fontsize=10)
ax.set_xlabel("Departments", fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")


# ── 8. Correlation Heatmap ───────────────────────────────────────────────────
st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(
    train_raw.select_dtypes(include=np.number).corr(),
    annot=True,
    linewidth=0.5,
    cmap="Wistia",
    ax=ax,
    fmt=".2f",
)
ax.set_title("Correlation Heatmap — All Numerical Features", fontsize=15)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")


# ── 9. SMOTE Before / After ──────────────────────────────────────────────────
st.subheader("Class Balance — Before vs After SMOTE")

before_counts = train_raw["is_promoted"].value_counts()
after_counts = pd.Series({0: before_counts[1], 1: before_counts[1]},
                          name="After SMOTE")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

before_counts.plot(kind="bar", ax=axes[0], color=["#E24B4A", "#1D9E75"])
axes[0].set_title("Before SMOTE", fontsize=13)
axes[0].set_xlabel("is_promoted")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(["Not Promoted", "Promoted"], rotation=0)

after_counts.plot(kind="bar", ax=axes[1], color=["#E24B4A", "#1D9E75"])
axes[1].set_title("After SMOTE (Balanced)", fontsize=13)
axes[1].set_xlabel("is_promoted")
axes[1].set_ylabel("Count")
axes[1].set_xticklabels(["Not Promoted", "Promoted"], rotation=0)

plt.suptitle("SMOTE Oversampling Effect", fontsize=14)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.caption("After SMOTE, minority class (Promoted) is oversampled to match majority class for fair model training.")
