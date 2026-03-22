# 🏆 Data-Driven Promotion Prediction in Organizational Hierarchies

A Streamlit web application that predicts employee promotions using Machine Learning. Upload a resume or fill in employee details to get instant promotion predictions with confidence scores and improvement suggestions.

---

## 🚀 Features

- 📊 **EDA Visualizations** — Interactive charts from the original dataset
- 🔧 **Preprocessing Pipeline** — Step-by-step data cleaning walkthrough
- 🤖 **Model Training** — Train Decision Tree, Random Forest, XGBoost with one click
- 📈 **Model Results** — Confusion matrix, feature importance, model comparison
- 🔮 **Predict** — Upload a resume PDF or fill manually to predict promotion

---

## 🛠️ Tech Stack

`Python` `Streamlit` `Scikit-learn` `XGBoost` `Imbalanced-learn` `Pandas` `Matplotlib` `Seaborn` `pdfplumber`

---


## 📊 Dataset

- **Source:** HR Analytics Dataset (Kaggle)
- **Train:** 54,808 records — **Test:** 23,490 records
- **Target:** `is_promoted` (binary — 0 or 1)
- **Class imbalance:** ~8.5% promoted → handled using **SMOTE**

---

## 🤖 Models Used

-**Decision Tree**
-**Random Forest**
-**XGBoost**

https://promotionpredictor.streamlit.app/Predict
