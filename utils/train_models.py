import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

MODELS_DIR = "models"


def apply_smote(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y.values.ravel())
    y_res = pd.DataFrame(y_res, columns=["is_promoted"])
    return X_res, y_res


def split_and_scale(X_res, y_res):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_res, y_res, test_size=0.2, random_state=0
    )
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_valid_sc = sc.transform(X_valid)
    return X_train_sc, X_valid_sc, y_train, y_valid, sc


def train_all_models(X_train, y_train):
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42
        ),
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train.values.ravel())
        trained[name] = model
    return trained


def evaluate_model(model, X_train, y_train, X_valid, y_valid):
    train_acc = model.score(X_train, y_train.values.ravel())
    valid_acc = model.score(X_valid, y_valid.values.ravel())
    y_pred = model.predict(X_valid)
    cm = confusion_matrix(y_valid, y_pred)
    report = classification_report(y_valid, y_pred, output_dict=True)
    return {
        "train_accuracy": round(train_acc * 100, 2),
        "valid_accuracy": round(valid_acc * 100, 2),
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
        "y_valid": y_valid,
    }


def save_artifacts(trained_models, scaler, feature_names):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
    for name, model in trained_models.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, os.path.join(MODELS_DIR, fname))


def load_artifacts():
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    models = {}
    for fname, label in [
        ("decision_tree.pkl", "Decision Tree"),
        ("random_forest.pkl", "Random Forest"),
        ("xgboost.pkl", "XGBoost"),
    ]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[label] = joblib.load(path)
    return models, scaler, feature_names


def models_exist():
    files = ["scaler.pkl", "decision_tree.pkl", "random_forest.pkl", "xgboost.pkl"]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in files)
