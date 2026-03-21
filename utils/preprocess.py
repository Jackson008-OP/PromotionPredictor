import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def get_missing_summary(df):
    total = df.isnull().sum()
    percent = ((df.isnull().sum() / df.shape[0]) * 100).round(2)
    summary = pd.DataFrame({"Total Missing": total, "Percentage (%)": percent})
    return summary[summary["Total Missing"] > 0]


def impute_missing(df):
    df = df.copy()
    df["education"] = df["education"].fillna(df["education"].mode()[0])
    df["previous_year_rating"] = df["previous_year_rating"].fillna(
        df["previous_year_rating"].mode()[0]
    )
    return df


def remove_outliers(df):
    df = df.copy()
    df = df[df["length_of_service"] <= 13]
    return df


def feature_engineering(df):
    df = df.copy()
    df["KPIs_met_80"] = df["avg_training_score"].apply(lambda x: 1 if x > 80 else 0)
    df["sum_metric"] = (
        df["awards_won?"] + df["KPIs_met_80"] + df["previous_year_rating"]
    )
    df["total_score"] = df["avg_training_score"] * df["no_of_trainings"]
    return df


def remove_noise_rows(df):
    df = df.copy()
    df = df.drop(
        df[
            (df["KPIs_met_80"] == 0)
            & (df["previous_year_rating"] == 1.0)
            & (df["awards_won?"] == 0)
            & (df["avg_training_score"] < 60)
            & (df["is_promoted"] == 1)
        ].index
    )
    return df


def drop_columns(df, is_train=True):
    df = df.copy()
    cols_to_drop = ["recruitment_channel", "region", "employee_id"]
    df = df.drop(cols_to_drop, axis=1)
    return df


def encode_features(df):
    df = df.copy()
    df["education"] = df["education"].replace(
        {"Master's & above": 3, "Bachelor's": 2, "Below Secondary": 1}
    )
    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"].astype(str))
    df["department"] = le.fit_transform(df["department"].astype(str))
    return df


def full_preprocess(train, test):
    train = impute_missing(train)
    test = impute_missing(test)

    train = remove_outliers(train)

    train = feature_engineering(train)
    test = feature_engineering(test)

    train = remove_noise_rows(train)

    train = drop_columns(train, is_train=True)
    test = drop_columns(test, is_train=False)

    train = encode_features(train)
    test = encode_features(test)

    return train, test


def prepare_xy(train):
    y = train["is_promoted"]
    X = train.drop(["is_promoted"], axis=1)
    return X, y


def get_feature_names():
    return [
        "department",
        "education",
        "gender",
        "no_of_trainings",
        "age",
        "previous_year_rating",
        "length_of_service",
        "awards_won?",
        "avg_training_score",
        "KPIs_met_80",
        "sum_metric",
        "total_score",
    ]
