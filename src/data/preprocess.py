# src/data/preprocess.py

import warnings
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────
# Khai báo cột — bao gồm cả feature mới từ feature_engineering
# ─────────────────────────────────────────────────

DROP_COLS = ["customer_id"]
TARGET    = "churn"

# Cột số liên tục (gốc + feature mới)
NUMERICAL_COLS = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "estimated_salary",
    "balance_per_product",      # feature mới
    "balance_salary_ratio",     # feature mới
    "tenure_active",            # feature mới
]

# Cột ordinal
ORDINAL_COLS = [
    "products_number",
]

# Cột nhị phân (gốc + feature mới)
BINARY_COLS = [
    "credit_card",
    "active_member",
    "is_zero_balance",          # feature mới
]

# Cột categorical (gốc + feature mới)
CATEGORICAL_COLS = [
    "country",
    "gender",
    "age_group",                # feature mới
    "credit_group",             # feature mới
]


def load_data(filepath: str):
    df = pd.read_csv(filepath)
    df = df.drop(columns=DROP_COLS, errors="ignore")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print(f"  Dataset shape : {df.shape}")
    print(f"  Churn rate    : {y.mean():.2%} "
          f"({y.sum()} churned / {len(y)} total)")

    return X, y


def build_preprocessor():

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first"
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline,   NUMERICAL_COLS),
            ("ord", ordinal_pipeline,     ORDINAL_COLS),
            ("bin", binary_pipeline,      BINARY_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop"
    )

    return preprocessor