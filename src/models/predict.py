# src/models/predict.py

import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from src.features.feature_engineering import add_features


def load_model(model_path: str = "models/GradientBoosting_best.pkl"):
    data = joblib.load(model_path)
    return data["model"], data["threshold"]


def predict_churn(model, X, threshold: float = 0.5):
    """
    Dùng nội bộ trong pipeline train/evaluate.
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    return y_pred, y_proba


def predict(customer_dict: dict, model_path: str = "models/GradientBoosting_best.pkl"):
    """
    Dùng cho FastAPI endpoint.
    Nhận dict từ request → trả về kết quả dự đoán.
    """
    # Load model + threshold đã lưu
    model, threshold = load_model(model_path)

    # Chuyển dict → DataFrame
    df = pd.DataFrame([customer_dict])

    # Áp dụng feature engineering như lúc train
    df = add_features(df)

    # Dự đoán
    y_proba = model.predict_proba(df)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    return {
        "churn_prediction": int(y_pred[0]),
        "churn_probability": round(float(y_proba[0]), 4),
        "risk_level": (
            "CAO"    if y_proba[0] >= 0.6 else
            "TRUNG"  if y_proba[0] >= 0.4 else
            "THAP"
        ),
        "threshold_used": threshold
    }