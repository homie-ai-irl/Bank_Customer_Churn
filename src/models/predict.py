import joblib
import pandas as pd
from src.features.feature_engineering import build_features

model = joblib.load("models/best_model.pkl")

def predict(data: dict):

    df = pd.DataFrame([data])
    df = build_features(df)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {"churn": int(pred), "probability": float(prob)}