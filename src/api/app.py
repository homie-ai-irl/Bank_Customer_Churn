# src/api/app.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from src.api.schemas import Customer
from src.models.predict import predict

app = FastAPI(title="Customer Churn Prediction API")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
def predict_api(customer: Customer):
    try:
        result = predict(customer.dict())
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model chua duoc train. Chay main.py truoc."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))