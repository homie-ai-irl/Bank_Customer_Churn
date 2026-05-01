from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from src.api.schemas import Customer
from src.models.predict import predict

app = FastAPI(title="Customer Churn Prediction API")

# mount static (CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
def predict_api(customer: Customer):
    return predict(customer.dict())