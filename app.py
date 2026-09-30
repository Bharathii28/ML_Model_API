
from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import joblib

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0"
)

# Load trained model
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")


class CustomerData(BaseModel):
    age: int = Field(..., ge=18, le=100)
    tenure: int = Field(..., ge=1, le=20)
    monthly_charges: float = Field(..., gt=0)
    support_calls: int = Field(..., ge=0, le=50)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running",
        "status": "success"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_customer_churn(customer: CustomerData):
    features = np.array([[
        customer.age,
        customer.tenure,
        customer.monthly_charges,
        customer.support_calls
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    if prediction == 1:
        result = "Customer is likely to churn"
    else:
        result = "Customer is unlikely to churn"

    return {
        "prediction": int(prediction),
        "result": result,
        "churn_probability": round(float(probability[1]), 4),
        "no_churn_probability": round(float(probability[0]), 4)
    }