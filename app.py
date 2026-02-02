from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd  # Add for DataFrame

app = FastAPI(title="Credit Card Fraud Detection API")

# Load model and scaler (use new file names)
model = joblib.load("model/xgb_model.pkl")
scaler = joblib.load("model/scaler.pkl")

THRESHOLD = 0.30  # business-optimized threshold

class Transaction(BaseModel):
    features: list

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        features = transaction.features

        if len(features) != 30:
            return {"error": f"Expected 30 features (Time + V1-V28 + Amount), got {len(features)}"}

        # Define column names for correct order
        columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

        # Convert list to DataFrame with proper columns/order
        df_input = pd.DataFrame([features], columns=columns)

        # Scale Time & Amount
        df_input[['Time', 'Amount']] = scaler.transform(df_input[['Time', 'Amount']])

        # To numpy array
        X = df_input.values  # Shape (1, 30)

        prob = model.predict_proba(X)[0][1]
        label = int(prob >= THRESHOLD)

        return {
            "fraud_probability": round(float(prob), 4),
            "fraud_prediction": label,
            "decision": "Fraud" if label else "Legitimate"
        }

    except Exception as e:
        return {"error": str(e)}