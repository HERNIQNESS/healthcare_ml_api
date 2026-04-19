from fastapi import FastAPI
import joblib
import pandas as pd
from app.schemas import PatientData

app = FastAPI()

# Load trained model and encoders
model = joblib.load("models/model.joblib")
encoders = joblib.load("models/encoders.joblib")

@app.get("/")
def home():
    return {"message": "Healthcare ML API is running"}

@app.post("/predict")
def predict(data: PatientData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Rename columns to match training data
    df.columns = [
        "Age",
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Billing Amount",
        "Admission Type",
        "Insurance Provider",
        "Medication",
        "Length of Stay"
    ]

    # Apply encoders
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    # ✅ FORCE CORRECT COLUMN ORDER (THIS FIXES YOUR ERROR)
    df = df[[
        "Age",
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Insurance Provider",
        "Billing Amount",
        "Admission Type",
        "Medication",
        "Length of Stay"
    ]]

    # Predict
    prediction = model.predict(df)[0]

    # Map output
    mapping = {
        0: "normal",
        1: "abnormal",
        2: "inconclusive"
    }

    return {"predicted_test_result": mapping[int(prediction)]}