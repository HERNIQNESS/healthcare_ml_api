from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import pandas as pd
import os

from app.schemas import PatientData

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + encoders
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "model.joblib")
encoder_path = os.path.join(BASE_DIR, "models", "encoders.joblib")

model = joblib.load(model_path)
encoders = joblib.load(encoder_path)

# Serve frontend
@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

app.mount("/frontend", StaticFiles(directory=os.path.join(BASE_DIR, "frontend")), name="frontend")


# Prediction endpoint
@app.post("/predict")
def predict(data: PatientData):
    try:
        df = pd.DataFrame([data.dict()])

        # Normalize inputs
        df["Gender"] = df["Gender"].str.strip().str.title()
        df["Blood Type"] = df["Blood Type"].str.strip().str.upper()
        df["Medical Condition"] = df["Medical Condition"].str.strip().str.title()
        df["Insurance Provider"] = df["Insurance Provider"].str.strip().str.title()
        df["Admission Type"] = df["Admission Type"].str.strip().str.title()
        df["Medication"] = df["Medication"].str.strip().str.title()

        # Force correct feature order
        df = df[[
            "Age", "Gender", "Blood Type", "Medical Condition",
            "Insurance Provider", "Billing Amount",
            "Admission Type", "Medication", "Length of Stay"
        ]]

        # Apply label encoders to categorical columns
        for col in df.columns:
            if col in encoders:
                le = encoders[col]
                val = df[col].iloc[0]
                if val in le.classes_:
                    df[col] = le.transform(df[col])
                else:
                    # Unknown value: use the most common class (index 0)
                    df[col] = 0

        # Predict — model returns an encoded integer
        raw_pred = model.predict(df)[0]

        # Decode the integer back to a label (Normal / Abnormal / Inconclusive)
        if "Test Results" in encoders:
            label = encoders["Test Results"].inverse_transform([int(raw_pred)])[0]
        elif "test_results" in encoders:
            label = encoders["test_results"].inverse_transform([int(raw_pred)])[0]
        else:
            # Fallback: manual mapping (check your training script to confirm order)
            mapping = {0: "Abnormal", 1: "Inconclusive", 2: "Normal"}
            label = mapping.get(int(raw_pred), str(raw_pred))

        return {"predicted_test_result": label}

    except Exception as e:
        return {"error": str(e)}