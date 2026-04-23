from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import pandas as pd
import os

from app.schemas import PatientData

app = FastAPI()


# CORS (frontend <-> backend)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#  Load model + encoders safely

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "model.joblib")
encoder_path = os.path.join(BASE_DIR, "models", "encoders.joblib")

model = joblib.load(model_path)
encoders = joblib.load(encoder_path)


#  Serve frontend

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

app.mount("/frontend", StaticFiles(directory=os.path.join(BASE_DIR, "frontend")), name="frontend")


# Prediction endpoint 

@app.post("/predict")
def predict(data: PatientData):
    try:
        # Convert input → DataFrame
        df = pd.DataFrame([data.dict()])

        #  Normalize inputs
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

        #  Apply encoders safely
        for col in df.columns:
            if col in encoders:
                try:
                    df[col] = encoders[col].transform(df[col])
                except Exception:
                    return {
                        "error": f"Invalid value '{df[col].iloc[0]}' for column '{col}'"
                    }

        #  Predict
        prediction = model.predict(df)[0]

        
        #  DEBUG OUTPUT 
        
        return {
            "raw_prediction": str(prediction),
            "type": str(type(prediction))
        }

    except Exception as e:
        return {
            "error": str(e)
        }