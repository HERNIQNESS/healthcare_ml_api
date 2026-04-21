from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import pandas as pd
import os

from app.schemas import PatientData

app = FastAPI()

# ✅ CORS (frontend <-> backend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model + encoders safely
model = joblib.load("models/model.joblib")
encoders = joblib.load("models/encoders.joblib")

# ✅ Serve frontend (THIS is what you were missing)
@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join("frontend", "index.html"))

# ✅ Optional: serve static files if needed later
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# ✅ Prediction endpoint
@app.post("/predict")
def predict(data: PatientData):
    try:
        # Convert input → DataFrame
        df = pd.DataFrame([data.dict()])

        # 🔥 FORCE correct feature order (this avoids XGBoost errors)
        df = df[[
            "Age", "Gender", "Blood Type", "Medical Condition",
            "Insurance Provider", "Billing Amount",
            "Admission Type", "Medication", "Length of Stay"
        ]]

        # 🔥 Apply encoders safely
        for col in df.columns:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])

        # 🔥 Predict
        prediction = model.predict(df)[0]

        # 🔥 Map output
        mapping = {
            0: "normal",
            1: "abnormal",
            2: "inconclusive"
        }

        return {"predicted_test_result": mapping[int(prediction)]}

    except Exception as e:
        return {"error": str(e)}