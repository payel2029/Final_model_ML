from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI()

# ---------------- CORS ----------------
origins = [
    "https://ornate-panda-3e6e34.netlify.app",  # Netlify site URL
    "http://localhost:5173",                     # local dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------------------

# Load updated pipeline
model = joblib.load("maternal_LGBM02_pipeline.pkl")

# ---------------- Input Schema ----------------
class InputData(BaseModel):
    Age: float
    Systolic_BP: float = Field(..., alias="Systolic BP")
    Diastolic: float
    BS: float
    Body_Temp: float = Field(..., alias="Body Temp")
    BMI: float
    Previous_Complications: int = Field(..., alias="Previous Complications")
    Preexisting_Diabetes: int = Field(..., alias="Preexisting Diabetes")
    Gestational_Diabetes: int = Field(..., alias="Gestational Diabetes")
    Mental_Health: int = Field(..., alias="Mental Health")
    Heart_Rate: float = Field(..., alias="Heart Rate")

    class Config:
        allow_population_by_field_name = True

# ---------------- Prediction Endpoint ----------------
@app.post("/predict")
def predict(data: InputData):
    # Create DataFrame with correct column names
    X = pd.DataFrame([{
        "Age": data.Age,
        "Systolic BP": data.Systolic_BP,
        "Diastolic": data.Diastolic,
        "BS": data.BS,
        "Body Temp": data.Body_Temp,
        "BMI": data.BMI,
        "Previous Complications": data.Previous_Complications,
        "Preexisting Diabetes": data.Preexisting_Diabetes,
        "Gestational Diabetes": data.Gestational_Diabetes,
        "Mental Health": data.Mental_Health,
        "Heart Rate": data.Heart_Rate
    }])

    # Predict
    pred_class = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]

    return {
        "prediction": int(pred_class),
        "probabilities": pred_proba.tolist()   # convert numpy array to list for JSON
    }

