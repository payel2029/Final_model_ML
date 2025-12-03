from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI()

# ---------------- CORS ----------------
origins = [
    "https://ornate-panda-3e6e34.netlify.app",  # Netlify site URL
    "http://localhost:5173",  # optional local dev
    # "*" can be used to allow all origins (not recommended for production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------------------

# Load model
model = joblib.load("maternal_pipeline.pkl")

class InputData(BaseModel):
    Age: float = Field(..., alias="Age")
    Body_Temp: float = Field(..., alias="Body Temp")
    Systolic_BP: float = Field(..., alias="Systolic BP")
    Heart_Rate: float = Field(..., alias="Heart Rate")
    Diastolic: float = Field(..., alias="Diastolic")
    BS: float = Field(..., alias="BS")

    class Config:
        allow_population_by_field_name = True

@app.post("/predict")
def predict(data: InputData):
    X = np.array([
        data.Age,
        data.Body_Temp,
        data.Systolic_BP,
        data.Heart_Rate,
        data.Diastolic,
        data.BS
    ]).reshape(1, -1)

    pred = model.predict(X)[0]

    return {"prediction": int(pred)}
