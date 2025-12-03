from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI()

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

