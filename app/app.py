import os

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()


BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "models", "model.pkl"),
    os.path.join(BASE_DIR, "Models", "model.pkl"),
    os.path.join(BASE_DIR, "model", "model.pkl"),
]
model_path = next((path for path in MODEL_CANDIDATES if os.path.exists(path)), MODEL_CANDIDATES[0])

print("Model path:", model_path)
print("Exists:", os.path.exists(model_path))

try:
    model = joblib.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)
    model = None


@app.get("/")
def home():
    return {"status": "Model is running"}


class PredictBody(BaseModel):
    data: list = Field(..., description="One row of feature values for the model")


@app.post("/predict")
def predict(body: PredictBody):
    if model is None:
        return {"error": "Model is not loaded on the server."}
    prediction = model.predict([body.data])
    return {"prediction": prediction.tolist()}
