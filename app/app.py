from fastapi import FastAPI
import joblib
import os

app = FastAPI()


BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
model_path = os.path.join(BASE_DIR, "model", "model.pkl")

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

@app.post("/predict")
def predict(data: list):
    prediction = model.predict([data])
    return {"prediction": prediction.tolist()}