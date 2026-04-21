from fastapi import FastAPI
import joblib
import os

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


@app.post("/predict")
def predict(data: list):
    if model is None:
        return {"error": "Model is not loaded on the server."}
    prediction = model.predict([data])
    return {"prediction": prediction.tolist()}
