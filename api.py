from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer

app = FastAPI()

# =========================
# LOAD MODEL (FIXED)
# =========================
model = load_model(
    "model_fixed.h5",
    compile=False,
    safe_mode=False,
    custom_objects={"InputLayer": InputLayer}
)

scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

SEQ_LEN = 10
THRESHOLD = 0.4

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: InputData):
    try:
        x = np.array(data.features).reshape(1, -1)

        x = scaler.transform(x)
        x = selector.transform(x)

        x_seq = np.repeat(x[:, np.newaxis, :], SEQ_LEN, axis=1)

        prob = model.predict(x_seq)[0][0]
        label = "ATTACK" if prob >= THRESHOLD else "BENIGN"

        return {
            "label": label,
            "confidence": float(prob)
        }

    except Exception as e:
        return {"error": str(e)}
