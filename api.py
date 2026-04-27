from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("../model.pkl")
scaler = joblib.load("../scaler.pkl")

@app.post("/predict")
def predict(data: list):
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    return {
        "fraud": int(pred),
        "probability": float(prob)
    }