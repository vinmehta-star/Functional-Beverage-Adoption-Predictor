from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List
from fastapi.staticfiles import StaticFiles
import os
import mimetypes
import warnings

# Suppress scikit-learn version and feature name warnings
warnings.filterwarnings("ignore")

# Fix Windows MIME type registry issue for static JS/CSS files
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"Warning: Could not load model or scaler. Place model.pkl and scaler.pkl in the backend folder. Error: {e}")

class PredictRequest(BaseModel):
    responses: dict

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictRequest):
    if not model or not scaler:
        return {"error": "Model or scaler not loaded. Please place model.pkl and scaler.pkl in the backend folder and restart."}
    
    expected_keys = [
        "HO1", "HO2", "HO3", "HO4",
        "AT1", "AT2", "AT3",
        "BR1", "BR2", "BR3", "BR4",
        "SN1", "SN2", "SN4",
        "PBC1", "PBC2", "PBC3", "PBC4",
        "INNO1", "INNO2", "INNO4"
    ]
    
    # Map dictionary to exact 21 features, using neutral 3.0 for any missing questions
    inputs = [float(request.responses.get(k, 3.0)) for k in expected_keys]
    
    # 4. Compute section_scores (raw average per section, before reverse coding)
    section_scores = {
        "HO": float(np.mean(inputs[0:4])),
        "AT": float(np.mean(inputs[4:7])),
        "BR": float(np.mean(inputs[7:11])),
        "SN": float(np.mean(inputs[11:14])),
        "PBC": float(np.mean(inputs[14:18])),
        "INNO": float(np.mean(inputs[18:21]))
    }
    
    # 1. Apply reverse coding (6 - value) to corresponding indices:
    # AT1(4), AT3(6), BR1(7), BR4(10), SN4(13), INNO1(18), INNO4(20)
    reverse_indices = [4, 6, 7, 10, 13, 18, 20]
    processed_inputs = list(inputs)
    for idx in reverse_indices:
        processed_inputs[idx] = 6.0 - processed_inputs[idx]
        
    # 2. Scale using the loaded scaler
    X = np.array([processed_inputs])
    X_scaled = scaler.transform(X)
    
    # 3. Run model.predict() and model.predict_proba()
    prediction = model.predict(X_scaled)[0]
    
    # The probability is typically [prob_0, prob_1]
    # In the training data, y=1 represents NON-ADOPTERS, and y=0 represents ADOPTERS.
    # proba[0] = probability of being an Adopter (class 0)
    # proba[1] = probability of being a Non-Adopter (class 1)
    proba = model.predict_proba(X_scaled)[0]

    if int(prediction) == 0:
        # Model predicted Adopter — show adopter confidence
        label = "Likely Adopter"
        probability = float(proba[0]) if len(proba) > 1 else float(proba[0])
    else:
        # Model predicted Non-Adopter — show non-adopter confidence
        label = "Non-Adopter"
        probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
    
    # 5. Return JSON
    return {
        "label": label,
        "probability": probability,
        "section_scores": section_scores
    }

# Resolve the absolute path to the local static folder dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount static files (HTML/CSS/JS frontend)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    import sys
    # Dynamically add the backend directory to sys.path so uvicorn can find 'main:app'
    sys.path.insert(0, BASE_DIR)
    # Start the server on port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
