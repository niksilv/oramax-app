# main.py
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re

app = FastAPI(title="Orama X API")

# --- CORS (ώστε να παίζει από landing) ---
origins = [
    "https://www.oramax.space",
    "https://oramax.space",
    "https://oramax-landing.vercel.app",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # αν θες όλα: ["*"]
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class PredictRequest(BaseModel):
    lightcurve: List[float]

class PredictResponse(BaseModel):
    planet_prob: float
    features: Optional[Dict[str, float]] = None

# ---------- Health ----------
@app.get("/healthz")
def healthz():
    return {"ok": True, "version": "0.1.0"}

# ---------- Helper  ----------
def predict_from_array(values: List[float]) -> (float, Dict[str, float]):
    """
    Ελαφρύ baseline χωρίς βαριές εξαρτήσεις.
    Υπολογίζει μερικά απλά features και μια "προβλεπόμενη" πιθανότητα.
    """
    if len(values) < 3:
        raise HTTPException(status_code=400, detail="Not enough data points.")

    # lazy imports (ελαφριά μεν, αλλά δείγμα πρακτικής)
    import math
    import statistics

    mean = sum(values) / len(values)
    std = statistics.pstdev(values)
    mad = statistics.median([abs(x - statistics.median(values)) for x in values])
    max_drop = max(0.0, 1.0 - min(values))  # πόσο κάτω από το 1.0 φτάσαμε

    # πολύ απλή «λογιστική» συνάρτηση για demo
    logit = -2.5 + 12.0 * max_drop - 1.5 * std
    prob = 1.0 / (1.0 + math.exp(-logit))

    feats = {
        "mean": mean,
        "std": std,
        "mad": mad,
        "max_drop": max_drop,
    }
    return float(prob), {k: float(v) for k, v in feats.items()}

# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"status": "ok", "app": "Orama X", "message": "Hello from app.oramax.space!"}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    prob, feats = predict_from_array(req.lightcurve)
    return PredictResponse(planet_prob=prob, features=feats)

@app.post("/predict-file", response_model=PredictResponse)
async def predict_file(file: UploadFile = File(...)):
    """
    Δέχεται .txt/.csv με τιμές χωρισμένες με κόμμα ή κενά/νέα γραμμή.
    """
    try:
        raw = (await file.read()).decode("utf-8-sig", errors="ignore")
        tokens = [t for t in re.split(r"[,\s]+", raw.strip()) if t]
        values = [float(t) for t in tokens]
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse file. Use comma or whitespace separated floats.")

    if len(values) < 3:
        raise HTTPException(status_code=400, detail="Not enough data points in file.")

    prob, feats = predict_from_array(values)
    return PredictResponse(planet_prob=prob, features=feats)
