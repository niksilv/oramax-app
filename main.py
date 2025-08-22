# main.py
from typing import List, Dict, Optional, Union
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import re
import math
import statistics

app = FastAPI(title="Orama X API")

# CORS
origins = [
    "https://www.oramax.space",
    "https://oramax.space",
    "https://oramax-landing.vercel.app",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # ή ["*"] αν θέλεις προσωρινά χωρίς περιορισμό
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Models --------
class PredictRequest(BaseModel):
    # επιτρέπουμε είτε λίστα floats είτε string με κόμματα/κενά
    lightcurve: Union[List[float], str]

    @field_validator("lightcurve", mode="before")
    @classmethod
    def _parse_str_if_needed(cls, v):
        if isinstance(v, str):
            tokens = [t for t in re.split(r"[,\s]+", v.strip()) if t]
            try:
                return [float(t) for t in tokens]
            except Exception:
                raise ValueError("Provide floats or a comma/space separated string")
        return v

class PredictResponse(BaseModel):
    planet_prob: float
    features: Optional[Dict[str, float]] = None

# -------- Health --------
@app.get("/healthz")
def healthz():
    return {"ok": True, "version": "0.1.0"}

# -------- Helper --------
def predict_from_array(values: List[float]) -> (float, Dict[str, float]):
    """Ελαφρύ baseline· υπολογίζει απλά features και μια pseudo-πιθανότητα."""
    if not values:
        raise HTTPException(status_code=400, detail="Empty light curve.")

    # Αν έχουμε λίγα σημεία, «μαξιλάρουμε» για σταθερούς υπολογισμούς
    if len(values) < 3:
        values = values + [values[-1]] * (3 - len(values))

    mean = sum(values) / len(values)
    try:
        std = statistics.pstdev(values)    # ορισμός πληθυσμιακής τυπικής απόκλισης (δεν σκάει σε μικρά δείγματα)
    except Exception:
        std = 0.0
    med = statistics.median(values)
    mad = statistics.median([abs(x - med) for x in values])
    max_drop = max(0.0, 1.0 - min(values))  # πόσο κάτω από το 1.0 φτάσαμε

    # απλή «λογιστική» για demo
    logit = -2.5 + 12.0 * max_drop - 1.5 * std
    prob = 1.0 / (1.0 + math.exp(-logit))

    feats = {"mean": mean, "std": std, "mad": mad, "max_drop": max_drop}
    return float(prob), {k: float(v) for k, v in feats.items()}

# -------- Endpoints --------
@app.get("/")
def root():
    return {"status": "ok", "app": "Orama X", "message": "Hello from app.oramax.space!"}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    values = req.lightcurve if isinstance(req.lightcurve, list) else []
    prob, feats = predict_from_array(values)
    return PredictResponse(planet_prob=prob, features=feats)

@app.post("/predict-file", response_model=PredictResponse)
async def predict_file(file: UploadFile = File(...)):
    """Δέχεται .txt/.csv με τιμές χωρισμένες με κόμματα/κενά/γραμμές."""
    try:
        raw = (await file.read()).decode("utf-8-sig", errors="ignore")
        tokens = [t for t in re.split(r"[,\s]+", raw.strip()) if t]
        values = [float(t) for t in tokens]
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse file. Use comma or whitespace separated floats.")
    prob, feats = predict_from_array(values)
    return PredictResponse(planet_prob=prob, features=feats)

if __name__ == "__main__":
    import os
    import uvicorn
    # Τρέξε τον server ακριβώς στη 0.0.0.0:8080 (ή PORT του Fly αν οριστεί)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        log_level="info",
    )
