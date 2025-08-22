# προσθέσεις στην κορυφή αν δεν υπάρχουν ήδη:
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Orama X API")

class PredictResponse(BaseModel):
    planet_prob: float

class LightCurveRequest(BaseModel):
    # δέξου οποιαδήποτε λίστα floats και έλεγξε το μήκος εσύ (όχι με validator)
    lightcurve: List[float]

# -------------------------------
# /predict (JSON)
# -------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict_json(req: LightCurveRequest):
    values = [float(x) for x in req.lightcurve]
    if len(values) < 3:
        # δώσε ξεκάθαρο μήνυμα στον χρήστη
        raise HTTPException(status_code=400, detail="Need at least 3 numbers.")
    prob, _feats = predict_from_array(values)  # κράτα τη δική σου predict_from_array
    return {"planet_prob": float(prob)}

# -------------------------------
# /predict-file (multipart/form-data)
# -------------------------------
@app.post("/predict-file", response_model=PredictResponse)
async def predict_file(file: UploadFile = File(...)):
    # Δέξου UTF-8 με BOM, αγνόησε περίεργους χαρακτήρες
    raw = (await file.read()).decode("utf-8-sig", errors="ignore").strip()
    # Σπάσε σε κόμματα ή/και κενά (π.χ. "1, 0.99 1.01")
    tokens = re.split(r"[,\s]+", raw)
    values: List[float] = []
    for t in tokens:
        if not t:
            continue
        try:
            values.append(float(t))
        except ValueError:
            # αγνόησε ό,τι δεν είναι αριθμός
            pass

    if len(values) < 3:
        raise HTTPException(
            status_code=400,
            detail="Couldn't parse enough numbers from file (need ≥ 3, comma/space-separated).",
        )

    prob, _feats = predict_from_array(values)
    return {"planet_prob": float(prob)}
