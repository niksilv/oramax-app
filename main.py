from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import importlib

app = FastAPI(title="Orama X API", version="0.1.0")

origins = ["https://www.oramax.space", "https://oramax.space", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "app": "Orama X", "message": "Hello from app.oramax.space!"}

@app.get("/healthz")
def healthz():
    # Καθόλου ML εδώ – πράσινο αν ο server τρέχει
    return {"ok": True, "version": "0.1.0"}

class PredictBody(BaseModel):
    lightcurve: List[float] = Field(min_length=5)

def _infer_module():
    # Lazy import για να μην μπλοκάρει το startup
    return importlib.import_module("inference")

@app.post("/predict")
def predict(body: PredictBody):
    try:
        arr = np.asarray(body.lightcurve, dtype=float)
        m = _infer_module()
        return m.predict_from_array(arr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    try:
        content = (await file.read()).decode("utf-8-sig")
        m = _infer_module()
        arr = np.asarray(m.read_flux_text(content), dtype=float)
        return m.predict_from_array(arr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
