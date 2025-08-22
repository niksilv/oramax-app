from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

APP_NAME = "Orama X API"
APP_VERSION = "0.1.0"

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# Επιτρεπόμενες προελεύσεις (site + τοπική ανάπτυξη)
origins = [
    "https://www.oramax.space",
    "https://oramax.space",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Utilities (placeholder "μοντέλο") --------
def model_inference(lightcurve: List[float]) -> float:
    """
    Placeholder λογική: υπολογίζει μια pseudo-πιθανότητα με βάση
    το "βάθος" (min vs. μέση τιμή) και τον θόρυβο (std-like).
    Αντικατάστησέ το με πραγματικό μοντέλο (π.χ. PyTorch/TF).
    """
    if not lightcurve:
        return 0.0
    n = len(lightcurve)
    mean = sum(lightcurve) / n
    var = sum((x - mean) ** 2 for x in lightcurve) / n
    std = var ** 0.5 if var > 0 else 1e-6
    depth = max(0.0, mean - min(lightcurve))
    snr = depth / (std + 1e-6)
    # squash σε [0,1] με "sigmoid"-like συνάρτηση (χωρίς imports)
    prob = 1.0 / (1.0 + (2.718281828 ** (-snr)))
    # clamp
    if prob < 0.0:
        prob = 0.0
    if prob > 1.0:
        prob = 1.0
    return float(prob)


# -------- Schemas --------
class PredictRequest(BaseModel):
    lightcurve: List[float]  # π.χ. [1.0, 0.998, 1.001, ...]


class PredictResponse(BaseModel):
    planet_prob: float


# -------- Routes --------
@app.get("/")
def root():
    return {
        "status": "ok",
        "app": "Orama X",
        "message": "Hello from app.oramax.space!",
        "version": APP_VERSION,
    }


@app.head("/")
def head_root():
    # Για να μην γεμίζουν τα logs με 405 από health checks
    return Response(status_code=200)


@app.get("/healthz")
def healthz():
    return {"ok": True, "version": APP_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    prob = model_inference(req.lightcurve)
    return {"planet_prob": prob}


@app.post("/predict-file", response_model=PredictResponse)
async def predict_file(file: UploadFile = File(...)):
    """
    Δέχεται .csv ή .txt με αριθμούς (comma/space/newline separated).
    Παράδειγμα περιεχόμενου:
      1.0, 0.998, 1.002, 0.997
    ή
      1.0
      0.998
      1.002
      0.997
    """
    try:
        raw = (await file.read()).decode("utf-8", errors="ignore")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read file")

    # Parse σε λίστα float (χωρίς numpy για να μην απαιτούνται extra deps)
    tokens = []
    for line in raw.replace(",", " ").split():
        try:
            tokens.append(float(line.strip()))
        except ValueError:
            # Αγνόησε σκουπίδια/κενά tokens
            continue

    if not tokens:
        raise HTTPException(status_code=400, detail="No numeric values found")

    prob = model_inference(tokens)
    return {"planet_prob": prob}


# Το uvicorn ξεκινά από το Docker CMD στο Fly.io.
# Για τοπικό test μπορείς να τρέξεις:
#   uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

