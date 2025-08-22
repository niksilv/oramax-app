from functools import lru_cache
from pathlib import Path
import json, re
import numpy as np
from xgboost import XGBClassifier

MODEL_DIR = Path(__file__).parent / "model"

def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def _safe_int(x, default=0):
    try:
        v = float(x)
        return int(round(v)) if np.isfinite(v) else default
    except Exception:
        return default

def _basic_stats(flux: np.ndarray) -> dict:
    if flux.size == 0:
        return {"std": 0.0, "mad": 0.0, "max_drop": 0.0, "acf1": 0.0}
    f = flux / np.median(flux) if np.median(flux) != 0 else flux
    std = float(np.std(f))
    mad = float(np.median(np.abs(f - np.median(f))))
    max_drop = float(max(0.0, 1.0 - float(np.min(f))))
    acf1 = 0.0
    if f.size > 2:
        c = np.corrcoef(f[:-1], f[1:])
        if np.isfinite(c[0,1]):
            acf1 = float(c[0,1])
    return {"std": std, "mad": mad, "max_drop": max_drop, "acf1": acf1}

def tls_features(time: np.ndarray, flux: np.ndarray) -> dict:
    m = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[m], flux[m]
    feats = _basic_stats(flux)

    if len(time) < 400:
        feats.update({
            "tls_SDE": 0.0, "tls_period": 0.0, "tls_duration": 0.0,
            "tls_depth": 0.0, "tls_snr": 0.0, "tls_transit_count": 0
        })
        return feats

    try:
        from transitleastsquares import transitleastsquares  # lazy import
        model = transitleastsquares(time, flux)
        res = model.power()
        feats.update({
            "tls_SDE": _safe_float(getattr(res, "SDE", 0.0)),
            "tls_period": _safe_float(getattr(res, "period", 0.0)),
            "tls_duration": _safe_float(getattr(res, "duration", 0.0)),
            "tls_depth": _safe_float(getattr(res, "depth", 0.0)),
            "tls_snr": _safe_float(getattr(res, "snr", 0.0)),
            "tls_transit_count": _safe_int(getattr(res, "transit_count", 0)),
        })
    except Exception:
        feats.update({
            "tls_SDE": 0.0, "tls_period": 0.0, "tls_duration": 0.0,
            "tls_depth": 0.0, "tls_snr": 0.0, "tls_transit_count": 0
        })
    return feats

def read_flux_text(text: str) -> np.ndarray:
    tokens = re.split(r"[,\s]+", text.strip())
    vals = [float(tok) for tok in tokens if tok]
    return np.asarray(vals, dtype=float)

def read_flux_file(path: str) -> np.ndarray:
    txt = Path(path).read_text(encoding="utf-8-sig")
    return read_flux_text(txt)

@lru_cache(maxsize=1)
def load_model():
    clf = XGBClassifier()
    clf.load_model(MODEL_DIR / "xgb_model.json")
    feat_order = json.loads((MODEL_DIR / "feature_order.json").read_text())
    return clf, feat_order

def predict_from_array(flux: np.ndarray, cadence_min: float = 2.0) -> dict:
    t = np.arange(len(flux)) * (cadence_min / (60*24))
    feats = tls_features(t, flux)
    clf, feat_order = load_model()
    X = np.array([[feats.get(k, 0.0) for k in feat_order]], dtype=float)
    prob = float(clf.predict_proba(X)[0, 1])
    return {"planet_prob": prob, **feats}
