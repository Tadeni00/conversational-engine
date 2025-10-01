# services/fasttext_service/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from typing import Optional, Tuple

logger = logging.getLogger("fasttext_service")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH", "/models/lid.176.ftz")
FASTTEXT_TIMEOUT = float(os.getenv("FASTTEXT_TIMEOUT", "2.0"))

app = FastAPI(title="fasttext-service")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    lang: str
    score: float

_fasttext = None

def load_model_if_needed():
    global _fasttext
    if _fasttext is not None:
        return
    try:
        import fasttext  # type: ignore
        if not os.path.exists(FASTTEXT_MODEL_PATH):
            logger.error("fasttext model file not found: %s", FASTTEXT_MODEL_PATH)
            _fasttext = None
            return
        _fasttext = fasttext.load_model(FASTTEXT_MODEL_PATH)
        logger.info("fasttext model loaded from %s", FASTTEXT_MODEL_PATH)
    except Exception as e:
        logger.exception("fasttext load failed: %s", e)
        _fasttext = None

@app.on_event("startup")
def startup_event():
    load_model_if_needed()

def _predict_local(text: str) -> Optional[Tuple[str, float]]:
    load_model_if_needed()
    if _fasttext is None:
        return None
    try:
        labels, probs = _fasttext.predict(text.replace("\n", " "), k=1)
        if labels and probs:
            label = labels[0].replace("__label__", "")
            score = float(probs[0])
            return label, score
    except Exception:
        logger.exception("fasttext predict failed")
    return None

# Very small marker fallback (if model fails). Conservative.
_MARKERS_PCM = {"abeg","abi","dey","how","much","biko","wahala"}
def _marker_heuristic(text: str) -> Tuple[str, float]:
    s = text.lower()
    for m in _MARKERS_PCM:
        if m in s:
            return "pcm", 0.6
    return "en", 0.6

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    # Try local fasttext model
    out = _predict_local(text)
    if out:
        lang, score = out
        return PredictResponse(lang=lang, score=score)
    # fallback heuristic when no model available
    lang, score = _marker_heuristic(text)
    return PredictResponse(lang=lang, score=score)
