# services/negotiation_service/app.py
import os
import uuid
import hashlib
import logging
from typing import Optional, Dict, Any

import requests
from fastapi import FastAPI
from pydantic import BaseModel

# Try flexible imports for llm_phraser:
# - first try package-style import (negotiation_service.llm_phraser)
# - fallback to top-level llm_phraser.py (for dev mounts)
try:
    from negotiation_service.llm_phraser import phrase  # type: ignore
except Exception:
    try:
        from llm_phraser import phrase  # type: ignore
    except Exception:
        phrase = None  # type: ignore

logger = logging.getLogger("negotiation")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Config
LANG_DETECT_URL = os.getenv("LANGUAGE_DETECTION_URL", "http://language-detection:8000/detect")
LANG_DETECT_TIMEOUT = float(os.getenv("LANG_DETECT_TIMEOUT", "0.5"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
DEFAULT_MIN_PRICE_RATIO = float(os.getenv("DEFAULT_MIN_PRICE_RATIO", "0.5"))

# Requests session
_session = requests.Session()

# Pydantic models
class ProductPayload(BaseModel):
    id: str
    name: str
    base_price: int
    min_price: Optional[int] = None

class DecideRequest(BaseModel):
    offer: Optional[int] = None
    product: ProductPayload
    state: Optional[Dict[str, Any]] = None

class DecideResponse(BaseModel):
    action: str
    price: Optional[int] = None
    confidence: float = 1.0
    strategy_tag: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    reply: Optional[str] = None

# Helpers
def detect_language_via_service(text: Optional[str]) -> (str, float):
    """
    POSTs to the language detection service at /detect.
    Expects JSON response { language: "en", confidence: 0.75 }.
    Falls back to ("en", 0.5) on any error.
    """
    if not text:
        return "en", 0.5
    try:
        resp = _session.post(LANG_DETECT_URL, json={"text": text}, timeout=LANG_DETECT_TIMEOUT)
        resp.raise_for_status()
        j = resp.json()
        lang = j.get("language") or j.get("lang") or "en"
        conf = float(j.get("confidence", j.get("score", 0.5)))
        # clamp confidence
        if conf < 0.0:
            conf = 0.0
        if conf > 1.0:
            conf = 1.0
        return str(lang), conf
    except Exception:
        logger.exception("[negotiation] language detection service call failed")
        return "en", 0.5

def _enforce_price_guard(resp: DecideResponse, product: ProductPayload) -> DecideResponse:
    """
    Ensures the returned response does not propose a price below merchant's min or default min ratio.
    If violation => escalate with meta.min_price set.
    """
    try:
        proposed = resp.price
        base = product.base_price
        if proposed is None:
            return resp
        if product.min_price and proposed < product.min_price:
            return DecideResponse(
                action="ESCALATE", price=proposed, confidence=resp.confidence,
                strategy_tag="merchant_min_violation", meta={"min_price": product.min_price}
            )
        default_min = int(base * DEFAULT_MIN_PRICE_RATIO)
        if proposed < default_min:
            return DecideResponse(
                action="ESCALATE", price=proposed, confidence=resp.confidence,
                strategy_tag="min_ratio_violation", meta={"min_price": default_min}
            )
        return resp
    except Exception:
        logger.exception("[negotiation] price guard error")
        return DecideResponse(action="ESCALATE", price=resp.price, confidence=resp.confidence, strategy_tag="guard_error")

def _generate_decision_id(user_id: Optional[str] = None) -> str:
    base_id = str(uuid.uuid4())
    if user_id:
        hash_id = hashlib.sha256(str(user_id).encode()).hexdigest()[:8]
        return f"{base_id}-{hash_id}"
    return base_id

# Minimal fallback phrase() if llm_phraser import failed — safe template
def _fallback_phrase(decision: Dict[str, Any], product: Dict[str, Any], lang: str = "en", context: Optional[str] = None) -> str:
    action = (decision.get("action") or "").upper()
    price = decision.get("price") or product.get("base_price")
    if action == "ACCEPT":
        return f"Deal — ₦{int(price):,} for {product.get('name')}. Please pay now."
    if action == "COUNTER":
        return f"I can do ₦{int(price):,} for {product.get('name')}. Take it now!"
    if action == "ASK_CLARIFY":
        return f"What price are you thinking for {product.get('name')}?"
    return f"Our price is ₦{int(price):,} for {product.get('name')}."

# FastAPI app
app = FastAPI(title="negotiation")

@app.get("/ping")
def ping():
    try:
        resp = _session.post(LANG_DETECT_URL, json={"text": "ping"}, timeout=0.5)
        if resp.status_code == 200:
            return {"status": "ok"}
    except Exception:
        pass
    return {"status": "fail"}

@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest):
    # normalize state access
    state = req.state or {}
    user_id = state.get("user_id")
    decision_id = _generate_decision_id(user_id)

    buyer_text = (state.get("meta") or {}).get("buyer_text") if state else None
    lang, lang_conf = detect_language_via_service(buyer_text or f"{req.offer or ''} {req.product.name}")

    # Decision logic: basic default
    action = "COUNTER" if req.offer else "ASK_CLARIFY"
    price = req.offer if req.offer else req.product.base_price
    resp = DecideResponse(action=action, price=price)

    # Enforce guard (may convert to ESCALATE)
    resp = _enforce_price_guard(resp, req.product)

    # Use llm_phraser phrase() if available; otherwise fallback template
    try:
        if phrase is not None:
            reply_text = phrase(resp.dict(), req.product.dict(), lang=lang, context=buyer_text)
        else:
            logger.warning("[negotiation] llm_phraser not available; using fallback phrase")
            reply_text = _fallback_phrase(resp.dict(), req.product.dict(), lang=lang, context=buyer_text)
    except Exception as e:
        logger.exception("[negotiation] phrase generation failed: %s", e)
        reply_text = _fallback_phrase(resp.dict(), req.product.dict(), lang=lang, context=buyer_text)

    resp.reply = reply_text

    logger.info("negotiation_log: %s", {
        "decision_id": decision_id,
        "model_version": MODEL_VERSION,
        "user_id_hash": hashlib.sha256(str(user_id or "").encode()).hexdigest()[:8],
        "action": resp.action,
        "price": resp.price,
        "lang": lang,
        "lang_conf": lang_conf,
        "strategy_tag": resp.strategy_tag
    })

    return resp
