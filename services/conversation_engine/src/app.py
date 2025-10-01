import os
import sys
import glob
import importlib
import importlib.util
import logging
from typing import Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from conversation_engine.core.language_pref import set_user_lang_pref, get_user_lang_pref, delete_user_lang_pref


logger = logging.getLogger("language-detection")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- normalization helper added ---
def _normalize_lang_code(s: Optional[str]) -> str:
    """
    Normalize a wide range of incoming language labels to one of:
      'en', 'pcm', 'yo', 'ha', 'ig'
    Accepts variants like 'pidgin', 'pcm_ng', 'yoruba', 'igbo', 'hausa', etc.
    """
    if not s:
        return "en"
    s0 = str(s).strip().lower()
    # English
    if s0 in ("en", "eng", "english", "en_us", "en-gb", "en-us"):
        return "en"
    # pidgin / pcm
    if s0 in ("pcm", "pidgin", "pidgin_ng", "pcm_ng", "pcm-nigeria", "pidgin-nigeria"):
        return "pcm"
    # yoruba
    if s0.startswith("yo") or s0 in ("yoruba", "yoruba_ng", "yo_ng"):
        return "yo"
    # hausa
    if s0.startswith("ha") or s0 in ("hausa", "hausa_ng", "ha_ng"):
        return "ha"
    # igbo
    if s0.startswith("ig") or s0 in ("igbo", "igbo_ng", "ig_ng"):
        return "ig"
    # heuristics: first letter
    if s0 and s0[0] in ("p", "y", "h", "i"):
        if s0[0] == "p":
            return "pcm"
        if s0[0] == "y":
            return "yo"
        if s0[0] == "h":
            return "ha"
        if s0[0] == "i":
            return "ig"
    return "en"

# attempt imports in multiple ways. If found, set detect_language(text)->(lang,conf)
detect_language = None

def _try_import_by_name(name: str):
    try:
        m = importlib.import_module(name)
        fn = getattr(m, "detect_language", None)
        if callable(fn):
            logger.info("Imported detect_language via module '%s'", name)
            return fn
    except Exception as e:
        logger.debug("import by name '%s' failed: %s", name, e)
    return None

def _try_import_by_path(path: str):
    try:
        spec = importlib.util.spec_from_file_location("lang_detect_impl", path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = getattr(mod, "detect_language", None)
            if callable(fn):
                logger.info("Imported detect_language from file: %s", path)
                return fn
    except Exception as e:
        logger.debug("import by path '%s' failed: %s", path, e)
    return None

# 1) try package style (conversation_engine.language_detection)
detect_language = _try_import_by_name("conversation_engine.language_detection")

# 2) try file under /app/conversation_engine (if cluster mounts sources there)
if detect_language is None:
    conv_dir = os.path.join(os.getcwd(), "conversation_engine")
    if os.path.isdir(conv_dir) and conv_dir not in sys.path:
        sys.path.insert(0, conv_dir)
        logger.info("Added %s to sys.path", conv_dir)
    detect_language = _try_import_by_name("language_detection")

# 3) search for a file named language_detection*.py under /app or /app/conversation_engine
if detect_language is None:
    search_paths = [os.path.join(os.getcwd(), "conversation_engine"), os.getcwd()]
    for sp in search_paths:
        if not os.path.isdir(sp):
            continue
        matches = glob.glob(os.path.join(sp, "language_detection*.py"))
        if matches:
            # prefer longest/first match
            for mpath in matches:
                fn = _try_import_by_path(mpath)
                if fn:
                    detect_language = fn
                    break
        if detect_language:
            break

if detect_language is None:
    logger.error("language_detection module not available (tried conversation_engine.language_detection, language_detection, and files under /app).")
else:
    logger.info("language_detection backend ready.")

app = FastAPI(title="language-detection")

class DetectRequest(BaseModel):
    text: Optional[str] = None
    user_id: Optional[str] = None

class DetectResponse(BaseModel):
    language: str
    confidence: float

@app.get("/ping")
def ping():
    return {"status": "ok"}

class SetUserLangRequest(BaseModel):
    language: str

@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    if detect_language is None:
        logger.error("language_detection implementation missing — cannot service /detect")
        raise HTTPException(status_code=500, detail="language detection backend missing (check service logs)")

    text = (req.text or "").strip()
    if not text:
        return DetectResponse(language="en", confidence=0.5)

    # 0) If caller provided user_id and we have a stored preference, honor it immediately.
    if getattr(req, "user_id", None):
        try:
            pref = get_user_lang_pref(req.user_id)
            if pref:
                lang_pref = _normalize_lang_code(pref)
                logger.info("[language-detection] honor stored user preference -> %s (user_id=%s)", lang_pref, req.user_id)
                return DetectResponse(language=lang_pref, confidence=0.99)
        except Exception:
            # don't fail on Redis errors — fall back to normal detection
            logger.exception("[language-detection] error reading user lang pref (continuing detection)")

    # Normal detection flow: pass user_id through to the detection function (if it supports it).
    try:
        # if your detect_language implementation accepts user_id, it will use it; if not, it will ignore extra arg.
        try:
            raw_lang, conf = detect_language(text, user_id=req.user_id)
        except TypeError:
            # older detect_language signature without user_id — call the old way to keep backward compatibility
            raw_lang, conf = detect_language(text)

        lang = _normalize_lang_code(raw_lang)
        conf = float(conf or 0.0)
        conf = max(0.0, min(1.0, conf))
        logger.info("[language-detection] detect -> raw=%s normalized=%s conf=%.3f", raw_lang, lang, conf)
        return DetectResponse(language=lang, confidence=conf)
    except Exception as e:
        logger.exception("language detection runtime error: %s", e)
        raise HTTPException(status_code=500, detail="language detection failed (see service logs)")

@app.post("/user/{user_id}/lang")
def set_user_lang(user_id: str, body: SetUserLangRequest):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    # normalize into your canonical short code
    normalized = _normalize_lang_code(body.language)
    # optionally, you can restrict to a set of supported languages:
    SUPPORTED = {"en", "pcm", "yo", "ig", "ha"}
    if normalized not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"unsupported language '{body.language}'")

    try:
        ok = set_user_lang_pref(user_id, normalized)
    except Exception:
        logger.exception("[language-detection] failed to set user lang pref")
        ok = False

    if not ok:
        raise HTTPException(status_code=500, detail="failed to store user language preference")

    # user-friendly confirmation which your bot/UI can echo to user
    return {"ok": True, "message": f"Preference saved: replies will be in {normalized}", "language": normalized}

@app.delete("/user/{user_id}/lang")
def delete_user_lang(user_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    try:
        ok = delete_user_lang_pref(user_id)
    except Exception:
        logger.exception("[language-detection] failed to delete user lang pref")
        ok = False

    if not ok:
        raise HTTPException(status_code=500, detail="failed to delete user language preference")

    return {"ok": True, "message": "Preference deleted", "user_id": user_id}
