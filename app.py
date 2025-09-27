# services/conversation_engine/src/core/app.py
import os
import sys
import glob
import importlib
import importlib.util
import logging
from typing import Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("language-detection")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# attempt imports in multiple ways. If found, set detect_language(text)->(lang,conf)
detect_language = None
_import_error = None

def _try_import_by_name(name: str):
    try:
        m = importlib.import_module(name)
        fn = getattr(m, "detect_language", None)
        if callable(fn):
            logger.info("Imported detect_language via module '%s'", name)
            return fn
    except Exception as e:
        # log debug
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

# final fallback: log the last import error if any
if detect_language is None:
    logger.error("language_detection module not available (tried conversation_engine.language_detection, language_detection, and files under /app).")
else:
    logger.info("language_detection backend ready.")

app = FastAPI(title="language-detection")

class DetectRequest(BaseModel):
    text: Optional[str] = None

class DetectResponse(BaseModel):
    language: str
    confidence: float

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    if detect_language is None:
        logger.error("language_detection implementation missing — cannot service /detect")
        # Helpful error for debugging; logs contain import attempts
        raise HTTPException(status_code=500, detail="language detection backend missing (check service logs)")

    text = (req.text or "").strip()
    if not text:
        return DetectResponse(language="en", confidence=0.5)

    try:
        lang, conf = detect_language(text)
        lang = str(lang or "en")
        conf = float(conf or 0.0)
        conf = max(0.0, min(1.0, conf))
        return DetectResponse(language=lang, confidence=conf)
    except Exception as e:
        logger.exception("language detection runtime error: %s", e)
        raise HTTPException(status_code=500, detail="language detection failed (see service logs)")
