# services/conversation_engine/src/app.py
"""
Robust adapter so `uvicorn app:app` (and the wrapper) can find the real ASGI app.
We try a number of absolute module names (both core.app and src.core.app) and
re-export the found `app` object.
"""
import importlib
import logging
from typing import Optional

logger = logging.getLogger("conversation_engine.app_adapter")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

_app: Optional[object] = None

candidates = [
    "conversation_engine.core.app",
    "conversation_engine.src.core.app",
    "conversation_engine.app",
    "conversation_engine.src.core",   # try module exposing app at package-level
    "conversation_engine.core",       # same as above
]

for cname in candidates:
    try:
        mod = importlib.import_module(cname)
        if hasattr(mod, "app"):
            _app = getattr(mod, "app")
            logger.info("Re-exporting ASGI app from module %s", cname)
            break
    except Exception as e:
        logger.debug("Failed to import %s: %s", cname, e)

if _app is None:
    raise ImportError(
        "Could not re-export ASGI `app` from conversation_engine; "
        "tried: %s" % (", ".join(candidates),)
    )

# publish symbol uvicorn expects
app = _app  # type: ignore
