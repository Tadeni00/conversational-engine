# services/conversation_engine/src/app.py
"""
Adapter shim so `uvicorn app:app` (the wrapper) can find the real ASGI app.
This re-exports `app` from the core implementation (where your real app lives).
It tolerantly tries likely paths so it works whether the implementation is at
conversation_engine.core.app or conversation_engine.src.core.app.
"""
from typing import Optional

_app: Optional[object] = None

# try core.app first (most likely)
try:
    from .core.app import app as _app  # type: ignore
except Exception:
    _app = None

# fallback to src.core.app (older layouts)
if _app is None:
    try:
        from .src.core.app import app as _app  # type: ignore
    except Exception:
        _app = None

# expose `app` symbol expected by uvicorn wrapper
if _app is not None:
    app = _app  # re-export
else:
    # keep failure obvious in logs if we ever import this module
    raise ImportError(
        "Could not re-export ASGI `app` from conversation_engine; "
        "check conversation_engine.core.app or conversation_engine.src.core.app"
    )
