# services/conversation_engine/src/app.py
"""
Light wrapper so 'uvicorn app:app' works even when the package is mounted at /app/conversation_engine.
It attempts to import a real ASGI object named `app` from likely modules inside the conversation_engine package.
"""
from __future__ import annotations
import importlib
import pkgutil
import logging
from types import ModuleType
from typing import Optional

logger = logging.getLogger("conversation_engine.app_wrapper")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _find_app_in_module(mod: ModuleType):
    if hasattr(mod, "app"):
        return getattr(mod, "app")
    # sometimes app object is named conversation_engine_app
    if hasattr(mod, "conversation_engine_app"):
        return getattr(mod, "conversation_engine_app")
    return None


def _scan_candidates() -> Optional[object]:
    # Common candidate module names (ordered)
    candidates = [
        "conversation_engine.app",
        "conversation_engine.src.core.app",
        "conversation_engine.core.app",
        "conversation_engine",
    ]
    for cname in candidates:
        try:
            mod = importlib.import_module(cname)
            app_obj = _find_app_in_module(mod)
            if app_obj is not None:
                logger.info("[app_wrapper] found app in module %s", cname)
                return app_obj
        except Exception:
            # ignore, try next
            continue

    # As a last resort, scan submodules of conversation_engine (if package present)
    try:
        pkg = importlib.import_module("conversation_engine")
        if hasattr(pkg, "__path__"):
            for finder, name, ispkg in pkgutil.walk_packages(pkg.__path__, prefix="conversation_engine."):
                try:
                    mod = importlib.import_module(name)
                    app_obj = _find_app_in_module(mod)
                    if app_obj is not None:
                        logger.info("[app_wrapper] found app in module %s", name)
                        return app_obj
                except Exception:
                    continue
    except Exception:
        pass

    return None


app = _scan_candidates()
if app is None:
    raise ImportError("Could not locate ASGI `app` inside the conversation_engine package. "
                      "Ensure the package exposes an `app` object.")
