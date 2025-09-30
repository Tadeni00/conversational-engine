# Adapter shim so the wrapper /uvicorn entrypoint can import `detect_language`
# from one of these names: conversation_engine.language_detection or language_detection
from .core.language_detection import detect_language  # type: ignore
__all__ = ("detect_language",)
