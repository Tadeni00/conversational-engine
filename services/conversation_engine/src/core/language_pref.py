# services/conversation_engine/src/core/language_pref.py
from __future__ import annotations
import os
import logging
from typing import Optional

logger = logging.getLogger("language_pref")
try:
    import redis
except Exception:
    redis = None

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# TTL in seconds (default 1 day now, adjust via env if needed)
DEFAULT_TTL = int(os.getenv("USER_LANG_PREF_TTL", str(1 * 24 * 3600)))

_redis_client = None

def _get_redis() -> Optional["redis.Redis"]:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if redis is None:
        logger.warning("[lang_pref] redis library not available")
        return None
    try:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        # quick ping to test connectivity
        _redis_client.ping()
        return _redis_client
    except Exception as e:
        logger.exception("[lang_pref] failed to connect to redis: %s", e)
        _redis_client = None
        return None

def set_user_lang_pref(user_id: str, lang: str, ttl_seconds: int = DEFAULT_TTL) -> bool:
    """
    Set user language preference. Returns True on success, False otherwise.
    """
    if not user_id or not lang:
        return False
    r = _get_redis()
    if r is None:
        logger.warning("[lang_pref] redis not available; cannot save pref for %s", user_id)
        return False
    try:
        key = f"user:lang:{user_id}"
        r.set(key, lang, ex=ttl_seconds)
        logger.info("[lang_pref] set %s -> %s (ttl=%s)", user_id, lang, ttl_seconds)
        return True
    except Exception as e:
        logger.exception("[lang_pref] failed to set pref for %s: %s", user_id, e)
        return False

def get_user_lang_pref(user_id: str) -> Optional[str]:
    """
    Return stored language code or None if none set / on error.
    """
    if not user_id:
        return None
    r = _get_redis()
    if r is None:
        return None
    try:
        key = f"user:lang:{user_id}"
        v = r.get(key)
        if v:
            return v
        return None
    except Exception as e:
        logger.exception("[lang_pref] failed to get pref for %s: %s", user_id, e)
        return None

def delete_user_lang_pref(user_id: str) -> bool:
    """
    Delete stored user language preference. Returns True on success (including if key didn't exist).
    False only when Redis not available or an error occurs.
    """
    if not user_id:
        return False
    r = _get_redis()
    if r is None:
        logger.warning("[lang_pref] redis not available; cannot delete pref for %s", user_id)
        return False
    try:
        key = f"user:lang:{user_id}"
        r.delete(key)
        logger.info("[lang_pref] deleted pref for %s", user_id)
        return True
    except Exception as e:
        logger.exception("[lang_pref] failed to delete pref for %s: %s", user_id, e)
        return False
