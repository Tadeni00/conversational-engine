#!/usr/bin/env python3
from __future__ import annotations
import sys
import os
import time

# ensure repo root is on sys.path so imports like services.* work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# import the helper
from services.conversation_engine.src.core.language_pref import (
    set_user_lang_pref,
    get_user_lang_pref,
    get_user_lang_pref_ttl,
)

def run():
    uid = "alice-test-1"
    print("setting pref -> pcm")
    ok = set_user_lang_pref(uid, "pcm", ttl_seconds=10)
    assert ok, "Failed to set preference in Redis"

    v = get_user_lang_pref(uid)
    print("stored:", v)
    assert v == "pcm", "Value read from Redis does not match set value"

    ttl = get_user_lang_pref_ttl(uid)
    print("TTL immediately after set:", ttl)
    assert ttl is not None and ttl <= 10, "TTL not set correctly"

    print("sleeping 12s to verify TTL expiry...")
    time.sleep(12)
    after = get_user_lang_pref(uid)
    print("after sleep:", after)
    assert after is None, "Preference should have expired"

if __name__ == "__main__":
    print("Running lang_pref test (repo root added to sys.path):", REPO_ROOT)
    run()
