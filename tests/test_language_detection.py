#!/usr/bin/env python3
"""
Quick terminal test for language_detection.detect_language()
Run: python scripts/test_language_detection.py
"""
import sys
import os
# adjust this path if your module lives elsewhere
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))  # repo_root/scripts -> repo_root
SRC_PATH = os.path.join(REPO_ROOT, "services", "conversation_engine", "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

try:
    from core import language_detection as ld   # path: services/.../src/core/language_detection.py
except Exception:
    # fallback if module is flat under src
    import language_detection as ld

tests = [
    "I fit pay 5000",
    "I can afford 5000",
    "Mo le san 5000",
    "Zan iya biyan 5000",
    "Enwere m ike ikwu 5000",
    "Abeg, how much for this?",
    "Abi you dey comot?",
    "Na so it be",
    "Good morning, how much for this?"
]

print("Running language detection tests:")
for t in tests:
    lang, conf = ld.detect_language(t)
    print(f"  Input: {t!r}\n    -> lang={lang!r}, conf={conf:.3f}\n")

