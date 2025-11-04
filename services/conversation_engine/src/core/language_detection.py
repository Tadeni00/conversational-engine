# """
# Robust language detection helper.
# See docstring at top for behavior & environment.
# """
# from __future__ import annotations
# from typing import Tuple, Optional, Dict
# import os, re, threading, logging
# import unicodedata
# from .language_pref import get_user_lang_pref

# logger = logging.getLogger("language_detection")
# if not logger.handlers:
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# # -------------------------------------------------------------------
# # Config & environment
# # -------------------------------------------------------------------
# _FASTTEXT_PATH = os.getenv("FASTTEXT_MODEL_PATH", "").strip()
# _FASTTEXT_SERVICE_URL = os.getenv("FASTTEXT_SERVICE_URL", "").strip()
# _FASTTEXT_SERVICE_TIMEOUT = float(os.getenv("FASTTEXT_SERVICE_TIMEOUT", "2.0"))

# # Override thresholds (tunable via env)
# _CONF_OVERRIDE_THRESHOLD = float(os.getenv("LANG_DETECT_CONF_THRESHOLD", "0.90"))
# _EVIDENCE_THRESHOLD = float(os.getenv("LANG_DETECT_EVIDENCE_THRESHOLD", "0.45"))
# _STRONG_WEIGHT = float(os.getenv("LANG_DETECT_STRONG_WEIGHT", "1.5"))
# _WEAK_WEIGHT = float(os.getenv("LANG_DETECT_WEAK_WEIGHT", "0.6"))
# _STRONG_AUTO_OVERRIDE_THRESHOLD = float(os.getenv("LANG_DETECT_STRONG_AUTO_THR", "2.0"))
# _OVERRIDE_CONFIDENCE = float(os.getenv("LANG_DETECT_OVERRIDE_CONF", "0.85"))

# # NEW: low-confidence trust threshold for non-English fastText predictions
# _LOW_CONF_TRUST = float(os.getenv("LANG_DETECT_LOW_CONF_TRUST", "0.60"))

# # -------------------------------------------------------------------
# # Marker dictionaries (expanded Pidgin markers; conservative additions for local langs)
# # -------------------------------------------------------------------
# # Note: keep these conservative. You can add more words/phrases via config if needed.
# STRONG_MARKERS_PCM = {"abi", "una", "wahala", "sef", "abeg", "jare", "omo", "no be", "how much for", "oya", "sharpy"}
# WEAK_MARKERS_PCM   = {"fit", "dey", "go", "chop", "make", "come", "waka", "na", "biko", "i fit", "e go", "you go"}

# STRONG_MARKERS_YO  = {"ẹkọ", "ọ̀sán", "ilé", "báwo", "àwọn", "rà", "rá", "jẹ"}
# WEAK_MARKERS_YO    = {"lọ", "jẹ", "ni", "sí", "ra"}

# STRONG_MARKERS_IG  = {"nwoke", "nne", "nwanne", "ihe", "oba", "daalụ", "ị"}
# WEAK_MARKERS_IG    = {"ga", "na", "so", "ka"}

# STRONG_MARKERS_HA  = {"ina", "kai", "yau", "don", "naira"}
# WEAK_MARKERS_HA    = {"ne", "da", "na"}

# LANG_MARKERS = {
#     "pcm": (STRONG_MARKERS_PCM, WEAK_MARKERS_PCM),
#     "yo":  (STRONG_MARKERS_YO,  WEAK_MARKERS_YO),
#     "ig":  (STRONG_MARKERS_IG,  WEAK_MARKERS_IG),
#     "ha":  (STRONG_MARKERS_HA,  WEAK_MARKERS_HA),
# }

# # improved token regex: allow apostrophes and accented characters
# _TOKEN_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


# def _strip_diacritics(s: str) -> str:
#     """Normalize to NFKD and remove combining marks for robust matching."""
#     if not s:
#         return s
#     nk = unicodedata.normalize("NFKD", s)
#     return "".join(ch for ch in nk if not unicodedata.combining(ch))


# def _tokenize(text: str):
#     # Lowercase + diacritics-stripped tokens for matching
#     if not text:
#         return []
#     s = text.lower()
#     s = _strip_diacritics(s)
#     return [t for t in _TOKEN_RE.findall(s)]


# # -------------------------------------------------------------------
# # Local FastText loader
# # -------------------------------------------------------------------
# _fasttext_model = None
# _fasttext_lock = threading.Lock()
# _use_local_fasttext = False


# def _try_load_local_fasttext() -> None:
#     global _fasttext_model, _use_local_fasttext
#     if _fasttext_model is not None or not _FASTTEXT_PATH:
#         return
#     with _fasttext_lock:
#         if _fasttext_model is not None:
#             return
#         try:
#             import fasttext  # type: ignore
#             if os.path.exists(_FASTTEXT_PATH):
#                 _fasttext_model = fasttext.load_model(_FASTTEXT_PATH)
#                 _use_local_fasttext = True
#                 logger.info(f"[language_detection] fastText model loaded from: {_FASTTEXT_PATH}")
#             else:
#                 logger.warning(
#                     f"[language_detection] FASTTEXT_MODEL_PATH set but file not found: {_FASTTEXT_PATH}"
#                 )
#         except Exception as e:
#             _use_local_fasttext = False
#             _fasttext_model = None
#             logger.exception(f"[language_detection] fastText failed to load: {e!r}")


# # -------------------------------------------------------------------
# # Remote FastText client
# # -------------------------------------------------------------------
# def _call_fasttext_service(text: str) -> Optional[Tuple[str, float]]:
#     try:
#         import requests  # type: ignore
#         resp = requests.post(
#             _FASTTEXT_SERVICE_URL, json={"text": text}, timeout=_FASTTEXT_SERVICE_TIMEOUT
#         )
#         if resp.status_code != 200:
#             logger.warning(f"[language_detection] service error {resp.status_code}: {resp.text}")
#             return None
#         j = resp.json()
#         lang = j.get("lang") or j.get("language") or "en"
#         score = float(j.get("score", j.get("confidence", 0.0)))
#         return lang, score
#     except Exception as e:
#         logger.exception(f"[language_detection] service call failed: {e!r}")
#         return None


# # -------------------------------------------------------------------
# # Decision logic
# # -------------------------------------------------------------------
# def _normalize_lang_code(l: str) -> str:
#     l = (l or "").strip().lower()
#     if l in ("eng", "english"):
#         return "en"
#     if l in ("pidgin", "pcm_ng", "pcm-nigeria", "pcm"):
#         return "pcm"
#     # fastText sometimes returns "__label__pcm" or similar
#     if l.startswith("__label__"):
#         return _normalize_lang_code(l.replace("__label__", ""))
#     # unknown noisy labels like "<UNABLE_PCM>" -> treat as unknown
#     if l.startswith("<") and l.endswith(">"):
#         return "unknown"
#     return l


# def decide_language_with_override(
#     text: str,
#     fasttext_lang: str,
#     fasttext_conf: float,
#     *,

#     conf_override_threshold: float = _CONF_OVERRIDE_THRESHOLD,
#     evidence_threshold: float = _EVIDENCE_THRESHOLD,
#     strong_weight: float = _STRONG_WEIGHT,
#     weak_weight: float = _WEAK_WEIGHT,
#     strong_auto_override_threshold: float = _STRONG_AUTO_OVERRIDE_THRESHOLD,
# ) -> Dict:
#     """
#     Compute weighted evidence for local languages and decide whether to override.

#     Returns a dict with keys:
#       - final_language
#       - override (bool)
#       - reason (str)
#       - meta (dict)
#     """
#     t = (text or "").strip()
#     token_list = _tokenize(t)
#     token_set = set(token_list)
#     low_text = _strip_diacritics(t.lower())

#     # compute evidence scores for each candidate local language
#     scores: Dict[str, float] = {}
#     per_lang_counts: Dict[str, Dict[str, int]] = {}
#     for lang, (strong_markers, weak_markers) in LANG_MARKERS.items():
#         # strong counts: allow substring matches for multi-word markers
#         strong_count = 0
#         for m in strong_markers:
#             m_norm = _strip_diacritics(m.lower())

#             # SKIP markers that normalize to a single character (too ambiguous,
#             # e.g. 'ị' -> 'i' would match English pronoun 'I'). This prevents
#             # false positives on short single-letter markers.
#             if len(m_norm) <= 1:
#                 continue

#             if " " in m_norm:
#                 if m_norm in low_text:
#                     strong_count += 1
#             else:
#                 if m_norm in token_set:
#                     strong_count += 1
#         weak_count = 0
#         for m in weak_markers:
#             m_norm = _strip_diacritics(m.lower())

#             # same skip for weak markers
#             if len(m_norm) <= 1:
#                 continue

#             if " " in m_norm:
#                 if m_norm in low_text:
#                     weak_count += 1
#             else:
#                 if m_norm in token_set:
#                     weak_count += 1
#         per_lang_counts[lang] = {"strong": strong_count, "weak": weak_count}
#         scores[lang] = strong_count * strong_weight + weak_count * weak_weight

#     # default decision = fastText result
#     final_lang = fasttext_lang
#     override = False
#     reason = "no_override"

#     # best local candidate
#     best_lang, best_score = max(scores.items(), key=lambda x: x[1])

#     # Auto override for very strong evidence regardless of fastText label/confidence
#     if best_score >= strong_auto_override_threshold:
#         final_lang = best_lang
#         override = True
#         reason = "strong_evidence_auto_override"
#     else:
#         # If fastText predicted English or unknown, use the conservative override logic
#         if fasttext_lang in ("en", "eng", "unknown"):
#             # require fastText conf below threshold and sufficient evidence
#             if fasttext_conf < conf_override_threshold and best_score >= evidence_threshold:
#                 final_lang = best_lang
#                 override = True
#                 reason = f"conf_below_{conf_override_threshold}_and_evidence"
#         else:
#             # fastText predicted some non-English label. If its confidence is low,
#             # allow evidence-based override to a local language.
#             if fasttext_conf < _LOW_CONF_TRUST and best_score >= evidence_threshold:
#                 final_lang = best_lang
#                 override = True
#                 reason = f"low_conf_non_en_and_evidence"

#     if override:
#         logger.info(
#             "[language_detection] override: text=%r fasttext=(%s,%.3f) -> %s (%s) evidence=%s counts=%s",
#             t,
#             fasttext_lang,
#             fasttext_conf,
#             final_lang,
#             reason,
#             scores,
#             per_lang_counts,
#         )

#     return {
#         "final_language": final_lang,
#         "override": override,
#         "reason": reason,
#         "meta": {
#             "tokens": len(token_list),
#             "evidence_scores": scores,
#             "per_lang_counts": per_lang_counts,
#             "fasttext_lang": fasttext_lang,
#             "fasttext_conf": fasttext_conf,
#             "best_local_candidate": best_lang,
#             "best_local_score": best_score,
#         },
#     }


# # -------------------------------------------------------------------
# # Public entrypoint
# # -------------------------------------------------------------------
# def detect_language(text: str, user_id: Optional[str] = None) -> Tuple[str, float]:
#     # 1) Check explicit per-user preference stored in Redis
#     if user_id:
#         try:
#             pref = get_user_lang_pref(user_id)
#             if pref:
#                 # Respect user preference strongly (high confidence)
#                 return pref, 0.99
#         except Exception:
#             # If Redis fails, continue with normal detection flow
#             pass

#     txt = (text or "").strip()
#     if not txt:
#         return "en", 0.5

#     # helper to normalize labels
#     def _norm(l: str) -> str:
#         return _normalize_lang_code(l)

#     # 1) Remote service
#     if _FASTTEXT_SERVICE_URL:
#         out = _call_fasttext_service(txt)
#         if out:
#             raw_lang, raw_conf = out
#             try:
#                 lang = _norm(raw_lang)
#                 conf = float(raw_conf or 0.0)
#             except Exception:
#                 lang = _norm(raw_lang)
#                 conf = float(raw_conf or 0.0)

#             # If fastText predicts a non-English language with sufficient confidence, trust it
#             if lang not in ("en", "eng", "unknown") and conf >= _LOW_CONF_TRUST:
#                 return lang, conf

#             # Otherwise (fastText says 'en' OR low-confidence non-en or unknown) decide whether to override
#             decision = decide_language_with_override(txt, lang, conf)
#             if decision["override"]:
#                 return decision["final_language"], max(conf, _OVERRIDE_CONFIDENCE)
#             return lang, conf

#     # 2) Local fastText
#     if _FASTTEXT_PATH:
#         _try_load_local_fasttext()
#         if _use_local_fasttext and _fasttext_model is not None:
#             try:
#                 labels, probs = _fasttext_model.predict(txt.replace("\n", " "), k=1)
#                 if labels and probs:
#                     raw_label = labels[0]
#                     prob = float(probs[0])
#                     lang = _norm(raw_label.replace("__label__", ""))

#                     # If fastText predicts non-English with enough confidence, trust it
#                     if lang not in ("en", "eng", "unknown") and prob >= _LOW_CONF_TRUST:
#                         return lang, prob

#                     # Otherwise (fastText says 'en' or non-en but low-conf or unknown) run override
#                     decision = decide_language_with_override(txt, lang, prob)
#                     if decision["override"]:
#                         return decision["final_language"], max(prob, _OVERRIDE_CONFIDENCE)
#                     return lang, prob
#             except Exception as e:
#                 logger.exception(f"[language_detection] local predict failed: {e!r}")

#     # 3) Default fallback
#     logger.warning("[language_detection] fallback default -> en,0.6 for text=%r", txt)
#     return "en", 0.6


# # eager load if path present
# if _FASTTEXT_PATH:
#     try:
#         _try_load_local_fasttext()
#     except Exception:
#         # loader logs its own exception; don't crash import
#         pass


"""
Robust language detection helper using N-ATLaS (Hugging Face Inference) instead of fastText by default.

Compatibility notes:
 - Reads HF token either as a raw token or as a path to a file (useful for your ./secrets/hf_token).
 - Accepts a full inference URL via N-ATLaS_URL (your .env contains that).
 - Falls back to HF_LANG_DETECT_MODEL model id if no full URL provided.
 - Uses LANG_DETECT_TIMEOUT (or HF_LANG_DETECT_TIMEOUT) from env for request timeout.
 - Keeps the original markers/override logic and the public detect_language() API.
"""
from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import os
import re
import threading
import logging
import unicodedata
import json

from .language_pref import get_user_lang_pref

logger = logging.getLogger("language_detection")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------------------------------------------------
# Config & environment (support file-path secrets)
# -------------------------------------------------------------------
# Model id fallback and explicit full-url env name (user provided N-ATLaS_URL in .env)
_HF_MODEL = os.getenv("HF_LANG_DETECT_MODEL", "NCAIR1/N-ATLaS").strip()
_HF_URL_ENV = os.getenv("N-ATLaS_URL", "").strip() or os.getenv("HF_LANG_DETECT_URL", "").strip()

# Token: either token string or path to file (we support both)
_HF_TOKEN_RAW = os.getenv("HF_TOKEN", "").strip()
# Timeout: prefer LANG_DETECT_TIMEOUT (your .env) then HF_LANG_DETECT_TIMEOUT then default
_HF_SERVICE_TIMEOUT = float(os.getenv("LANG_DETECT_TIMEOUT", os.getenv("HF_LANG_DETECT_TIMEOUT", "3.0")))

# Backwards-compatible env keys for thresholds (same as before)
_CONF_OVERRIDE_THRESHOLD = float(os.getenv("LANG_DETECT_CONF_THRESHOLD", "0.90"))
_EVIDENCE_THRESHOLD = float(os.getenv("LANG_DETECT_EVIDENCE_THRESHOLD", "0.45"))
_STRONG_WEIGHT = float(os.getenv("LANG_DETECT_STRONG_WEIGHT", "1.5"))
_WEAK_WEIGHT = float(os.getenv("LANG_DETECT_WEAK_WEIGHT", "0.6"))
_STRONG_AUTO_OVERRIDE_THRESHOLD = float(os.getenv("LANG_DETECT_STRONG_AUTO_THR", "2.0"))
_OVERRIDE_CONFIDENCE = float(os.getenv("LANG_DETECT_OVERRIDE_CONF", "0.85"))
_LOW_CONF_TRUST = float(os.getenv("LANG_DETECT_LOW_CONF_TRUST", "0.60"))

# -------------------------------------------------------------------
# Helper to load token if a file path was provided
# -------------------------------------------------------------------
def _read_token(maybe_path: str) -> str:
    if not maybe_path:
        return ""
    maybe_path = maybe_path.strip()
    # if user provided a path to a file that exists, read it
    try:
        if os.path.exists(maybe_path) and os.path.isfile(maybe_path):
            with open(maybe_path, "r", encoding="utf-8") as fh:
                return fh.read().strip()
    except Exception:
        # if reading fails, fallback to returning original string
        logger.exception("[language_detection] failed to read HF token from path=%s", maybe_path)
    return maybe_path

_HF_TOKEN = _read_token(_HF_TOKEN_RAW)

# Build the final HF inference URL: prefer explicit full URL env, else model id
if _HF_URL_ENV:
    _HF_INFERENCE_URL = _HF_URL_ENV
else:
    _HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{_HF_MODEL}"

logger.debug("[language_detection] HF inference url=%s timeout=%s token_present=%s",
             _HF_INFERENCE_URL, _HF_SERVICE_TIMEOUT, bool(_HF_TOKEN))

# -------------------------------------------------------------------
# Marker dictionaries (conservative)
# -------------------------------------------------------------------
STRONG_MARKERS_PCM = {"abi", "una", "wahala", "sef", "abeg", "jare", "omo", "no be", "how much for", "oya", "sharpy"}
WEAK_MARKERS_PCM   = {"fit", "dey", "go", "chop", "make", "come", "waka", "na", "biko", "i fit", "e go", "you go"}

STRONG_MARKERS_YO  = {"ẹkọ", "ọ̀sán", "ilé", "báwo", "àwọn", "rà", "rá", "jẹ"}
WEAK_MARKERS_YO    = {"lọ", "jẹ", "ni", "sí", "ra"}

STRONG_MARKERS_IG  = {"nwoke", "nne", "nwanne", "ihe", "oba", "daalụ", "ị"}
WEAK_MARKERS_IG    = {"ga", "na", "so", "ka"}

STRONG_MARKERS_HA  = {"ina", "kai", "yau", "don", "naira"}
WEAK_MARKERS_HA    = {"ne", "da", "na"}

LANG_MARKERS = {
    "pcm": (STRONG_MARKERS_PCM, WEAK_MARKERS_PCM),
    "yo":  (STRONG_MARKERS_YO,  WEAK_MARKERS_YO),
    "ig":  (STRONG_MARKERS_IG,  WEAK_MARKERS_IG),
    "ha":  (STRONG_MARKERS_HA,  WEAK_MARKERS_HA),
}

# improved token regex: allow apostrophes and accented characters
_TOKEN_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


def _strip_diacritics(s: str) -> str:
    """Normalize to NFKD and remove combining marks for robust matching."""
    if not s:
        return ""
    nk = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nk if not unicodedata.combining(ch))


def _tokenize(text: str):
    # Lowercase + diacritics-stripped tokens for matching
    if not text:
        return []
    s = text.lower()
    s = _strip_diacritics(s)
    return [t for t in _TOKEN_RE.findall(s)]

# ------------------ HF / N-ATLaS caller ------------------
def _extract_text_from_hf_response(j: Any) -> Optional[str]:
    """
    Minimal extraction: handle common HF response shapes.
    """
    try:
        if isinstance(j, dict):
            if "generated_text" in j and isinstance(j["generated_text"], str):
                return j["generated_text"].strip()
            if "outputs" in j and isinstance(j["outputs"], list) and j["outputs"]:
                first = j["outputs"][0]
                if isinstance(first, dict) and "generated_text" in first:
                    return first["generated_text"].strip()
            if "choices" in j and j["choices"]:
                c0 = j["choices"][0]
                if isinstance(c0, dict):
                    if "text" in c0 and isinstance(c0["text"], str):
                        return c0["text"].strip()
                    if "message" in c0 and isinstance(c0["message"], dict):
                        msg = c0["message"]
                        if "content" in msg and isinstance(msg["content"], str):
                            return msg["content"].strip()
        if isinstance(j, list) and j:
            first = j[0]
            if isinstance(first, dict):
                for k in ("generated_text", "text", "content"):
                    if k in first and isinstance(first[k], str):
                        return first[k].strip()
            if isinstance(first, str):
                return first.strip()
    except Exception:
        pass
    return None

def _call_natlas_service(text: str) -> Optional[Tuple[str, float]]:
    """
    Call HF inference API with N-ATLaS (or model specified by _HF_INFERENCE_URL).
    The prompt requests JSON like {"language":"pcm","confidence":0.9}.
    Returns (lang_code, confidence) or None on failure.
    """
    try:
        import requests
    except Exception:
        logger.exception("[language_detection] requests not available for HF inference")
        return None

    if not _HF_INFERENCE_URL:
        logger.warning("[language_detection] no HF inference URL configured")
        return None

    headers = {"Content-Type": "application/json"}
    if _HF_TOKEN:
        headers["Authorization"] = f"Bearer {_HF_TOKEN}"
    else:
        logger.warning("[language_detection] HF_TOKEN empty; unauthenticated calls may be rejected or rate-limited")

    # few-shot prompt asking for JSON-only output
    prompt = (
        "Detect the primary language of the message and OUTPUT ONLY JSON.\n"
        "Allowed codes: en, pcm, yo, ig, ha, unknown.\n"
        "Return JSON: {\"language\":\"<code>\",\"confidence\":0.00}\n\n"
        "EXAMPLES:\n"
        "Input: 'How much is this?'  -> {\"language\":\"en\",\"confidence\":0.95}\n"
        "Input: 'I fit pay small'   -> {\"language\":\"pcm\",\"confidence\":0.92}\n"
        "Input: 'Ṣe o le ran mi lọ́wọ́?' -> {\"language\":\"yo\",\"confidence\":0.86}\n\n"
        "Now analyze the following message and respond with JSON only.\n\n"
        "### MESSAGE:\n"
        f"{text}\n\nJSON:"
    )

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 40, "temperature": 0.0, "top_p": 1.0},
    }

    try:
        resp = requests.post(_HF_INFERENCE_URL, headers=headers, json=payload, timeout=_HF_SERVICE_TIMEOUT)
    except Exception as e:
        logger.exception("[language_detection] HF inference request failed: %s", e)
        return None

    if resp is None:
        return None

    if resp.status_code != 200:
        try:
            body = resp.text[:800]
        except Exception:
            body = "<no-body>"
        logger.warning("[language_detection] HF inference returned %s: %s", resp.status_code, body)
        return None

    # try parse json
    try:
        j = resp.json()
    except Exception:
        j = None

    text_out = _extract_text_from_hf_response(j) if j is not None else (resp.text or "")
    text_out = (text_out or "").strip()
    if not text_out:
        return None

    # extract JSON blob from model response
    json_obj = None
    try:
        m = re.search(r"\{.*\}", text_out, flags=re.DOTALL)
        if m:
            candidate = m.group(0).strip()
            json_obj = json.loads(candidate)
    except Exception:
        json_obj = None

    if json_obj:
        raw_lang = str(json_obj.get("language", "unknown") or "unknown")
        try:
            conf_val = float(json_obj.get("confidence", 0.0) or 0.0)
            conf_val = max(0.0, min(1.0, conf_val))
        except Exception:
            conf_val = 0.0
        return raw_lang, conf_val

    # fallback: attempt to glean language token and numeric from text_out
    lang_candidates = {
        "en": ["english", "en"],
        "pcm": ["pidgin", "pcm", "pidgin english", "nigerian pidgin"],
        "yo": ["yoruba", "yo"],
        "ig": ["igbo", "ig"],
        "ha": ["hausa", "ha"],
    }
    found_lang = None
    lower = text_out.lower()
    for code, terms in lang_candidates.items():
        for t in terms:
            if re.search(r"\b" + re.escape(t) + r"\b", lower):
                found_lang = code
                break
        if found_lang:
            break

    conf_match = re.search(r"([01](?:\.\d+))|(?:\b0?\.\d+\b)|(?:\b1\.0\b)|(?:\b1\b)", lower)
    conf_val = None
    if conf_match:
        try:
            conf_val = float(conf_match.group(0))
            conf_val = max(0.0, min(1.0, conf_val))
        except Exception:
            conf_val = None

    if found_lang:
        return found_lang, (conf_val if conf_val is not None else 0.6)

    return "unknown", (conf_val if conf_val is not None else 0.0)


# -------------------------------------------------------------------
# Decision logic (unchanged)
# -------------------------------------------------------------------
def _normalize_lang_code(l: str) -> str:
    l = (l or "").strip().lower()
    if l in ("eng", "english"):
        return "en"
    if l in ("pidgin", "pcm_ng", "pcm-nigeria", "pcm"):
        return "pcm"
    if l.startswith("__label__"):
        return _normalize_lang_code(l.replace("__label__", ""))
    if l.startswith("<") and l.endswith(">"):
        return "unknown"
    if l in ("yo", "yoruba"):
        return "yo"
    if l in ("ig", "igbo"):
        return "ig"
    if l in ("ha", "hausa"):
        return "ha"
    return l


def decide_language_with_override(
    text: str,
    model_lang: str,
    model_conf: float,
    *,
    conf_override_threshold: float = _CONF_OVERRIDE_THRESHOLD,
    evidence_threshold: float = _EVIDENCE_THRESHOLD,
    strong_weight: float = _STRONG_WEIGHT,
    weak_weight: float = _WEAK_WEIGHT,
    strong_auto_override_threshold: float = _STRONG_AUTO_OVERRIDE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Compute weighted evidence for local languages and decide whether to override.
    """
    t = (text or "").strip()
    token_list = _tokenize(t)
    token_set = set(token_list)
    low_text = _strip_diacritics(t.lower())

    scores: Dict[str, float] = {}
    per_lang_counts: Dict[str, Dict[str, int]] = {}
    for lang, (strong_markers, weak_markers) in LANG_MARKERS.items():
        strong_count = 0
        for m in strong_markers:
            m_norm = _strip_diacritics(m.lower())
            if len(m_norm) <= 1:
                continue
            if " " in m_norm:
                if m_norm in low_text:
                    strong_count += 1
            else:
                if m_norm in token_set:
                    strong_count += 1
        weak_count = 0
        for m in weak_markers:
            m_norm = _strip_diacritics(m.lower())
            if len(m_norm) <= 1:
                continue
            if " " in m_norm:
                if m_norm in low_text:
                    weak_count += 1
            else:
                if m_norm in token_set:
                    weak_count += 1
        per_lang_counts[lang] = {"strong": strong_count, "weak": weak_count}
        scores[lang] = strong_count * strong_weight + weak_count * weak_weight

    final_lang = model_lang
    override = False
    reason = "no_override"

    best_lang, best_score = max(scores.items(), key=lambda x: x[1])

    if best_score >= strong_auto_override_threshold:
        final_lang = best_lang
        override = True
        reason = "strong_evidence_auto_override"
    else:
        if model_lang in ("en", "eng", "unknown"):
            if model_conf < conf_override_threshold and best_score >= evidence_threshold:
                final_lang = best_lang
                override = True
                reason = f"conf_below_{conf_override_threshold}_and_evidence"
        else:
            if model_conf < _LOW_CONF_TRUST and best_score >= evidence_threshold:
                final_lang = best_lang
                override = True
                reason = f"low_conf_non_en_and_evidence"

    if override:
        logger.info(
            "[language_detection] override: text=%r model=(%s,%.3f) -> %s (%s) evidence=%s counts=%s",
            t,
            model_lang,
            model_conf,
            final_lang,
            reason,
            scores,
            per_lang_counts,
        )

    return {
        "final_language": final_lang,
        "override": override,
        "reason": reason,
        "meta": {
            "tokens": len(token_list),
            "evidence_scores": scores,
            "per_lang_counts": per_lang_counts,
            "model_lang": model_lang,
            "model_conf": model_conf,
            "best_local_candidate": best_lang,
            "best_local_score": best_score,
        },
    }


# -------------------------------------------------------------------
# Public entrypoint (drop-in replacement)
# -------------------------------------------------------------------
def detect_language(text: str, user_id: Optional[str] = None) -> Tuple[str, float]:
    # 1) Respect per-user preference if present
    if user_id:
        try:
            pref = get_user_lang_pref(user_id)
            if pref:
                return pref, 0.99
        except Exception:
            pass

    txt = (text or "").strip()
    if not txt:
        return "en", 0.5

    def _norm(l: str) -> str:
        return _normalize_lang_code(l)

    # 1) Try N-ATLaS / HF inference
    model_call_res = None
    try:
        model_call_res = _call_natlas_service(txt)
    except Exception:
        logger.exception("[language_detection] N-ATLaS call failed unexpectedly")

    if model_call_res:
        raw_lang, raw_conf = model_call_res
        try:
            lang = _norm(raw_lang)
            conf = float(raw_conf or 0.0)
        except Exception:
            lang = _norm(raw_lang)
            conf = float(raw_conf or 0.0)

        # trust non-en high-confidence predictions
        if lang not in ("en", "eng", "unknown") and conf >= _LOW_CONF_TRUST:
            return lang, conf

        # otherwise apply override logic
        decision = decide_language_with_override(txt, lang, conf)
        if decision["override"]:
            return decision["final_language"], max(conf, _OVERRIDE_CONFIDENCE)
        return lang, conf

    # 2) If HF failed, fallback to marker-only decision (conservative)
    decision = decide_language_with_override(txt, "unknown", 0.0)
    if decision["override"]:
        return decision["final_language"], max(0.5, _OVERRIDE_CONFIDENCE)

    logger.warning("[language_detection] fallback default -> en,0.6 for text=%r", txt)
    return "en", 0.6
