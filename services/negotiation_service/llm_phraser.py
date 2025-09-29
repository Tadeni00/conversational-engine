# llm_phraser.py — improved from your last-working copy (patched for template-only guard)
from __future__ import annotations
import os
import re
import threading
import logging
import random
import math
from typing import Dict, Any, Optional, Tuple, List
import requests

logger = logging.getLogger("llm_phraser")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ------------------ Config (env) ------------------
LLM_MODE = os.getenv("LLM_MODE", "REMOTE").upper()            # REMOTE | LOCAL | TEMPLATE
LLM_REMOTE_PROVIDER = os.getenv("LLM_REMOTE_PROVIDER", "GROQ").upper()  # GROQ | HF | OPENAI
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-7b-instruct")
LLM_REMOTE_URL = os.getenv("LLM_REMOTE_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "80"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))

# new: make remote timeout configurable
LLM_REMOTE_TIMEOUT = int(os.getenv("LLM_REMOTE_TIMEOUT", "20"))

# negotiation default min ratio (used when product.min_price missing)
DEFAULT_MIN_PRICE_RATIO = float(os.getenv("DEFAULT_MIN_PRICE_RATIO", "0.5"))

# --- NEW FLAGS (opt-in) ---
# require this many context marker matches before auto-overriding to local templates (safer default)
LLM_CONTEXT_MATCHES = int(os.getenv("LLM_CONTEXT_MATCHES", "2"))
# allow remote/local outputs even if numeric price check fails (useful for testing)
LLM_ALLOW_REMOTE_WITHOUT_PRICE = os.getenv("LLM_ALLOW_REMOTE_WITHOUT_PRICE", "false").lower() in ("1", "true", "y", "yes")
# verbose remote debug logging (show remote JSON/text previews, extraction diagnostics)
LLM_DEBUG_REMOTE = os.getenv("LLM_DEBUG_REMOTE", "false").lower() in ("1", "true", "y", "yes")

# ------------------ startup diagnostic (masked) ------------------
def _mask_key(s: Optional[str]) -> str:
    if not s:
        return "<missing>"
    s = str(s)
    return (s[:6] + "..." + s[-4:]) if len(s) > 12 else "<present>"

logger.info(
    "[llm_phraser] startup: LLM_MODE=%s LLM_PROVIDER=%s LLM_REMOTE_URL=%s LLM_MODEL=%s LLM_REMOTE_TIMEOUT=%s",
    LLM_MODE, LLM_REMOTE_PROVIDER, LLM_REMOTE_URL or "<none>", LLM_MODEL, LLM_REMOTE_TIMEOUT
)
logger.info(
    "[llm_phraser] startup keys: GROQ=%s LLM_API=%s OPENAI=%s HF=%s DEFAULT_MIN_PRICE_RATIO=%s CONTEXT_MATCHES=%s ALLOW_REMOTE_NO_PRICE=%s DEBUG_REMOTE=%s",
    _mask_key(GROQ_API_KEY), _mask_key(LLM_API_KEY), _mask_key(OPENAI_API_KEY), _mask_key(HF_TOKEN), DEFAULT_MIN_PRICE_RATIO,
    LLM_CONTEXT_MATCHES, LLM_ALLOW_REMOTE_WITHOUT_PRICE, LLM_DEBUG_REMOTE
)

# ------------------ requests session w/ retries ------------------
from requests.adapters import HTTPAdapter, Retry

_session = requests.Session()
_retries = Retry(total=2, backoff_factor=0.25, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST", "GET"])
_adapter = HTTPAdapter(max_retries=_retries)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

# ------------------ runtime helpers ------------------
def _auth_headers() -> Dict[str, str]:
    """
    Build Authorization headers: prefer provider-specific key, fall back to generic.
    Always include Content-Type.
    """
    h = {"Content-Type": "application/json"}
    provider = LLM_REMOTE_PROVIDER.upper()
    # Provider-specific priority
    if provider == "GROQ" and GROQ_API_KEY:
        h["Authorization"] = f"Bearer {GROQ_API_KEY}"
        return h
    # Generic fallbacks
    if LLM_API_KEY:
        h["Authorization"] = f"Bearer {LLM_API_KEY}"
        return h
    if OPENAI_API_KEY:
        h["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        return h
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
        return h
    # No auth found — return headers without Authorization; callers will log
    return h

_model_lock = threading.Lock()
_local_ready = False
_tokenizer = None
_model = None

# ------------------ context sanitizer ------------------
def _sanitize_context(ctx: Optional[str]) -> str:
    if not ctx:
        return ""
    s = re.sub(r'[\x00-\x1f\x7f]+', ' ', ctx)
    s = re.sub(r'\s+', ' ', s).strip()
    # remove currency symbols to avoid double-embedding them in templates
    s = re.sub(r'[₦$€£]', '', s)
    return s[:800]

# ------------------ small context-language heuristic ------------------
# These are compact high-signal tokens to detect if text is likely Yoruba/Hausa/Igbo.
# We keep the list conservative so we don't over-trigger.
_CTX_MARKERS: Dict[str, List[str]] = {
    "yo": ["mo", "le", "san", "ra", "rà", "rá", "ṣe", "jẹ", "kọ"],  # common short tokens in Yoruba phrases
    "ha": ["zan", "iya", "biya", "sayi", "saya", "naira", "ka", "zai"],          # Hausa clues
    "ig": ["enwere", "nwoke", "nne", "nwanne", "daalụ", "ị", "na"],  # Igbo clues (conservative)
}
_CTX_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

def _detect_local_language_from_context(ctx: Optional[str]) -> Optional[str]:
    """
    Heuristic: if context contains multiple tokens that match a local-language marker list,
    return the language code 'yo'|'ha'|'ig'. Requires >= LLM_CONTEXT_MATCHES hits to trigger.
    """
    if not ctx:
        return None
    s = ctx.lower()
    # tokenize
    tokens = set(_CTX_WORD_RE.findall(s))
    for lang, markers in _CTX_MARKERS.items():
        # count matches: allow either token match or substring match to be more robust
        matches = 0
        for m in markers:
            if not m:
                continue
            try:
                if m in tokens:
                    matches += 1
                elif m in s:
                    # substring appearance (helps catch forms like "enwere", "nwoke", etc.)
                    matches += 1
            except Exception:
                # defensive: skip broken marker
                continue
        if matches >= max(1, LLM_CONTEXT_MATCHES):
            logger.debug("[llm_phraser] context heuristic matched lang=%s matches=%s tokens_sample=%s (threshold=%s)",
                         lang, matches, list(tokens)[:10], LLM_CONTEXT_MATCHES)
            return lang
    return None

# ------------------ local model loader (optional) ------------------
def _try_load_local_model():
    global _local_ready, _tokenizer, _model
    if _local_ready:
        return
    with _model_lock:
        if _local_ready:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            logger.info("[llm_phraser] Loading local model: %s", LLM_MODEL)
            _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_fast=True)
            try:
                _model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL, load_in_4bit=True, device_map="auto", trust_remote_code=True
                )
                logger.info("[llm_phraser] Loaded local model in 4-bit mode.")
            except Exception:
                logger.warning("[llm_phraser] 4-bit load failed, trying standard load.")
                _model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
            _local_ready = True
            logger.info("[llm_phraser] Local model ready.")
        except Exception as e:
            logger.exception("[llm_phraser] Local model load failed: %s", e)
            _local_ready = False
            _tokenizer = None
            _model = None

# ------------------ response extractor (generic) ------------------
def _extract_text_from_remote_response(j: Any) -> Optional[str]:
    """General extractor used for HF/Groq/OpenAI-style shapes."""
    try:
        if isinstance(j, dict):
            # HF inference api common shape
            if "generated_text" in j and isinstance(j["generated_text"], str):
                return j["generated_text"].strip()
            # OpenAI/Groq style: choices -> message/content or text
            if "choices" in j and j["choices"]:
                c0 = j["choices"][0]
                if isinstance(c0, dict):
                    if "message" in c0 and isinstance(c0["message"], dict):
                        msg = c0["message"]
                        if "content" in msg and isinstance(msg["content"], str):
                            return msg["content"].strip()
                        if "content" in msg and isinstance(msg["content"], list):
                            parts = []
                            for p in msg["content"]:
                                if isinstance(p, dict) and "text" in p:
                                    parts.append(p["text"])
                                elif isinstance(p, str):
                                    parts.append(p)
                            if parts:
                                return " ".join(parts).strip()
                    if "text" in c0 and isinstance(c0["text"], str):
                        return c0["text"].strip()
            if "outputs" in j and isinstance(j["outputs"], list) and j["outputs"]:
                out0 = j["outputs"][0]
                if isinstance(out0, dict):
                    for key in ("generated_text", "text", "content", "prediction"):
                        if key in out0 and isinstance(out0[key], str):
                            return out0[key].strip()
                    cont = out0.get("content")
                    if isinstance(cont, list):
                        texts = []
                        for block in cont:
                            if isinstance(block, dict):
                                if "text" in block and isinstance(block["text"], str):
                                    texts.append(block["text"])
                                elif "content" in block and isinstance(block["content"], str):
                                    texts.append(block["content"])
                        if texts:
                            return " ".join(texts).strip()
        if isinstance(j, list) and j:
            if isinstance(j[0], str):
                return j[0].strip()
            if isinstance(j[0], dict):
                if "generated_text" in j[0] and isinstance(j[0]["generated_text"], str):
                    return j[0]["generated_text"].strip()
                if "text" in j[0] and isinstance(j[0]["text"], str):
                    return j[0]["text"].strip()
    except Exception:
        pass
    return None

# ------------------ remote caller ------------------
def _call_remote_llm(prompt: str, timeout: int = None, lang_key: str = "en") -> Optional[str]:
    """
    Calls the configured remote LLM provider.
    lang_key en|pcm used to force remote response language (en or pcm).
    """
    provider = LLM_REMOTE_PROVIDER.upper()
    headers = _auth_headers()
    url = LLM_REMOTE_URL or None
    timeout = timeout or LLM_REMOTE_TIMEOUT

    logger.debug("[llm_phraser] _call_remote_llm provider=%s url=%s model=%s timeout=%s", provider, url, LLM_MODEL, timeout)
    if not url:
        if provider == "GROQ":
            url = "https://api.groq.com/openai/v1/chat/completions"
        elif provider == "HF":
            url = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"
        elif provider == "OPENAI":
            url = "https://api.openai.com/v1/chat/completions"
        else:
            logger.warning("[llm_phraser] No LLM_REMOTE_URL configured and no sane default for provider=%s", provider)
            return None

    if "Authorization" not in headers:
        logger.warning("[llm_phraser] No Authorization header set for provider=%s — remote call may be rejected", provider)

    if lang_key == "pcm":
        remote_sys_lang = "You MUST reply only in Nigerian Pidgin (pcm). Do not include English."
    else:
        remote_sys_lang = "You MUST reply only in English (do not include other languages)."

    def _try_post(cur_url: str) -> Optional[requests.Response]:
        try:
            if provider == "GROQ" or provider == "OPENAI":
                payload = {
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": f"You are a polite Nigerian market seller and negotiator. {remote_sys_lang}"},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": LLM_MAX_TOKENS,
                    "temperature": LLM_TEMPERATURE,
                    "top_p": LLM_TOP_P
                }
                return _session.post(cur_url, json=payload, headers=headers, timeout=timeout)

            if provider == "HF":
                inputs = f"{remote_sys_lang}\n\n{prompt}"
                payload = {"inputs": inputs, "parameters": {"temperature": LLM_TEMPERATURE, "max_new_tokens": LLM_MAX_TOKENS, "top_p": LLM_TOP_P}}
                return _session.post(cur_url, json=payload, headers=headers, timeout=timeout)

            logger.warning("[llm_phraser] Unknown provider in runtime: %s", provider)
            return None
        except Exception as e:
            logger.exception("[llm_phraser] _try_post exception for url=%s: %s", cur_url, e)
            return None

    tried_urls = []
    candidate_urls: List[str] = [url]
    if provider == "GROQ" or provider == "OPENAI":
        if "chat/completions" in url:
            candidate_urls.append(url.replace("chat/completions", "chat"))
            candidate_urls.append(url.replace("chat/completions", "completions"))
        if url.endswith("/chat"):
            candidate_urls.append(url + "/completions")
    for cur in candidate_urls:
        if cur in tried_urls:
            continue
        tried_urls.append(cur)
        logger.debug("[llm_phraser] Attempting remote LLM POST to %s", cur)
        resp = _try_post(cur)
        if resp is None:
            continue
        body_preview = (resp.text or "")[:4000]
        if LLM_DEBUG_REMOTE:
            logger.info("[llm_phraser] remote raw response preview (first 4000 chars): %s", body_preview)
        logger.debug("[llm_phraser] remote status=%s preview=%s", resp.status_code, body_preview[:1000])
        if resp.status_code >= 400:
            logger.warning("[llm_phraser] remote returned HTTP %s for %s: %s", resp.status_code, cur, body_preview[:1000])
            continue
        try:
            j = resp.json()
        except Exception:
            logger.exception("[llm_phraser] failed to parse JSON from remote response for %s", cur)
            j = None
        if j is not None:
            if LLM_DEBUG_REMOTE:
                logger.info("[llm_phraser] remote JSON: %s", str(j)[:4000])
            txt = _extract_text_from_remote_response(j)
            if txt:
                if LLM_DEBUG_REMOTE:
                    logger.info("[llm_phraser] remote extracted text preview: %s", (txt[:1000] + "...") if len(txt) > 1000 else txt)
                return txt.strip()
            if isinstance(j, str):
                return j.strip()
        if resp.text and len(resp.text) > 0:
            raw = resp.text.strip()
            if raw:
                if LLM_DEBUG_REMOTE:
                    logger.info("[llm_phraser] remote raw text fallback used: %s", raw[:1000])
                return raw
    logger.warning("[llm_phraser] remote LLM call failed after trying %d URLs", len(tried_urls))
    return None

# ------------------ local generation helper ------------------
def _run_local_generation(prompt: str) -> Optional[str]:
    global _local_ready, _tokenizer, _model
    if not _local_ready:
        _try_load_local_model()
    if not _local_ready or _model is None or _tokenizer is None:
        return None
    try:
        import torch
        inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
        gen = _model.generate(**inputs, max_new_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE, do_sample=True)
        out = _tokenizer.decode(gen[0], skip_special_tokens=True)
        if out.startswith(prompt):
            out = out[len(prompt):].strip()
        return out.strip()
    except Exception as e:
        logger.exception("[llm_phraser] local generation failed: %s", e)
        return None

# ------------------ templates (unchanged structure, lists allowed) ------------------
_TEMPLATES = {
    "en": {
        "accept": [
            "My dear, deal done! ₦{price:,} for {product}, you’ll shine! Pay now!",
            "Thank you, we’re set! ₦{price:,} for {product}, grab it quick!"
        ],
        "counter": [
            "Nice try, but let’s do ₦{price:,} for {product}. Take it now!",
            "I hear you! Best I can do is ₦{price:,} for {product}. Buy now?"
        ],
        "reject": [
            "Ouch, too low for {product}! Best is ₦{price:,}, don’t miss it!",
            "That price no fit o! ₦{price:,} for {product}, let’s deal!"
        ],
        "clarify": [
            "What price are you thinking for {product}? Share now, let’s deal!",
            "My friend, what’s your budget for {product}? Let me know!"
        ]
    },
    "pcm": {
        "accept": [
            "Correct deal! ₦{price:,} for {product}, e go sweet you! Pay sharp!",
            "Na so! ₦{price:,} for {product}, you go love am! Buy now!"
        ],
        "counter": [
            "You try, but I fit do ₦{price:,} for {product}. Oya, take am!",
            "No wahala, I go give ₦{price:,} for {product}. You dey in?",
        ],
        "reject": [
            "Haba, dat one too small for {product}! ₦{price:,}, abeg grab am!",
            "No way o, {product} worth ₦{price:,}! Oya, make we talk business!"
        ],
        "clarify": [
            "Abeg, wetin price you dey reason for {product}? Talk quick!",
            "Bros, which price you dey eye for {product}? Tell me now!"
        ]
    },
    "yo": {
        "accept": [
            "O ṣeun! ₦{price:,} fún {product}, ẹ rẹwà pẹ̀lú rẹ̀! Ra báyìí!",
            "Ẹ ṣé, a ti ṣe! ₦{price:,} fún {product}, kíá ra ni!",
        ],
        "counter": [
            "Ó dá, ṣùgbọ́n mo lè gba ₦{price:,} fún {product}. Ṣé ẹ rà?",
            "Ẹ̀gbọ́n, mo lè fun ọ ni ₦{price:,} fún {product}. Ra kíákíá!",
        ],
        "reject": [
            "Há, kéré jù fún {product}! ₦{price:,} ni mo lè gba, má ṣe yọ̀ọ̀!",
            "Kò tó fún {product}! ₦{price:,} ni, jọ̀ọ́, ṣé ẹ rà?",
        ],
        "clarify": [
            "Jọ̀ọ́, mélòó ni ẹ rò fún {product}? Sọ fún mi kíákíá!",
            "Ẹ̀gbọ́n, kí ni ìdíwọ̀n rẹ fún {product}? Sọ báyìí!"
        ]
    },
    "ha": {
        "accept": [
            "Nagode! ₦{price:,} domin {product}, zai sa ka haske! Saya yanzu!",
            "Na gode, mun gama! ₦{price:,} don {product}, ka saya da sauri!",
        ],
        "counter": [
            "Ka ji, amma zan iya ₦{price:,} don {product}. Ka saya yanzu?",
            "Nagode, zan iya baka ₦{price:,} don {product}. Shin za ka saya?",
        ],
        "reject": [
            "Haba, wannan ƙasa ga {product}! ₦{price:,}, ka ji yanzu!",
            "Farashin bai isa ba ga {product}! ₦{price:,}, ka saya ko?",
        ],
        "clarify": [
            "Don Allah, wane farashi kake tunani don {product}? Fada min yanzu!",
            "Aboki, wane farashi kake so don {product}? Ka gaya min!",
        ]
    },
    "ig": {
        "accept": [
            "Daalụ! ₦{price:,} maka {product}, ọ ga-eme gị ka ọgaranya! Zụta ugbu a!",
            "Ekele, anyị kwụsịrị! ₦{price:,} maka {product}, zụta ngwa ngwa!",
        ],
        "counter": [
            "Ị dị mma, mana m nwere ike inye ₦{price:,} maka {product}. Zụta ugbu a?",
            "Daalụ, enwere m ike ịnye ₦{price:,} maka {product}. Ị dị njikere?",
        ],
        "reject": [
            "Haba, ọnụahịa a dị ala maka {product}! ₦{price:,}, biko zụta ya!",
            "Ọ dịghị mma maka {product}! ₦{price:,} ka m nwere, zụta ugbu a!",
        ],
        "clarify": [
            "Biko, kedụ ọnụahịa ị na-eche maka {product}? Kwee ngwa ngwa!",
            "Nna, gịnị ka ị chọrọ maka ọnụahịa {product}? Kwee m ugbu a!"
        ]
    }
}

FEW_SHOT_PROMPT = """
    SYSTEM: You are a vibrant Nigerian market seller and expert negotiator, fluent in English (en), Pidgin (pcm), Yoruba (yo), Hausa (ha), and Igbo (ig), speaking like a native with authentic market energy. Use {lang_key} tone (short, direct, rich with Nigerian charm and banter).
        - Detect if a customer is speaking to you in Yoruba, English, Pidgin, Igbo, Hausa, or mixed languages and reply in the same language

        - Infuse replies with Nigerian market flair: use slang, proverbs, or playful haggling to build trust and warmth.

        - Appeal to emotions (gratitude, community, pride) to persuade

        - Be firm yet polite to protect seller margin and brand value.

        - Use currency symbol ₦ and round prices to whole naira.

        - Keep replies short (1-2 sentences, max 40 words). End with a clear CTA.

        - Do not invent shipping, freebies, or discounts.

        - Echo numeric prices exactly as provided in {final_price}.

Context: {context}

NUMERIC_GROUNDS:
final_price: {final_price}
"""

# ------------------ policy (refined) ------------------
def compute_counter_price(base_price: int, offer: Optional[int], min_price: Optional[int] = None) -> Tuple[str, Optional[int]]:
    if offer is None:
        return "ASK_CLARIFY", None
    try:
        base = int(base_price)
        off = int(offer)
    except Exception:
        return "ASK_CLARIFY", None
    if base <= 0:
        return "ASK_CLARIFY", None

    computed_min = int(round(base * DEFAULT_MIN_PRICE_RATIO))
    if min_price is not None:
        try:
            min_eff = max(int(min_price), computed_min)
        except Exception:
            min_eff = computed_min
    else:
        min_eff = computed_min
    min_eff = min(min_eff, base)

    buyer_pct_of_base = off / base if base > 0 else 0.0
    buyer_pct_of_min = off / min_eff if min_eff > 0 else 0.0

    if buyer_pct_of_base >= 0.90:
        return "ACCEPT", off

    def make_prop(pct_low: float, pct_high: float, bias_toward_buyer: bool = False) -> int:
        pct = random.uniform(pct_low, pct_high)
        prop = int(round(base * pct))
        if bias_toward_buyer and off is not None:
            prop = int(round((prop + off) / 2.0))
        if prop < min_eff:
            prop = min_eff
        prop = min(prop, base)
        return int(prop)

    if buyer_pct_of_min >= 0.80:
        prop = make_prop(0.60, 0.75, bias_toward_buyer=True)
        action = "COUNTER" if prop != off else "ACCEPT"
        return action, prop

    if 0.50 <= buyer_pct_of_min < 0.80:
        prop = make_prop(0.60, 0.75, bias_toward_buyer=True)
        return "COUNTER", prop

    prop = make_prop(0.70, 0.80, bias_toward_buyer=False)
    if prop == min_eff and prop < base:
        try:
            prop = min_eff + 1
            if prop > base:
                prop = min_eff
        except Exception:
            pass
    return "REJECT", prop

# ------------------ improved numeric matcher ------------------
def _reply_contains_price(reply: str, price: int) -> bool:
    if not reply or price is None:
        return False
    cleaned = re.sub(r'[₦$€£]', '', reply)
    tokens = re.findall(r'\d[\d,]*', cleaned)
    for t in tokens:
        try:
            val = int(t.replace(",", ""))
            if val == int(price):
                return True
        except Exception:
            continue
    return False

# ------------------ helpers for dynamic negotiation ------------------
def _compute_floor(min_price: Optional[int], base_price: int) -> int:
    try:
        if min_price is not None and int(min_price) > 0:
            mp = int(min_price)
        else:
            mp = int(round(base_price * DEFAULT_MIN_PRICE_RATIO))
        floor = int(math.ceil(mp * 1.10))
        return max(floor, 1)
    except Exception:
        return max(int(round(base_price * DEFAULT_MIN_PRICE_RATIO * 1.10)), 1)

def _initial_dynamic_counter(buyer_offer: Optional[int], min_price: int, base_price: int) -> int:
    dyn = _dynamic_counter_price(buyer_offer, min_price, base_price)
    floor = _compute_floor(min_price, base_price)
    return max(dyn, floor)

def _next_proposal_after_reject(prev_proposals: List[int], buyer_offer: Optional[int], min_price: int, base_price: int) -> int:
    floor = _compute_floor(min_price, base_price)
    try:
        if not prev_proposals:
            return _initial_dynamic_counter(buyer_offer, min_price, base_price)
        last = int(prev_proposals[-1])
        if last <= floor:
            return floor
        gap = max(last - floor, 0)
        step1 = int(math.ceil(gap * 0.40))
        step2 = int(math.ceil(base_price * 0.03))
        step = max(step1, step2, 1)
        next_prop = last - step
        next_prop = max(next_prop, floor)
        return int(next_prop)
    except Exception as e:
        logger.exception("[llm_phraser] _next_proposal_after_reject error: %s", e)
        return floor

def _dynamic_counter_price(buyer_offer: Optional[int], min_price: int, base_price: int) -> int:
    try:
        mp = int(min_price) if min_price is not None else int(round(base_price * DEFAULT_MIN_PRICE_RATIO))
        bp = int(base_price)
        if mp <= 0:
            mp = int(round(bp * DEFAULT_MIN_PRICE_RATIO))
        candidate75 = max(int(round(bp * 0.75)), mp)
        candidate60 = max(int(round(bp * 0.60)), mp)
        if buyer_offer is None:
            return candidate75
        bo = int(buyer_offer)
        closeness = (bo / mp) if mp > 0 else (bo / bp if bp > 0 else 0.0)
        if closeness >= 0.8:
            mid = int(round((bo + candidate75) / 2.0))
            return max(mid, mp)
        if 0.5 <= closeness < 0.8:
            mid = int(round((candidate75 + bo) / 2.0))
            return max(min(mid, candidate75), mp)
        return max(candidate75, mp)
    except Exception as e:
        logger.exception("[llm_phraser] _dynamic_counter_price error: %s", e)
        try:
            fallback = int(round(base_price * 0.75))
            return max(fallback, int(min_price or 0))
        except Exception:
            return int(min_price or base_price)

def _format_naira(n: Optional[int]) -> str:
    try:
        return f"₦{int(n):,}"
    except Exception:
        return f"₦{n}"

# ------------------ template rendering helper ------------------
def _choose_template_variant(candidate: Any, ratio: Optional[float]) -> str:
    if isinstance(candidate, list):
        if not candidate:
            return ""
        if ratio is None:
            return random.choice(candidate)
        try:
            if ratio >= 0.8:
                idx = 1 if len(candidate) > 1 else 0
            elif ratio >= 0.5:
                idx = 0
            else:
                idx = 0
            return candidate[idx]
        except Exception:
            return random.choice(candidate)
    return str(candidate)

def _render_template_reply(template_map: Dict[str, Any], action_key: str, price: Optional[int], product_name: str, ratio: Optional[float] = None) -> str:
    try:
        candidate = template_map.get(action_key, template_map.get("counter"))
        tmpl = _choose_template_variant(candidate, ratio)
        if not isinstance(tmpl, str):
            tmpl = str(tmpl)
        tpl_price_int = None
        try:
            tpl_price_int = int(price) if price is not None else 0
        except Exception:
            tpl_price_int = 0
        return tmpl.format(price=tpl_price_int, product=product_name)
    except Exception as e:
        logger.exception("[llm_phraser] template rendering failed: %s", e)
        try:
            return f"Our counter price is {_format_naira(price)} for {product_name}."
        except Exception:
            return f"Our counter price is ₦{price} for {product_name}."

# ------------------ language normalization helper ------------------
def _normalize_lang_code(s: Optional[str]) -> str:
    if not s:
        return "en"
    s0 = str(s).strip().lower()
    if s0 in ("en", "eng", "english", "en_us", "en-gb", "en-us"):
        return "en"
    if s0 in ("pcm", "pidgin", "pidgin_ng", "pcm_ng", "pcm-nigeria", "pidgin-nigeria"):
        return "pcm"
    if s0.startswith("yo") or s0 in ("yoruba", "yoruba_ng", "yo_ng"):
        return "yo"
    if s0.startswith("ha") or s0 in ("hausa", "hausa_ng", "ha_ng"):
        return "ha"
    if s0.startswith("ig") or s0 in ("igbo", "igbo_ng", "ig_ng"):
        return "ig"
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

# ------------------ main phrase() ------------------
def phrase(decision: Dict[str, Any], product: Dict[str, Any], lang: str = "en", context: Optional[str] = None) -> str:
    """
    decision: dict possibly containing 'action', 'price', 'offer', 'meta'
    product: dict with 'name' and 'base_price'
    lang: language key (en|pcm|yo|ig|ha) or variant
    Returns a user-facing string reply.
    """
    # Normalize incoming language codes
    lang_key = _normalize_lang_code(lang)

    prod_name = product.get("name") or product.get("id") or "product"
    base_price = int(product.get("base_price", 0))

    # read decision fields
    explicit_action = (decision.get("action") or "").upper() or None
    explicit_price = decision.get("price")
    buyer_offer = None
    if decision.get("offer") is not None:
        try:
            buyer_offer = int(decision.get("offer"))
        except Exception:
            buyer_offer = None

    # negotiation meta (may carry min_price and previous proposals)
    meta = decision.get("meta") or {}
    min_price_meta = None
    try:
        if isinstance(meta, dict) and "min_price" in meta:
            min_price_meta = int(meta["min_price"])
    except Exception:
        min_price_meta = None

    prev_proposals: List[int] = []
    try:
        if isinstance(meta, dict) and "prev_proposals" in meta and isinstance(meta["prev_proposals"], list):
            prev_proposals = [int(x) for x in meta["prev_proposals"] if isinstance(x, (int, str)) or hasattr(x, "__int__")]
    except Exception:
        prev_proposals = []

    floor = _compute_floor(min_price_meta, base_price)

    # If the decision is ESCALATE (from guard), compute a dynamic counter price >= floor
    if explicit_action == "ESCALATE":
        if min_price_meta is not None:
            if prev_proposals:
                dyn_price = _next_proposal_after_reject(prev_proposals, buyer_offer, min_price_meta, base_price)
            else:
                dyn_price = _initial_dynamic_counter(buyer_offer, min_price_meta, base_price)
            dyn_price = max(dyn_price, floor)
            logger.info("[llm_phraser] ESCALATE -> dyn_price=%s (floor=%s) prev_proposals=%s buyer_offer=%s", dyn_price, floor, prev_proposals, buyer_offer)
            explicit_action = "COUNTER"
            explicit_price = dyn_price
            meta = dict(meta or {})
            meta["next_proposal"] = dyn_price
            meta["floor"] = floor
            meta.setdefault("prev_proposals", prev_proposals)
        else:
            explicit_action = "REJECT"
            explicit_price = explicit_price if explicit_price is not None else (buyer_offer or base_price)

    if explicit_action:
        computed_action = explicit_action
        computed_price = explicit_price if explicit_price is not None else (buyer_offer or base_price)
    else:
        computed_action, computed_price = compute_counter_price(base_price, buyer_offer)

    if min_price_meta is not None and computed_price is not None:
        computed_price = max(int(computed_price), floor)

    final_price = int(computed_price) if computed_price is not None else None

    # sanitize context for prompts
    sanitized_ctx = _sanitize_context(context)
    fs = FEW_SHOT_PROMPT.format(lang_key=lang_key, context=sanitized_ctx, final_price=final_price)

    input_block = (
        f"\nINPUT:\nproduct_name: \"{prod_name}\"\n"
        f"base_price: {base_price}\n"
        f"offer: {buyer_offer if buyer_offer is not None else 'null'}\n"
        f"counter_price: {final_price if final_price is not None else 'null'}\n"
        f"decision: {computed_action}\n"
    )
    instruction = "\nINSTRUCTION:\nReply in one or two short sentences that are friendly, respectful, persuasive and end with a clear next step (CTA). Match the numeric values shown above exactly. Keep replies short and culturally appropriate."
    prompt = "\n".join(["SYSTEM PROMPT (few-shot examples):", fs, input_block, instruction])

    logger.debug("[llm_phraser] phrase() computed_action=%s final_price=%s prod=%s lang=%s floor=%s prev_proposals=%s",
                 computed_action, final_price, prod_name, lang_key, floor, prev_proposals)

    # Compute ratio for tone selection when templates are used
    min_price_for_ratio = None
    try:
        if min_price_meta is not None:
            min_price_for_ratio = int(min_price_meta)
        else:
            min_price_for_ratio = int(round(base_price * DEFAULT_MIN_PRICE_RATIO))
    except Exception:
        min_price_for_ratio = int(round(base_price * DEFAULT_MIN_PRICE_RATIO))

    ratio = None
    try:
        if min_price_for_ratio > 0 and buyer_offer is not None:
            ratio = float(buyer_offer) / float(min_price_for_ratio)
    except Exception:
        ratio = None

    # --- LANGUAGE-BASED TEMPLATE OVERRIDE ---
    # STRICT: only Yoruba, Hausa and Igbo MUST use templates (no remote/local)
    TEMPLATE_ONLY_LANGS = {"yo", "ha", "ig"}

    # If lang_key not already in template-only set, use context heuristic to detect
    if lang_key not in TEMPLATE_ONLY_LANGS:
        detected_from_ctx = _detect_local_language_from_context(sanitized_ctx)
        if detected_from_ctx in TEMPLATE_ONLY_LANGS:
            logger.info("[llm_phraser] overriding lang_key -> %s based on context heuristic", detected_from_ctx)
            lang_key = detected_from_ctx

    if lang_key in TEMPLATE_ONLY_LANGS:
        template_map = _TEMPLATES.get(lang_key, _TEMPLATES["en"])
        action_map = {"ACCEPT": "accept", "REJECT": "reject", "COUNTER": "counter", "ASK_CLARIFY": "clarify"}
        key = action_map.get(computed_action, "counter")
        try:
            tpl_price = final_price if final_price is not None else base_price
            rendered = _render_template_reply(template_map, key, tpl_price, prod_name, ratio)
            try:
                meta_out = dict(meta or {})
                prev = meta_out.get("prev_proposals", []) or []
                if computed_action == "COUNTER":
                    prev = list(prev) + [int(tpl_price)]
                    meta_out["prev_proposals"] = prev
                meta_out["floor"] = floor
            except Exception:
                meta_out = meta
            logger.info("[llm_phraser] template_override=true lang=%s action=%s price=%s prod=%s ratio=%s meta_prev=%s",
                        lang_key, computed_action, tpl_price, prod_name, ratio, meta_out.get("prev_proposals") if isinstance(meta_out, dict) else None)
            return rendered
        except Exception:
            logger.exception("[llm_phraser] template override failed for lang=%s key=%s — falling through", lang_key, key)
            # safe fallback to English template to avoid crash
            return _render_template_reply(_TEMPLATES.get("en"), "counter", final_price or base_price, prod_name, ratio)

    # Safety guard: if we reach remote/local generation, do not allow TEMPLATE_ONLY_LANGS to be handled by remote/local.
    # (This is defensive in case calling code changed flow; template-only languages MUST use templates.)
    if lang_key in TEMPLATE_ONLY_LANGS:
        logger.debug("[llm_phraser] lang_key=%s is template-only; skipping remote/local generation (defensive guard)", lang_key)
        template_map = _TEMPLATES.get(lang_key, _TEMPLATES["en"])
        key = {"ACCEPT": "accept", "REJECT": "reject", "COUNTER": "counter", "ASK_CLARIFY": "clarify"}.get(computed_action, "counter")
        return _render_template_reply(template_map, key, final_price if final_price is not None else base_price, prod_name, ratio)

    # --- REMOTE preferred (English or pidgin) ---
    if LLM_MODE == "REMOTE":
        out = None
        try:
            # Do not call remote LLM for template-only languages (defensive)
            if lang_key in TEMPLATE_ONLY_LANGS:
                logger.debug("[llm_phraser] skipping remote LLM for template-only language=%s", lang_key)
            else:
                remote_lang_forcing = "en" if lang_key not in ("pcm",) else "pcm"
                out = _call_remote_llm(prompt, lang_key=remote_lang_forcing)
                # determine whether reply contains the expected numeric price
                price_match = bool(out and final_price is not None and _reply_contains_price(out, final_price))
                # Accept remote if price matches OR operator explicitly allows remote without price.
                if out and (final_price is None or price_match or LLM_ALLOW_REMOTE_WITHOUT_PRICE):
                    logger.info("[llm_phraser] remote accepted (price_match=%s allow_without=%s) lang_forcing=%s preview=%s",
                                price_match, LLM_ALLOW_REMOTE_WITHOUT_PRICE, remote_lang_forcing,
                                (out[:120] + "...") if len(out) > 120 else out)
                    return out.strip()
                logger.warning("[llm_phraser] remote returned no usable text or numeric mismatch; falling back (price_match=%s allow_without=%s)",
                               price_match, LLM_ALLOW_REMOTE_WITHOUT_PRICE)
        except Exception:
            logger.exception("[llm_phraser] remote generation error")

    # --- LOCAL fallback ---
    if LLM_MODE == "LOCAL":
        try:
            # Do not run local generation for template-only languages
            if lang_key in TEMPLATE_ONLY_LANGS:
                logger.debug("[llm_phraser] skipping local generation for template-only language=%s", lang_key)
            else:
                out = _run_local_generation(prompt)
                price_match = bool(out and final_price is not None and _reply_contains_price(out, final_price))
                if out and (final_price is None or price_match or LLM_ALLOW_REMOTE_WITHOUT_PRICE):
                    logger.info("[llm_phraser] local_generation_ok (price_match=%s allow_without=%s) lang=%s preview=%s",
                                price_match, LLM_ALLOW_REMOTE_WITHOUT_PRICE, lang_key, (out[:120] + "...") if len(out) > 120 else out)
                    return out.strip()
        except Exception:
            logger.exception("[llm_phraser] local generation error")

    # --- TEMPLATE fallback (final) ---
    template_map = _TEMPLATES.get(lang_key, _TEMPLATES["en"])
    action_map = {"ACCEPT": "accept", "REJECT": "reject", "COUNTER": "counter", "ASK_CLARIFY": "clarify"}
    key = action_map.get(computed_action, "counter")
    try:
        tpl_price = final_price if final_price is not None else base_price
        logger.info("[llm_phraser] final_template lang=%s action=%s price=%s prod=%s prev_proposals=%s floor=%s",
                    lang_key, computed_action, tpl_price, prod_name, prev_proposals, floor)
        return _render_template_reply(template_map, key, tpl_price, prod_name, ratio)
    except Exception:
        try:
            return f"Our counter price is {_format_naira(final_price)} for {prod_name}."
        except Exception:
            return f"Our counter price is ₦{final_price} for {prod_name}."
