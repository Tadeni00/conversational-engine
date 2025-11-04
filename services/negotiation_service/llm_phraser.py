# from __future__ import annotations
# import os
# import re
# import threading
# import logging
# import random
# import math
# from typing import Dict, Any, Optional, Tuple, List
# import requests

# logger = logging.getLogger("llm_phraser")
# if not logger.handlers:
#     logging.basicConfig(level=logging.INFO)

# # ------------------ Config (env) ------------------
# LLM_MODE = os.getenv("LLM_MODE", "REMOTE").upper()            # REMOTE | LOCAL | TEMPLATE
# LLM_REMOTE_PROVIDER = os.getenv("LLM_REMOTE_PROVIDER", "GROQ").upper()  # GROQ | HF | OPENAI
# LLM_MODEL = os.getenv("LLM_MODEL", "mistral-7b-instruct")
# LLM_REMOTE_URL = os.getenv("LLM_REMOTE_URL", "").strip()
# LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
# HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
# LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "80"))
# LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))

# # new: make remote timeout configurable
# LLM_REMOTE_TIMEOUT = int(os.getenv("LLM_REMOTE_TIMEOUT", "20"))

# # negotiation default min ratio (used when product.min_price missing)
# DEFAULT_MIN_PRICE_RATIO = float(os.getenv("DEFAULT_MIN_PRICE_RATIO", "0.5"))

# # ------------------ startup diagnostic (masked) ------------------
# def _mask_key(s: Optional[str]) -> str:
#     if not s:
#         return "<missing>"
#     s = str(s)
#     return (s[:6] + "..." + s[-4:]) if len(s) > 12 else "<present>"

# logger.info(
#     "[llm_phraser] startup: LLM_MODE=%s LLM_PROVIDER=%s LLM_REMOTE_URL=%s LLM_MODEL=%s LLM_REMOTE_TIMEOUT=%s",
#     LLM_MODE, LLM_REMOTE_PROVIDER, LLM_REMOTE_URL or "<none>", LLM_MODEL, LLM_REMOTE_TIMEOUT
# )
# logger.info(
#     "[llm_phraser] startup keys: GROQ=%s LLM_API=%s OPENAI=%s HF=%s DEFAULT_MIN_PRICE_RATIO=%s",
#     _mask_key(GROQ_API_KEY), _mask_key(LLM_API_KEY), _mask_key(OPENAI_API_KEY), _mask_key(HF_TOKEN), DEFAULT_MIN_PRICE_RATIO
# )

# # ------------------ requests session w/ retries ------------------
# from requests.adapters import HTTPAdapter, Retry

# _session = requests.Session()
# _retries = Retry(total=2, backoff_factor=0.25, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST", "GET"])
# _adapter = HTTPAdapter(max_retries=_retries)
# _session.mount("http://", _adapter)
# _session.mount("https://", _adapter)

# # ------------------ runtime helpers ------------------
# def _auth_headers() -> Dict[str, str]:
#     """
#     Build Authorization headers: prefer provider-specific key, fall back to generic.
#     Always include Content-Type.
#     """
#     h = {"Content-Type": "application/json"}
#     provider = LLM_REMOTE_PROVIDER.upper()
#     # Provider-specific priority
#     if provider == "GROQ" and GROQ_API_KEY:
#         h["Authorization"] = f"Bearer {GROQ_API_KEY}"
#         return h
#     # Generic fallbacks
#     if LLM_API_KEY:
#         h["Authorization"] = f"Bearer {LLM_API_KEY}"
#         return h
#     if OPENAI_API_KEY:
#         h["Authorization"] = f"Bearer {OPENAI_API_KEY}"
#         return h
#     if HF_TOKEN:
#         h["Authorization"] = f"Bearer {HF_TOKEN}"
#         return h
#     # No auth found — return headers without Authorization; callers will log
#     return h

# _model_lock = threading.Lock()
# _local_ready = False
# _tokenizer = None
# _model = None

# # ------------------ context sanitizer ------------------
# def _sanitize_context(ctx: Optional[str]) -> str:
#     if not ctx:
#         return ""
#     s = re.sub(r'[\x00-\x1f\x7f]+', ' ', ctx)
#     s = re.sub(r'\s+', ' ', s).strip()
#     # remove currency symbols to avoid double-embedding them in templates
#     s = re.sub(r'[₦$€£]', '', s)
#     return s[:800]

# # ------------------ small context-language heuristic ------------------
# # These are compact high-signal tokens to detect if text is likely Yoruba/Hausa/Igbo.
# # We keep the list conservative so we don't over-trigger.
# _CTX_MARKERS: Dict[str, List[str]] = {
#     "yo": ["mo", "le", "san", "ra", "rà", "rá", "ṣe", "jẹ", "kọ"],  # common short tokens in Yoruba phrases
#     "ha": ["zan", "iya", "biya", "sayi", "saya", "naira", "ka", "zai"],          # Hausa clues
#     "ig": ["enwere", "nwoke", "nne", "nwanne", "daalụ", "ị", "na"],  # Igbo clues (conservative)
# }
# _CTX_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

# def _detect_local_language_from_context(ctx: Optional[str]) -> Optional[str]:
#     """
#     Heuristic: if context contains multiple tokens that match a local-language marker list,
#     return the language code 'yo'|'ha'|'ig'. Conservative: require >=1 strong hit (we keep it light).
#     """
#     if not ctx:
#         return None
#     s = ctx.lower()
#     # tokenize
#     tokens = set(_CTX_WORD_RE.findall(s))
#     for lang, markers in _CTX_MARKERS.items():
#         # count matches
#         matches = sum(1 for m in markers if m and (m in tokens or (" " + m + " ") in (" " + s + " ")))
#         if matches >= 1:
#             logger.debug("[llm_phraser] context heuristic matched lang=%s matches=%s tokens=%s", lang, matches, list(tokens)[:10])
#             return lang
#     return None

# # ------------------ local model loader (optional) ------------------
# def _try_load_local_model():
#     global _local_ready, _tokenizer, _model
#     if _local_ready:
#         return
#     with _model_lock:
#         if _local_ready:
#             return
#         try:
#             from transformers import AutoTokenizer, AutoModelForCausalLM
#             import torch
#             logger.info("[llm_phraser] Loading local model: %s", LLM_MODEL)
#             _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_fast=True)
#             try:
#                 _model = AutoModelForCausalLM.from_pretrained(
#                     LLM_MODEL, load_in_4bit=True, device_map="auto", trust_remote_code=True
#                 )
#                 logger.info("[llm_phraser] Loaded local model in 4-bit mode.")
#             except Exception:
#                 logger.warning("[llm_phraser] 4-bit load failed, trying standard load.")
#                 _model = AutoModelForCausalLM.from_pretrained(
#                     LLM_MODEL,
#                     device_map="auto",
#                     torch_dtype=torch.float16 if torch.cuda.is_available() else None,
#                     trust_remote_code=True
#                 )
#             _local_ready = True
#             logger.info("[llm_phraser] Local model ready.")
#         except Exception as e:
#             logger.exception("[llm_phraser] Local model load failed: %s", e)
#             _local_ready = False
#             _tokenizer = None
#             _model = None

# # ------------------ response extractor (generic) ------------------
# def _extract_text_from_remote_response(j: Any) -> Optional[str]:
#     """General extractor used for HF/Groq/OpenAI-style shapes."""
#     try:
#         if isinstance(j, dict):
#             # HF inference api common shape
#             if "generated_text" in j and isinstance(j["generated_text"], str):
#                 return j["generated_text"].strip()
#             # OpenAI/Groq style: choices -> message/content or text
#             if "choices" in j and j["choices"]:
#                 c0 = j["choices"][0]
#                 if isinstance(c0, dict):
#                     if "message" in c0 and isinstance(c0["message"], dict):
#                         msg = c0["message"]
#                         if "content" in msg and isinstance(msg["content"], str):
#                             return msg["content"].strip()
#                         if "content" in msg and isinstance(msg["content"], list):
#                             parts = []
#                             for p in msg["content"]:
#                                 if isinstance(p, dict) and "text" in p:
#                                     parts.append(p["text"])
#                                 elif isinstance(p, str):
#                                     parts.append(p)
#                             if parts:
#                                 return " ".join(parts).strip()
#                     if "text" in c0 and isinstance(c0["text"], str):
#                         return c0["text"].strip()
#             if "outputs" in j and isinstance(j["outputs"], list) and j["outputs"]:
#                 out0 = j["outputs"][0]
#                 if isinstance(out0, dict):
#                     for key in ("generated_text", "text", "content", "prediction"):
#                         if key in out0 and isinstance(out0[key], str):
#                             return out0[key].strip()
#                     cont = out0.get("content")
#                     if isinstance(cont, list):
#                         texts = []
#                         for block in cont:
#                             if isinstance(block, dict):
#                                 if "text" in block and isinstance(block["text"], str):
#                                     texts.append(block["text"])
#                                 elif "content" in block and isinstance(block["content"], str):
#                                     texts.append(block["content"])
#                         if texts:
#                             return " ".join(texts).strip()
#         if isinstance(j, list) and j:
#             if isinstance(j[0], str):
#                 return j[0].strip()
#             if isinstance(j[0], dict):
#                 if "generated_text" in j[0] and isinstance(j[0]["generated_text"], str):
#                     return j[0]["generated_text"].strip()
#                 if "text" in j[0] and isinstance(j[0]["text"], str):
#                     return j[0]["text"].strip()
#     except Exception:
#         pass
#     return None

# # ------------------ remote caller ------------------
# def _call_remote_llm(prompt: str, timeout: int = None, lang_key: str = "en") -> Optional[str]:
#     """
#     Calls the configured remote LLM provider.
#     lang_key en|pcm used to force remote response language (en or pcm).
#     If lang_key == 'pcm' the system prompt instructs the remote model to reply with
#     the single token '<UNABLE_PCM>' if it cannot respond in Pidgin; caller will fallback.
#     """
#     provider = LLM_REMOTE_PROVIDER.upper()
#     headers = _auth_headers()
#     url = LLM_REMOTE_URL or None
#     timeout = timeout or LLM_REMOTE_TIMEOUT

#     logger.debug("[llm_phraser] _call_remote_llm provider=%s url=%s model=%s timeout=%s", provider, url, LLM_MODEL, timeout)
#     if not url:
#         if provider == "GROQ":
#             url = "https://api.groq.com/openai/v1/chat/completions"
#         elif provider == "HF":
#             url = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"
#         elif provider == "OPENAI":
#             url = "https://api.openai.com/v1/chat/completions"
#         else:
#             logger.warning("[llm_phraser] No LLM_REMOTE_URL configured and no sane default for provider=%s", provider)
#             return None

#     if "Authorization" not in headers:
#         logger.warning("[llm_phraser] No Authorization header set for provider=%s — remote call may be rejected", provider)

#     if lang_key == "pcm":
#         # Strong instruction for PCM + sentinel fallback
#         remote_sys_lang = (
#             "You MUST reply only in Nigerian Pidgin (pcm). Do not include English. "
#             "Use short, market-seller style. If you cannot produce the reply in Nigerian Pidgin exactly, "
#             "respond with the single token: <UNABLE_PCM>"
#         )
#     else:
#         remote_sys_lang = (
#             "You MUST reply only in English (do not include other languages). "
#             "Use short, market-seller style. If you cannot produce the reply in English exactly, respond with a single token <UNABLE_EN>"
#         )

#     def _try_post(cur_url: str) -> Optional[requests.Response]:
#         try:
#             if provider == "GROQ" or provider == "OPENAI":
#                 payload = {
#                     "model": LLM_MODEL,
#                     "messages": [
#                         {"role": "system", "content": f"You are a polite Nigerian market seller and negotiator. {remote_sys_lang}"},
#                         {"role": "user", "content": prompt}
#                     ],
#                     "max_tokens": LLM_MAX_TOKENS,
#                     "temperature": LLM_TEMPERATURE,
#                     "top_p": LLM_TOP_P
#                 }
#                 return _session.post(cur_url, json=payload, headers=headers, timeout=timeout)

#             if provider == "HF":
#                 inputs = f"{remote_sys_lang}\n\n{prompt}"
#                 payload = {"inputs": inputs, "parameters": {"temperature": LLM_TEMPERATURE, "max_new_tokens": LLM_MAX_TOKENS, "top_p": LLM_TOP_P}}
#                 return _session.post(cur_url, json=payload, headers=headers, timeout=timeout)

#             logger.warning("[llm_phraser] Unknown provider in runtime: %s", provider)
#             return None
#         except Exception as e:
#             logger.exception("[llm_phraser] _try_post exception for url=%s: %s", cur_url, e)
#             return None

#     tried_urls = []
#     candidate_urls: List[str] = [url]
#     if provider == "GROQ" or provider == "OPENAI":
#         if "chat/completions" in url:
#             candidate_urls.append(url.replace("chat/completions", "chat"))
#             candidate_urls.append(url.replace("chat/completions", "completions"))
#         if url.endswith("/chat"):
#             candidate_urls.append(url + "/completions")
#     for cur in candidate_urls:
#         if cur in tried_urls:
#             continue
#         tried_urls.append(cur)
#         logger.debug("[llm_phraser] Attempting remote LLM POST to %s", cur)
#         resp = _try_post(cur)
#         if resp is None:
#             continue
#         body_preview = (resp.text or "")[:4000]
#         logger.debug("[llm_phraser] remote status=%s preview=%s", resp.status_code, body_preview[:1000])
#         if resp.status_code >= 400:
#             logger.warning("[llm_phraser] remote returned HTTP %s for %s: %s", resp.status_code, cur, body_preview[:1000])
#             continue
#         try:
#             j = resp.json()
#         except Exception:
#             logger.exception("[llm_phraser] failed to parse JSON from remote response for %s", cur)
#             j = None
#         if j is not None:
#             txt = _extract_text_from_remote_response(j)
#             if txt:
#                 return txt.strip()
#             if isinstance(j, str):
#                 return j.strip()
#         if resp.text and len(resp.text) > 0:
#             raw = resp.text.strip()
#             if raw:
#                 return raw
#     logger.warning("[llm_phraser] remote LLM call failed after trying %d URLs", len(tried_urls))
#     return None

# # ------------------ local generation helper ------------------
# def _run_local_generation(prompt: str) -> Optional[str]:
#     global _local_ready, _tokenizer, _model
#     if not _local_ready:
#         _try_load_local_model()
#     if not _local_ready or _model is None or _tokenizer is None:
#         return None
#     try:
#         import torch
#         inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
#         gen = _model.generate(**inputs, max_new_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE, do_sample=True)
#         out = _tokenizer.decode(gen[0], skip_special_tokens=True)
#         if out.startswith(prompt):
#             out = out[len(prompt):].strip()
#         return out.strip()
#     except Exception as e:
#         logger.exception("[llm_phraser] local generation failed: %s", e)
#         return None

# # ------------------ templates (from provided TEMPLATES) ------------------
# _TEMPLATES = {
#     "en": {
#         "accept": [
#             "Deal done — ₦{price:,} for {product}. Solid choice. Ready when you are.",
#             "Price locked: ₦{price:,} for {product}. Don’t sleep on this — confirm payment.",
#             "It’s a yes! ₦{price:,} for {product}. You pay, I seal the deal.",
#             "Sweet deal: ₦{price:,} for {product}. Secure it now and thank me later.",
#             "Aunty/Uncle approved — ₦{price:,} for {product}. Complete payment to seal."
#         ],
#         "counter": [
#             "I can meet you at ₦{price:,} for {product}. Fair offer — you in?",
#             "Closest I can go is ₦{price:,} for {product}. Take it and prosper.",
#             "I stretch to ₦{price:,} for {product} — that’s my best. Oya decide.",
#             "Make it ₦{price:,} for {product} and we’re shaking hands. Shall we?",
#             "I’ll drop to ₦{price:,} for {product} but that’s the floor. Grab?"
#         ],
#         "reject": [
#             "Sorry, can’t take that — {product} needs at least ₦{price:,}. Let’s be realistic.",
#             "That offer won’t cut it. I need ₦{price:,} for {product}. Try again?",
#             "Too low, friend — ₦{price:,} is the honest price for {product}.",
#             "I can’t accept that; {product} value is ₦{price:,}. Think it over.",
#             "Not workable — lowest is ₦{price:,} for {product}. Come back if you can."
#         ],
#         "clarify": [
#             "What price are you thinking for {product}? Tell me a number.",
#             "How much are you thinking for {product}? Give me your best offer.",
#             "Help me understand — what’s your budget for {product}?",
#             "State the price you want for {product} and I’ll respond.",
#             "What’s your best offer for {product}? Speak now so we move fast."
#         ]
#     },

#     "pcm": {  # Nigerian Pidgin
#         "accept": [
#             "Na deal! ₦{price:,} for {product}. Correct choice. You go make payment now?", 
#             "Price set: ₦{price:,} for {product}. No dulling — make I receive alert nah.", 
#             "Yes o! ₦{price:,} for {product}. You pay, I package am, no dulling.", 
#             "Gbedu done: ₦{price:,} for {product}. Oya send money make I arrange am for you sharp sharp.", 
#             "I dey with you — ₦{price:,} for {product}. Make you pay to lock."
#         ],
#         "counter": [
#             "I fit give ₦{price:,} for {product}. E balance — you ready?",
#             "Closest I go down na ₦{price:,} for {product}. Oya take am.",
#             "I go drop to ₦{price:,} for {product}, that one be my last.",
#             "Make we meet for ₦{price:,} for {product}. Na fair play.",
#             "I fit accept ₦{price:,} for {product}, na small small we dey do market." 
#         ],
#         "reject": [
#             "Haba, dat one too small — {product} need ₦{price:,}. You sef think am nah.", 
#             "Ah my oga go sack me if I accept that price — ₦{price:,} na minimum for {product}.", 
#             "That offer no correct. I need ₦{price:,} for {product}.",
#             "E no go work — lowest na ₦{price:,} for {product}.",
#             "Abeg, dat price no reach. ₦{price:,} be the real one."
#         ],
#         "clarify": [
#             "Which price you dey reason for {product}? Tell me now.",
#             "How much you fit pay for {product}? Give figure.",
#             "Wetin be your budget for {product}? Make we yan.",
#             "Give your best price for {product} so we fit move.",
#             "Abeg state the price you want for {product} — quick answer."
#         ]
#     },

#     "yo": {  # Yoruba
#         "accept": [
#             "Ẹ̀wẹ̀ — ₦{price:,} fún {product}. Oun tí ó dá. Tẹ̀ síwájú kí o san.",
#             "Iye ti a tẹ̀sí: ₦{price:,} fún {product}. Má ṣe fà áyá — ra bayìí.",
#             "Ó dá! ₦{price:,} fún {product}. San, a máa ranṣẹ́.",
#             "Ìpinnu wà: ₦{price:,} fún {product}. Gba kíákíá.",
#             "Mo fọwọ́sowọ́ pọ̀ — ₦{price:,} fún {product}. San láti fi dékun."
#         ],
#         "counter": [
#             "Mo lè gba ₦{price:,} fún {product}. Ṣe o nífẹ̀ẹ́?",
#             "Ìwọ̀n tó sunmọ̀ jẹ́ ₦{price:,} fún {product}. Oya, gba.",
#             "Mo dínkù sí ₦{price:,} fún {product}, ìyẹn ni ìpinnu mi.",
#             "Ṣe ₦{price:,} fún {product} kí a lè pari ìsọ̀kan.",
#             "Èmi yóò gba ₦{price:,} fún {product} — ìyẹn ni àfojúsùn mi."
#         ],
#         "reject": [
#             "Rárá, kò tó — {product} tọ́ ₦{price:,}. Jọwọ ròyìn.",
#             "Ibere ko wulo. Mo nilo ₦{price:,} fún {product}.",
#             "Kò ṣeé gba — iye kéré. ₦{price:,} ni otitọ.",
#             "Ẹ jọ̀, ìfilọ́lẹ̀ yìí kì í dé. ₦{price:,} ni mo lè gba.",
#             "Ó kéré ju — ìwọ lè tún rò, ₦{price:,} ni ìwòye mi."
#         ],
#         "clarify": [
#             "Ṣé mélòó ni o rò fún {product}? Sọ fún mi.",
#             "Kí ni ìsúná rẹ fún {product}? Jẹ́ kó ye mi.",
#             "Ẹ sọ iye tí o lè san fún {product}, má fi dúró.",
#             "Ṣe o lè fi owó kan ṣàpèjúwe? ₦… fún {product}?",
#             "Jọwọ, sọ iye rẹ fún {product} kí a lè bá a ṣèrànwọ́."
#         ]
#     },

#     "ha": {  # Hausa
#         "accept": [
#             "Nagode — ₦{price:,} don {product}. Kyakkyawan zabi. Sai ka biya.",
#             "Farashi an kulle: ₦{price:,} don {product}. Ka tabbatar da biyan kuɗi.",
#             "Eh, mun amince! ₦{price:,} don {product}. Ka biya, mu shirya.",
#             "Yau dai: ₦{price:,} don {product}. Kar ka bari ya wuce.",
#             "Na amince — ₦{price:,} don {product}. Biya yanzu ka tabbatar."
#         ],
#         "counter": [
#             "Zan iya ₦{price:,} don {product}. Wannan mafi kusa ne.",
#             "Mafi kusa da ni: ₦{price:,} don {product}. Ka yanke shawara.",
#             "Zan sauƙaƙa zuwa ₦{price:,} don {product}, wannan ne iyaka.",
#             "Ka kawo ₦{price:,} don {product} sai mu gama.",
#             "Ina iya bada ₦{price:,} don {product}, amintacce ne."
#         ],
#         "reject": [
#             "Yi haƙuri, wannan ƙasa ne — {product} na ₦{price:,}.",
#             "Ba zan iya karɓa ba. Ina bukatar ₦{price:,} don {product}.",
#             "Farashin bai isa ba. ₦{price:,} shine gaskiya.",
#             "Ba zai yiwu ba — mafi ƙasa shine ₦{price:,} don {product}.",
#             "Kayi haƙuri, wannan tayi ba zata yi ba. ₦{price:,} ne."
#         ],
#         "clarify": [
#             "Wane farashi kake tunani don {product}? Fada min.",
#             "Menene kasafin kuɗin ku don {product}? Ka gaya min.",
#             "Don Allah fa, mene ne adadin da zaka iya biya?",
#             "Bayyana farashin da kake so don {product}, mu tattauna.",
#             "Fada min adadin da zaka iya biya don {product} yanzu."
#         ]
#     },

#     "ig": {  # Igbo
#         "accept": [
#             "Daalụ — ₦{price:,} maka {product}. Nhọrọ ọma. Banye kwụọ.",
#             "E jidere ọnụahịa: ₦{price:,} maka {product}. Biko kwụọ ụgwọ.",
#             "Ọ dị mma! ₦{price:,} maka {product}. Kwụọ, anyị ga-eziga.",
#             "Emeela: ₦{price:,} maka {product}. Nwee obi ike, zụta ugbu a.",
#             "Ana m akwado — ₦{price:,} maka {product}. Kwụọ ka e mechaa."
#         ],
#         "counter": [
#             "Enwere m ike ime ₦{price:,} maka {product}. Nke a kacha nso.",
#             "Ihe kacha m nwee bụ ₦{price:,} maka {product}. Ị nọ n’aka?",
#             "M ga-ebelata ruo ₦{price:,} maka {product}, nke a bụ ikpeazụ.",
#             "Mee ka ọ bụrụ ₦{price:,} maka {product} ka anyị kwụsịtụ.",
#             "Anọ m na ₦{price:,} maka {product} — ọ bụ ezi onyinye."
#         ],
#         "reject": [
#             "Ndo, nke ahụ adịghị eru — {product} dị ₦{price:,}.",
#             "Agaghị m anabata nke ahụ. Achọrọ m ₦{price:,} maka {product}.",
#             "Ego ahụ dị ala. ₦{price:,} bụ ezi ọnụahịa.",
#             "O nweghị ụzọ — ala kacha nta bụ ₦{price:,} maka {product}.",
#             "Biko tụlee ọzọ, ọnụahịa kwesịrị ịbụ ₦{price:,}."
#         ],
#         "clarify": [
#             "Kedụ ọnụahịa ị na-eche maka {product}? Gwa m.",
#             "Kedu ego ị nwere maka {product}? Kọwaa.",
#             "Biko, tinyekwuo ego ị ga-akwụ maka {product}.",
#             "Gosi ọnụahịa kacha mma gị maka {product}. Ka anyị kwuo.",
#             "Kedụ ọnụahịa ị na-enye maka {product}? Zaghachi ngwa ngwa."
#         ]
#     }
# }

# FEW_SHOT_PROMPT = """
# SYSTEM: You are a lively Nigerian market seller, a sharp negotiator bursting with authentic swagger, fluent in English (en), Pidgin (pcm), Yoruba (yo), Hausa (ha), and Igbo (ig). Speak like a native with bold market energy, using {lang_key} tone (short, punchy, dripping with Nigerian charm and banter).

# LANGUAGE RULE (mandatory):
#  - Reply ONLY in the language the user uses in their message (English, Pidgin, Yoruba, Igbo, or Hausa).
#  - If the user mixes languages, detect the dominant language and reply in that dominant language. You MAY mirror short mixed phrases sparingly **only if the user mixed**. Do NOT introduce mixed-language content otherwise.
#  - If the user’s language is not detectable or unsupported, default to English with Nigerian seller flair.

# GOALS:
#  - Keep replies short: 1–2 sentences, maximum 40 words.
#  - Be persuasive using warmth, light humour, and culturally natural banter — but never rude or abusive.
#  - Protect seller margin: be firm but polite; when refusing an offer, always offer a forward path (a counter or next step).
#  - Vary phrasing: rotate among templates and CTAs so responses don’t repeat the exact same line across interactions.
  
# PRICING & NUMBERS (strict):
#  - Always use the currency symbol ₦.
#  - If {final_price} is provided, **echo the numeric price exactly as given** (do not reformat or invent decimals). Use {final_price} verbatim where required.
#  - If {final_price} is NOT provided, format the price as a whole-naira integer with comma separators (e.g., 1,250) when rendering templates and replies.
#  - Always include the numeric price and the product name somewhere in the reply.
#  - Numeric reinforcement: you MAY repeat the numeric price once for emphasis (e.g., “₦15,000 — yes, ₦15,000”), but keep the reply within length limits.

# NEGOTIATION BEHAVIOR:
#  - ACCEPT: use an acceptance template, express brief gratitude/energy, echo the price (prefer {final_price} when provided), include CTA to pay/confirm.
#  - COUNTER: propose a single clear counter price (one number only), explain briefly if needed, end with CTA.
#  - REJECT: politely refuse, state the honest minimum (prefer using {final_price} when available or the computed minimum), then immediately present a path (e.g., “I can do ₦X” or “Come back if you can reach ₦X”).
#  - CLARIFY: ask for a numeric offer or budget in the user’s language; be short and direct.

# STYLE RULES:
#  - Tone by language: 
#    - English = savvy, confident, with market swagger.
#    - Pidgin = playful, street-smart, chop-life vibe.
#    - Yoruba = warm, respectful, with local proverbs.
#    - Hausa = polite, direct, with community warmth.
#    - Igbo = friendly, persuasive, with trader’s charm.
#  - Keep replies punchy, avoid long explanations. Use 1–2 short clauses.
#  - Do not invent shipping, freebies, discounts, or timelines unless specified.
#  - Mirror short mixed-language phrases only if user mixes; reply in dominant language.

# CONTEXT: {context}

# NUMERIC_GROUNDS:
#  final_price: {final_price}

# OPERATIONS EXAMPLES (behavioural):
#  - If user accepts and final_price is 15000 → reply with acceptance phrase echoing ₦15000 and a CTA.
#  - If user counters with an offer → reply with one-number counter (₦X), short rationale optional, end with CTA.
#  - If user offers too low → politely state the computed minimum or {final_price} (if provided) and offer a compromise or next step.

# IMPLEMENTATION NOTES:
#  - Rotate templates so the same template is not reused back-to-back for the same user.
#  - Keep replies short enough for chat UI bubbles; prioritize clarity and action.
#  - If unsure about language detection, default to English with Nigerian market persona and ask for clarification only once, concisely.

# Final instruction: Use this prompt plus the TEMPLATES for {lang_key} as your core. Always be authentic, persuasive, and business-savvy — like a real seller from the market who knows value and respects the customer.
# """

# # ------------------ policy (refined) ------------------
# def compute_counter_price(base_price: int, offer: Optional[int], min_price: Optional[int] = None) -> Tuple[str, Optional[int]]:
#     if offer is None:
#         return "ASK_CLARIFY", None
#     try:
#         base = int(base_price)
#         off = int(offer)
#     except Exception:
#         return "ASK_CLARIFY", None
#     if base <= 0:
#         return "ASK_CLARIFY", None

#     computed_min = int(round(base * DEFAULT_MIN_PRICE_RATIO))
#     if min_price is not None:
#         try:
#             min_eff = max(int(min_price), computed_min)
#         except Exception:
#             min_eff = computed_min
#     else:
#         min_eff = computed_min
#     min_eff = min(min_eff, base)

#     buyer_pct_of_base = off / base if base > 0 else 0.0
#     buyer_pct_of_min = off / min_eff if min_eff > 0 else 0.0

#     if buyer_pct_of_base >= 0.90:
#         return "ACCEPT", off

#     def make_prop(pct_low: float, pct_high: float, bias_toward_buyer: bool = False) -> int:
#         pct = random.uniform(pct_low, pct_high)
#         prop = int(round(base * pct))
#         if bias_toward_buyer and off is not None:
#             prop = int(round((prop + off) / 2.0))
#         if prop < min_eff:
#             prop = min_eff
#         prop = min(prop, base)
#         return int(prop)

#     if buyer_pct_of_min >= 0.80:
#         prop = make_prop(0.60, 0.75, bias_toward_buyer=True)
#         action = "COUNTER" if prop != off else "ACCEPT"
#         return action, prop

#     if 0.50 <= buyer_pct_of_min < 0.80:
#         prop = make_prop(0.60, 0.75, bias_toward_buyer=True)
#         return "COUNTER", prop

#     prop = make_prop(0.70, 0.80, bias_toward_buyer=False)
#     if prop == min_eff and prop < base:
#         try:
#             prop = min_eff + 1
#             if prop > base:
#                 prop = min_eff
#         except Exception:
#             pass
#     return "REJECT", prop

# # ------------------ improved numeric matcher ------------------
# def _reply_contains_price(reply: str, price: int) -> bool:
#     if not reply or price is None:
#         return False
#     cleaned = re.sub(r'[₦$€£]', '', reply)
#     tokens = re.findall(r'\d[\d,]*', cleaned)
#     for t in tokens:
#         try:
#             val = int(t.replace(",", ""))
#             if val == int(price):
#                 return True
#         except Exception:
#             continue
#     return False

# # ------------------ helpers for dynamic negotiation ------------------
# def _compute_floor(min_price: Optional[int], base_price: int) -> int:
#     try:
#         if min_price is not None and int(min_price) > 0:
#             mp = int(min_price)
#         else:
#             mp = int(round(base_price * DEFAULT_MIN_PRICE_RATIO))
#         floor = int(math.ceil(mp * 1.10))
#         return max(floor, 1)
#     except Exception:
#         return max(int(round(base_price * DEFAULT_MIN_PRICE_RATIO * 1.10)), 1)

# def _initial_dynamic_counter(buyer_offer: Optional[int], min_price: int, base_price: int) -> int:
#     dyn = _dynamic_counter_price(buyer_offer, min_price, base_price)
#     floor = _compute_floor(min_price, base_price)
#     return max(dyn, floor)

# def _next_proposal_after_reject(prev_proposals: List[int], buyer_offer: Optional[int], min_price: int, base_price: int) -> int:
#     floor = _compute_floor(min_price, base_price)
#     try:
#         if not prev_proposals:
#             return _initial_dynamic_counter(buyer_offer, min_price, base_price)
#         last = int(prev_proposals[-1])
#         if last <= floor:
#             return floor
#         gap = max(last - floor, 0)
#         step1 = int(math.ceil(gap * 0.40))
#         step2 = int(math.ceil(base_price * 0.03))
#         step = max(step1, step2, 1)
#         next_prop = last - step
#         next_prop = max(next_prop, floor)
#         return int(next_prop)
#     except Exception as e:
#         logger.exception("[llm_phraser] _next_proposal_after_reject error: %s", e)
#         return floor

# def _dynamic_counter_price(buyer_offer: Optional[int], min_price: int, base_price: int) -> int:
#     try:
#         mp = int(min_price) if min_price is not None else int(round(base_price * DEFAULT_MIN_PRICE_RATIO))
#         bp = int(base_price)
#         if mp <= 0:
#             mp = int(round(bp * DEFAULT_MIN_PRICE_RATIO))
#         candidate75 = max(int(round(bp * 0.75)), mp)
#         candidate60 = max(int(round(bp * 0.60)), mp)
#         if buyer_offer is None:
#             return candidate75
#         bo = int(buyer_offer)
#         closeness = (bo / mp) if mp > 0 else (bo / bp if bp > 0 else 0.0)
#         if closeness >= 0.8:
#             mid = int(round((bo + candidate75) / 2.0))
#             return max(mid, mp)
#         if 0.5 <= closeness < 0.8:
#             mid = int(round((candidate75 + bo) / 2.0))
#             return max(min(mid, candidate75), mp)
#         return max(candidate75, mp)
#     except Exception as e:
#         logger.exception("[llm_phraser] _dynamic_counter_price error: %s", e)
#         try:
#             fallback = int(round(base_price * 0.75))
#             return max(fallback, int(min_price or 0))
#         except Exception:
#             return int(min_price or base_price)

# def _format_naira(n: Optional[int]) -> str:
#     try:
#         return f"₦{int(n):,}"
#     except Exception:
#         return f"₦{n}"

# # ------------------ template rendering helper ------------------
# def _choose_template_variant(candidate: Any, ratio: Optional[float]) -> str:
#     if isinstance(candidate, list):
#         if not candidate:
#             return ""
#         if ratio is None:
#             return random.choice(candidate)
#         try:
#             if ratio >= 0.8:
#                 idx = 1 if len(candidate) > 1 else 0
#             elif ratio >= 0.5:
#                 idx = 0
#             else:
#                 idx = 0
#             return candidate[idx]
#         except Exception:
#             return random.choice(candidate)
#     return str(candidate)

# def _render_template_reply(template_map: Dict[str, Any], action_key: str, price: Optional[int], product_name: str, ratio: Optional[float] = None, customer: Optional[str] = None) -> str:
#     try:
#         candidate = template_map.get(action_key, template_map.get("counter"))
#         tmpl = _choose_template_variant(candidate, ratio)
#         if not isinstance(tmpl, str):
#             tmpl = str(tmpl)
#         tpl_price_int = None
#         try:
#             tpl_price_int = int(price) if price is not None else 0
#         except Exception:
#             tpl_price_int = 0
#         cust = customer or "friend"
#         # allow templates to optionally include {customer}
#         return tmpl.format(price=tpl_price_int, product=product_name, customer=cust)
#     except Exception as e:
#         logger.exception("[llm_phraser] template rendering failed: %s", e)
#         try:
#             return f"Our counter price is {_format_naira(price)} for {product_name}."
#         except Exception:
#             return f"Our counter price is ₦{price} for {product_name}."

# # ------------------ language normalization helper ------------------
# def _normalize_lang_code(s: Optional[str]) -> str:
#     if not s:
#         return "en"
#     s0 = str(s).strip().lower()
#     if s0 in ("en", "eng", "english", "en_us", "en-gb", "en-us"):
#         return "en"
#     if s0 in ("pcm", "pidgin", "pidgin_ng", "pcm_ng", "pcm-nigeria", "pidgin-nigeria"):
#         return "pcm"
#     if s0.startswith("yo") or s0 in ("yoruba", "yoruba_ng", "yo_ng"):
#         return "yo"
#     if s0.startswith("ha") or s0 in ("hausa", "hausa_ng", "ha_ng"):
#         return "ha"
#     if s0.startswith("ig") or s0 in ("igbo", "igbo_ng", "ig_ng"):
#         return "ig"
#     if s0 and s0[0] in ("p", "y", "h", "i"):
#         if s0[0] == "p":
#             return "pcm"
#         if s0[0] == "y":
#             return "yo"
#         if s0[0] == "h":
#             return "ha"
#         if s0[0] == "i":
#             return "ig"
#     return "en"

# # ------------------ main phrase() ------------------
# def phrase(decision: Dict[str, Any], product: Dict[str, Any], lang: str = "en", context: Optional[str] = None, use_remote_expected: Optional[bool] = None, **kwargs) -> str:
#     """
#     decision: dict possibly containing 'action', 'price', 'offer', 'meta'
#     product: dict with 'name' and 'base_price'
#     lang: language key (en|pcm|yo|ig|ha) or variant
#     use_remote_expected: optional compatibility flag (ignored by this implementation)
#     Returns a user-facing string reply.
#     """
#     # Normalize incoming language codes
#     lang_key = _normalize_lang_code(lang)

#     prod_name = product.get("name") or product.get("id") or "product"
#     base_price = int(product.get("base_price", 0))

#     # optional customer name from kwargs
#     customer_name = None
#     if "customer_name" in kwargs:
#         customer_name = kwargs.get("customer_name")
#     elif "customer" in kwargs:
#         customer_name = kwargs.get("customer")

#     # read decision fields
#     explicit_action = (decision.get("action") or "").upper() or None
#     explicit_price = decision.get("price")
#     buyer_offer = None
#     if decision.get("offer") is not None:
#         try:
#             buyer_offer = int(decision.get("offer"))
#         except Exception:
#             buyer_offer = None

#     # negotiation meta (may carry min_price and previous proposals)
#     meta = decision.get("meta") or {}
#     min_price_meta = None
#     try:
#         if isinstance(meta, dict) and "min_price" in meta:
#             min_price_meta = int(meta["min_price"])
#     except Exception:
#         min_price_meta = None

#     prev_proposals: List[int] = []
#     try:
#         if isinstance(meta, dict) and "prev_proposals" in meta and isinstance(meta["prev_proposals"], list):
#             prev_proposals = [int(x) for x in meta["prev_proposals"] if isinstance(x, (int, str)) or hasattr(x, "__int__")]
#     except Exception:
#         prev_proposals = []

#     floor = _compute_floor(min_price_meta, base_price)

#     # If the decision is ESCALATE (from guard), compute a dynamic counter price >= floor
#     if explicit_action == "ESCALATE":
#         if min_price_meta is not None:
#             if prev_proposals:
#                 dyn_price = _next_proposal_after_reject(prev_proposals, buyer_offer, min_price_meta, base_price)
#             else:
#                 dyn_price = _initial_dynamic_counter(buyer_offer, min_price_meta, base_price)
#             dyn_price = max(dyn_price, floor)
#             logger.info("[llm_phraser] ESCALATE -> dyn_price=%s (floor=%s) prev_proposals=%s buyer_offer=%s", dyn_price, floor, prev_proposals, buyer_offer)
#             explicit_action = "COUNTER"
#             explicit_price = dyn_price
#             meta = dict(meta or {})
#             meta["next_proposal"] = dyn_price
#             meta["floor"] = floor
#             meta.setdefault("prev_proposals", prev_proposals)
#         else:
#             explicit_action = "REJECT"
#             explicit_price = explicit_price if explicit_price is not None else (buyer_offer or base_price)

#     if explicit_action:
#         computed_action = explicit_action
#         computed_price = explicit_price if explicit_price is not None else (buyer_offer or base_price)
#     else:
#         computed_action, computed_price = compute_counter_price(base_price, buyer_offer)

#     if min_price_meta is not None and computed_price is not None:
#         computed_price = max(int(computed_price), floor)

#     final_price = int(computed_price) if computed_price is not None else None

#     # sanitize context for prompts
#     sanitized_ctx = _sanitize_context(context)
#     fs = FEW_SHOT_PROMPT.format(lang_key=lang_key, context=sanitized_ctx, final_price=final_price)

#     input_block = (
#         f"\nINPUT:\nproduct_name: \"{prod_name}\"\n"
#         f"base_price: {base_price}\n"
#         f"offer: {buyer_offer if buyer_offer is not None else 'null'}\n"
#         f"counter_price: {final_price if final_price is not None else 'null'}\n"
#         f"decision: {computed_action}\n"
#     )
#     instruction = "\nINSTRUCTION:\nReply in one or two short sentences that are friendly, respectful, persuasive and end with a clear next step (CTA). Match the numeric values shown above exactly. Keep replies short and culturally appropriate."
#     prompt = "\n".join(["SYSTEM PROMPT (few-shot examples):", fs, input_block, instruction])

#     logger.debug("[llm_phraser] phrase() computed_action=%s final_price=%s prod=%s lang=%s floor=%s prev_proposals=%s",
#                  computed_action, final_price, prod_name, lang_key, floor, prev_proposals)

#     # Compute ratio for tone selection when templates are used
#     min_price_for_ratio = None
#     try:
#         if min_price_meta is not None:
#             min_price_for_ratio = int(min_price_meta)
#         else:
#             min_price_for_ratio = int(round(base_price * DEFAULT_MIN_PRICE_RATIO))
#     except Exception:
#         min_price_for_ratio = int(round(base_price * DEFAULT_MIN_PRICE_RATIO))

#     ratio = None
#     try:
#         if min_price_for_ratio > 0 and buyer_offer is not None:
#             ratio = float(buyer_offer) / float(min_price_for_ratio)
#     except Exception:
#         ratio = None

#     # --- LANGUAGE-BASED TEMPLATE OVERRIDE ---
#     # STRICT: only Yoruba, Hausa and Igbo MUST use templates (no remote/local)
#     TEMPLATE_ONLY_LANGS = {"yo", "ha", "ig"}

#     # If lang_key not already in template-only set, use context heuristic to detect
#     if lang_key not in TEMPLATE_ONLY_LANGS:
#         detected_from_ctx = _detect_local_language_from_context(sanitized_ctx)
#         if detected_from_ctx in TEMPLATE_ONLY_LANGS:
#             logger.info("[llm_phraser] overriding lang_key -> %s based on context heuristic", detected_from_ctx)
#             lang_key = detected_from_ctx

#     if lang_key in TEMPLATE_ONLY_LANGS:
#         template_map = _TEMPLATES.get(lang_key, _TEMPLATES["en"])
#         action_map = {"ACCEPT": "accept", "REJECT": "reject", "COUNTER": "counter", "ASK_CLARIFY": "clarify"}
#         key = action_map.get(computed_action, "counter")
#         try:
#             tpl_price = final_price if final_price is not None else base_price
#             rendered = _render_template_reply(template_map, key, tpl_price, prod_name, ratio, customer_name)
#             try:
#                 meta_out = dict(meta or {})
#                 prev = meta_out.get("prev_proposals", []) or []
#                 if computed_action == "COUNTER":
#                     prev = list(prev) + [int(tpl_price)]
#                     meta_out["prev_proposals"] = prev
#                 meta_out["floor"] = floor
#             except Exception:
#                 meta_out = meta
#             logger.info("[llm_phraser] template_override=true lang=%s action=%s price=%s prod=%s ratio=%s meta_prev=%s",
#                         lang_key, computed_action, tpl_price, prod_name, ratio, meta_out.get("prev_proposals") if isinstance(meta_out, dict) else None)
#             return rendered
#         except Exception:
#             logger.exception("[llm_phraser] template override failed for lang=%s key=%s — falling through", lang_key, key)
#             # safe fallback to English template to avoid crash
#             return _render_template_reply(_TEMPLATES.get("en"), "counter", final_price or base_price, prod_name, ratio, customer_name)

#     # --- REMOTE preferred (English or pidgin) ---
#     if LLM_MODE == "REMOTE":
#         out = None
#         try:
#             remote_lang_forcing = "en" if lang_key not in ("pcm",) else "pcm"
#             out = _call_remote_llm(prompt, lang_key=remote_lang_forcing)
#             if out:
#                 # If PCM mode: handle sentinel <UNABLE_PCM> -> fallback to templates
#                 if remote_lang_forcing == "pcm" and out.strip() == "<UNABLE_PCM>":
#                     logger.info("[llm_phraser] remote signalled UNABLE_PCM; falling back to templates")
#                 else:
#                     # numeric safety: must contain final_price when final_price provided
#                     if final_price is None or _reply_contains_price(out, final_price):
#                         logger.info("[llm_phraser] remote_generation_ok lang=%s preview=%s", lang_key, (out[:120] + "...") if len(out) > 120 else out)
#                         return out.strip()
#                     else:
#                         logger.warning("[llm_phraser] remote returned no usable text or numeric mismatch; falling back")
#             else:
#                 logger.warning("[llm_phraser] remote returned empty response; falling back")
#         except Exception:
#             logger.exception("[llm_phraser] remote generation error")

#     # --- LOCAL fallback ---
#     if LLM_MODE == "LOCAL":
#         try:
#             out = _run_local_generation(prompt)
#             if out and (final_price is None or _reply_contains_price(out, final_price)):
#                 logger.info("[llm_phraser] local_generation_ok lang=%s preview=%s", lang_key, (out[:120] + "...") if len(out) > 120 else out)
#                 return out.strip()
#         except Exception:
#             logger.exception("[llm_phraser] local generation error")

#     # --- TEMPLATE fallback (final) ---
#     template_map = _TEMPLATES.get(lang_key, _TEMPLATES["en"])
#     action_map = {"ACCEPT": "accept", "REJECT": "reject", "COUNTER": "counter", "ASK_CLARIFY": "clarify"}
#     key = action_map.get(computed_action, "counter")
#     try:
#         tpl_price = final_price if final_price is not None else base_price
#         logger.info("[llm_phraser] final_template lang=%s action=%s price=%s prod=%s prev_proposals=%s floor=%s customer_name=%s",
#                     lang_key, computed_action, tpl_price, prod_name, prev_proposals, floor, customer_name)
#         return _render_template_reply(template_map, key, tpl_price, prod_name, ratio, customer_name=customer_name, lang_key=lang_key)
#     except Exception:
#         try:
#             if customer_name:
#                 return f"{customer_name}, Our counter price is {_format_naira(final_price)} for {prod_name}."
#             return f"Our counter price is {_format_naira(final_price)} for {prod_name}."
#         except Exception:
#             return f"Our counter price is ₦{final_price} for {prod_name}."


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
    "[llm_phraser] startup keys: GROQ=%s LLM_API=%s OPENAI=%s HF=%s DEFAULT_MIN_PRICE_RATIO=%s",
    _mask_key(GROQ_API_KEY), _mask_key(LLM_API_KEY), _mask_key(OPENAI_API_KEY), _mask_key(HF_TOKEN), DEFAULT_MIN_PRICE_RATIO
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
    return the language code 'yo'|'ha'|'ig'. Conservative: require >=1 strong hit (we keep it light).
    """
    if not ctx:
        return None
    s = ctx.lower()
    # tokenize
    tokens = set(_CTX_WORD_RE.findall(s))
    for lang, markers in _CTX_MARKERS.items():
        # count matches
        matches = sum(1 for m in markers if m and (m in tokens or (" " + m + " ") in (" " + s + " ")))
        if matches >= 1:
            logger.debug("[llm_phraser] context heuristic matched lang=%s matches=%s tokens=%s", lang, matches, list(tokens)[:10])
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
    If lang_key == 'pcm' the system prompt instructs the remote model to reply with
    the single token '<UNABLE_PCM>' if it cannot respond in Pidgin; caller will fallback.
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
        # Strong instruction for PCM + sentinel fallback
        remote_sys_lang = (
            "You MUST reply only in Nigerian Pidgin (pcm). Do not include English. "
            "Use short, market-seller style. If you cannot produce the reply in Nigerian Pidgin exactly, "
            "respond with the single token: <UNABLE_PCM>"
        )
    else:
        remote_sys_lang = (
            "You MUST reply only in English (do not include other languages). "
            "Use short, market-seller style. If you cannot produce the reply in English exactly, respond with a single token <UNABLE_EN>"
        )

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
            txt = _extract_text_from_remote_response(j)
            if txt:
                return txt.strip()
            if isinstance(j, str):
                return j.strip()
        if resp.text and len(resp.text) > 0:
            raw = resp.text.strip()
            if raw:
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

# ------------------ templates (from provided TEMPLATES) ------------------
_TEMPLATES = {
    "en": {
        "accept": [
            "Deal done — ₦{price:,} for {product}. Solid choice. Ready when you are.",
            "Price locked: ₦{price:,} for {product}. Don’t sleep on this — confirm payment.",
            "It’s a yes! ₦{price:,} for {product}. You pay, I seal the deal.",
            "Sweet deal: ₦{price:,} for {product}. Secure it now and thank me later.",
            "Aunty/Uncle approved — ₦{price:,} for {product}. Complete payment to seal."
        ],
        "counter": [
            "I can meet you at ₦{price:,} for {product}. Fair offer — you in?",
            "Closest I can go is ₦{price:,} for {product}. Take it and prosper.",
            "I stretch to ₦{price:,} for {product} — that’s my best. Oya decide.",
            "Make it ₦{price:,} for {product} and we’re shaking hands. Shall we?",
            "I’ll drop to ₦{price:,} for {product} but that’s the floor. Grab?"
        ],
        "reject": [
            "Sorry, can’t take that — {product} needs at least ₦{price:,}. Let’s be realistic.",
            "That offer won’t cut it. I need ₦{price:,} for {product}. Try again?",
            "Too low, friend — ₦{price:,} is the honest price for {product}.",
            "I can’t accept that; {product} value is ₦{price:,}. Think it over.",
            "Not workable — lowest is ₦{price:,} for {product}. Come back if you can."
        ],
        "clarify": [
            "What price are you thinking for {product}? Tell me a number.",
            "How much are you thinking for {product}? Give me your best offer.",
            "Help me understand — what’s your budget for {product}?",
            "State the price you want for {product} and I’ll respond.",
            "What’s your best offer for {product}? Speak now so we move fast."
        ]
    },

    "pcm": {  # Nigerian Pidgin
        "accept": [
            "Na deal! ₦{price:,} for {product}. Correct choice. You go make payment now?", 
            "Price set: ₦{price:,} for {product}. No dulling — make I receive alert nah.", 
            "Yes o! ₦{price:,} for {product}. You pay, I package am, no dulling.", 
            "Gbedu done: ₦{price:,} for {product}. Oya send money make I arrange am for you sharp sharp.", 
            "I dey with you — ₦{price:,} for {product}. Make you pay to lock."
        ],
        "counter": [
            "I fit give ₦{price:,} for {product}. E balance — you ready?",
            "Closest I go down na ₦{price:,} for {product}. Oya take am.",
            "I go drop to ₦{price:,} for {product}, that one be my last.",
            "Make we meet for ₦{price:,} for {product}. Na fair play.",
            "I fit accept ₦{price:,} for {product}, na small small we dey do market." 
        ],
        "reject": [
            "Haba, dat one too small — {product} need ₦{price:,}. You sef think am nah.", 
            "Ah my oga go sack me if I accept that price — ₦{price:,} na minimum for {product}.", 
            "That offer no correct. I need ₦{price:,} for {product}.",
            "E no go work — lowest na ₦{price:,} for {product}.",
            "Abeg, dat price no reach. ₦{price:,} be the real one."
        ],
        "clarify": [
            "Which price you dey reason for {product}? Tell me now.",
            "How much you fit pay for {product}? Give figure.",
            "Wetin be your budget for {product}? Make we yan.",
            "Give your best price for {product} so we fit move.",
            "Abeg state the price you want for {product} — quick answer."
        ]
    },

    "yo": {  # Yoruba
        "accept": [
            "Ẹ̀wẹ̀ — ₦{price:,} fún {product}. Oun tí ó dá. Tẹ̀ síwájú kí o san.",
            "Iye ti a tẹ̀sí: ₦{price:,} fún {product}. Má ṣe fà áyá — ra bayìí.",
            "Ó dá! ₦{price:,} fún {product}. San, a máa ranṣẹ́.",
            "Ìpinnu wà: ₦{price:,} fún {product}. Gba kíákíá.",
            "Mo fọwọ́sowọ́ pọ̀ — ₦{price:,} fún {product}. San láti fi dékun."
        ],
        "counter": [
            "Mo lè gba ₦{price:,} fún {product}. Ṣe o nífẹ̀ẹ́?",
            "Ìwọ̀n tó sunmọ̀ jẹ́ ₦{price:,} fún {product}. Oya, gba.",
            "Mo dínkù sí ₦{price:,} fún {product}, ìyẹn ni ìpinnu mi.",
            "Ṣe ₦{price:,} fún {product} kí a lè pari ìsọ̀kan.",
            "Èmi yóò gba ₦{price:,} fún {product} — ìyẹn ni àfojúsùn mi."
        ],
        "reject": [
            "Rárá, kò tó — {product} tọ́ ₦{price:,}. Jọwọ ròyìn.",
            "Ibere ko wulo. Mo nilo ₦{price:,} fún {product}.",
            "Kò ṣeé gba — iye kéré. ₦{price:,} ni otitọ.",
            "Ẹ jọ̀, ìfilọ́lẹ̀ yìí kì í dé. ₦{price:,} ni mo lè gba.",
            "Ó kéré ju — ìwọ lè tún rò, ₦{price:,} ni ìwòye mi."
        ],
        "clarify": [
            "Ṣé mélòó ni o rò fún {product}? Sọ fún mi.",
            "Kí ni ìsúná rẹ fún {product}? Jẹ́ kó ye mi.",
            "Ẹ sọ iye tí o lè san fún {product}, má fi dúró.",
            "Ṣe o lè fi owó kan ṣàpèjúwe? ₦… fún {product}?",
            "Jọwọ, sọ iye rẹ fún {product} kí a lè bá a ṣèrànwọ́."
        ]
    },

    "ha": {  # Hausa
        "accept": [
            "Nagode — ₦{price:,} don {product}. Kyakkyawan zabi. Sai ka biya.",
            "Farashi an kulle: ₦{price:,} don {product}. Ka tabbatar da biyan kuɗi.",
            "Eh, mun amince! ₦{price:,} don {product}. Ka biya, mu shirya.",
            "Yau dai: ₦{price:,} don {product}. Kar ka bari ya wuce.",
            "Na amince — ₦{price:,} don {product}. Biya yanzu ka tabbatar."
        ],
        "counter": [
            "Zan iya ₦{price:,} don {product}. Wannan mafi kusa ne.",
            "Mafi kusa da ni: ₦{price:,} don {product}. Ka yanke shawara.",
            "Zan sauƙaƙa zuwa ₦{price:,} don {product}, wannan ne iyaka.",
            "Ka kawo ₦{price:,} don {product} sai mu gama.",
            "Ina iya bada ₦{price:,} don {product}, amintacce ne."
        ],
        "reject": [
            "Yi haƙuri, wannan ƙasa ne — {product} na ₦{price:,}.",
            "Ba zan iya karɓa ba. Ina bukatar ₦{price:,} don {product}.",
            "Farashin bai isa ba. ₦{price:,} shine gaskiya.",
            "Ba zai yiwu ba — mafi ƙasa shine ₦{price:,} don {product}.",
            "Kayi haƙuri, wannan tayi ba zata yi ba. ₦{price:,} ne."
        ],
        "clarify": [
            "Wane farashi kake tunani don {product}? Fada min.",
            "Menene kasafin kuɗin ku don {product}? Ka gaya min.",
            "Don Allah fa, mene ne adadin da zaka iya biya?",
            "Bayyana farashin da kake so don {product}, mu tattauna.",
            "Fada min adadin da zaka iya biya don {product} yanzu."
        ]
    },

    "ig": {  # Igbo
        "accept": [
            "Daalụ — ₦{price:,} maka {product}. Nhọrọ ọma. Banye kwụọ.",
            "E jidere ọnụahịa: ₦{price:,} maka {product}. Biko kwụọ ụgwọ.",
            "Ọ dị mma! ₦{price:,} maka {product}. Kwụọ, anyị ga-eziga.",
            "Emeela: ₦{price:,} maka {product}. Nwee obi ike, zụta ugbu a.",
            "Ana m akwado — ₦{price:,} maka {product}. Kwụọ ka e mechaa."
        ],
        "counter": [
            "Enwere m ike ime ₦{price:,} maka {product}. Nke a kacha nso.",
            "Ihe kacha m nwee bụ ₦{price:,} maka {product}. Ị nọ n’aka?",
            "M ga-ebelata ruo ₦{price:,} maka {product}, nke a bụ ikpeazụ.",
            "Mee ka ọ bụrụ ₦{price:,} maka {product} ka anyị kwụsịtụ.",
            "Anọ m na ₦{price:,} maka {product} — ọ bụ ezi onyinye."
        ],
        "reject": [
            "Ndo, nke ahụ adịghị eru — {product} dị ₦{price:,}.",
            "Agaghị m anabata nke ahụ. Achọrọ m ₦{price:,} maka {product}.",
            "Ego ahụ dị ala. ₦{price:,} bụ ezi ọnụahịa.",
            "O nweghị ụzọ — ala kacha nta bụ ₦{price:,} maka {product}.",
            "Biko tụlee ọzọ, ọnụahịa kwesịrị ịbụ ₦{price:,}."
        ],
        "clarify": [
            "Kedụ ọnụahịa ị na-eche maka {product}? Gwa m.",
            "Kedu ego ị nwere maka {product}? Kọwaa.",
            "Biko, tinyekwuo ego ị ga-akwụ maka {product}.",
            "Gosi ọnụahịa kacha mma gị maka {product}. Ka anyị kwuo.",
            "Kedụ ọnụahịa ị na-enye maka {product}? Zaghachi ngwa ngwa."
        ]
    }
}

FEW_SHOT_PROMPT = """
SYSTEM: You are a lively Nigerian market seller, a sharp negotiator bursting with authentic swagger, fluent in English (en), Pidgin (pcm), Yoruba (yo), Hausa (ha), and Igbo (ig). Speak like a native with bold market energy, using {lang_key} tone (short, punchy, dripping with Nigerian charm and banter).

LANGUAGE RULE (mandatory):
 - Reply ONLY in the language the user uses in their message (English, Pidgin, Yoruba, Igbo, or Hausa).
 - If the user mixes languages, detect the dominant language and reply in that dominant language. You MAY mirror short mixed phrases sparingly **only if the user mixed**. Do NOT introduce mixed-language content otherwise.
 - If the user’s language is not detectable or unsupported, default to English with Nigerian seller flair.

GOALS:
 - Keep replies short: 1–2 sentences, maximum 40 words.
 - Be persuasive using warmth, light humour, and culturally natural banter — but never rude or abusive.
 - Protect seller margin: be firm but polite; when refusing an offer, always offer a forward path (a counter or next step).
 - Vary phrasing: rotate among templates and CTAs so responses don’t repeat the exact same line across interactions.
  
PRICING & NUMBERS (strict):
 - Always use the currency symbol ₦.
 - If {final_price} is provided, **echo the numeric price exactly as given** (do not reformat or invent decimals). Use {final_price} verbatim where required.
 - If {final_price} is NOT provided, format the price as a whole-naira integer with comma separators (e.g., 1,250) when rendering templates and replies.
 - Always include the numeric price and the product name somewhere in the reply.
 - Numeric reinforcement: you MAY repeat the numeric price once for emphasis (e.g., “₦15,000 — yes, ₦15,000”), but keep the reply within length limits.

NEGOTIATION BEHAVIOR:
 - ACCEPT: use an acceptance template, express brief gratitude/energy, echo the price (prefer {final_price} when provided), include CTA to pay/confirm.
 - COUNTER: propose a single clear counter price (one number only), explain briefly if needed, end with CTA.
 - REJECT: politely refuse, state the honest minimum (prefer using {final_price} when available or the computed minimum), then immediately present a path (e.g., “I can do ₦X” or “Come back if you can reach ₦X”).
 - CLARIFY: ask for a numeric offer or budget in the user’s language; be short and direct.

STYLE RULES:
 - Tone by language: 
   - English = savvy, confident, with market swagger.
   - Pidgin = playful, street-smart, chop-life vibe.
   - Yoruba = warm, respectful, with local proverbs.
   - Hausa = polite, direct, with community warmth.
   - Igbo = friendly, persuasive, with trader’s charm.
 - Keep replies punchy, avoid long explanations. Use 1–2 short clauses.
 - Do not invent shipping, freebies, discounts, or timelines unless specified.
 - Mirror short mixed-language phrases only if user mixes; reply in dominant language.

CONTEXT: {context}

NUMERIC_GROUNDS:
 final_price: {final_price}

OPERATIONS EXAMPLES (behavioural):
 - If user accepts and final_price is 15000 → reply with acceptance phrase echoing ₦15000 and a CTA.
 - If user counters with an offer → reply with one-number counter (₦X), short rationale optional, end with CTA.
 - If user offers too low → politely state the computed minimum or {final_price} (if provided) and offer a compromise or next step.

IMPLEMENTATION NOTES:
 - Rotate templates so the same template is not reused back-to-back for the same user.
 - Keep replies short enough for chat UI bubbles; prioritize clarity and action.
 - If unsure about language detection, default to English with Nigerian market persona and ask for clarification only once, concisely.

Final instruction: Use this prompt plus the TEMPLATES for {lang_key} as your core. Always be authentic, persuasive, and business-savvy — like a real seller from the market who knows value and respects the customer.
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

def _render_template_reply(template_map: Dict[str, Any], action_key: str, price: Optional[int], product_name: str, ratio: Optional[float] = None, customer: Optional[str] = None) -> str:
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
        cust = customer or "friend"
        # allow templates to optionally include {customer}
        return tmpl.format(price=tpl_price_int, product=product_name, customer=cust)
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

# ------------------ INTENT CLASSIFICATION ------------------
INTENTS = ("GREETING", "ASK_PRICE", "NEGOTIATE", "BUY", "OTHER")

_greeting_re = re.compile(r"\b(hi|hello|hey|good (morning|afternoon|evening)|howdy)\b", re.I)
_ask_price_re = re.compile(r"\b(how much|what(?:'s| is) the price|price|cost|how much for|what(?:'s| is) the cost)\b", re.I)
_buy_re = re.compile(r"\b(i want to buy|i'll take|i will buy|i want this|i want to purchase|i want to order|i want|i buy|i\s+buy|i(?:'ll| will)\s+pay|checkout|pay now|i paid|send payment|send money)\b", re.I)
_negotiate_re = re.compile(r"\b(offer|i can pay|i can afford|i fit pay|i fit|i offer|can you do|will you take|how about|meet you at|i can give|afford|offer)\b", re.I)
_digits_re = re.compile(r"\b\d{3,}\b")  # numbers with at least 3 digits (i.e., offers like 5000)

def classify_intent(text: Optional[str]) -> str:
    """
    Deterministic rule-based intent classifier. Conservative and fast.
    """
    if not text:
        return "OTHER"
    s = text.strip().lower()
    # greeting check first
    if _greeting_re.search(s):
        return "GREETING"
    # buy (explicit purchase intent)
    if _buy_re.search(s):
        return "BUY"
    # ask price explicit question about price
    if _ask_price_re.search(s):
        return "ASK_PRICE"
    # negotiate (offers, afford, numeric offers)
    if _negotiate_re.search(s) or _digits_re.search(s):
        return "NEGOTIATE"
    return "OTHER"

# small language-local friendly quick replies for GREETING/ASK_PRICE/BUY fallbacks
_SIMPLE_INTENT_REPLIES = {
    "GREETING": {
        "en": "Hi! How can I help with this product?",
        "pcm": "Hi! Wetin you need for this product?",
        "yo": "Báwo! Ṣé mo lè ràn ẹ́ lọ́wọ́?",
        "ha": "Sannu! Yaya zan iya taimaka?",
        "ig": "Ndewo! Kedu ihe m nwere ike inyere gị?"
    },
    "ASK_PRICE": {
        # ASK_PRICE should state price (prefer final_price then base_price)
        "en": "The price for {product} is {price}.",
        "pcm": "Price for {product} na {price}.",
        "yo": "{product} jẹ́ {price}.",
        "ha": "Farashin {product} shine {price}.",
        "ig": "{product} bụ {price}."
    },
    "BUY_CONFIRM": {
        "en": "Do you want to confirm purchase for {product} at {price}? Reply yes to confirm.",
        "pcm": "You wan confirm buy {product} for {price}? Reply yes if na so.",
        "yo": "Ṣe o fẹ́ ra {product} fun {price}? Dahun 'bẹẹni' lati jẹrisi.",
        "ha": "Kuna son tabbatar da sayen {product} a {price}? Amsa 'yes' don tabbatarwa.",
        "ig": "Ị chọrọ ikwenye ịzụta {product} na {price}? Zaa 'yes' ka ọ bụrụ nke a."
    }
}

def _intent_override_response(intent: str, lang_key: str, product_name: str, final_price: Optional[int], base_price:int) -> Optional[str]:
    """
    Return a short immediate reply for simple intents (greeting, ask_price, buy with no offer).
    Returns None if no override (i.e., proceed with normal negotiation flow).
    """
    lang_key = _normalize_lang_code(lang_key)
    if intent == "GREETING":
        return _SIMPLE_INTENT_REPLIES["GREETING"].get(lang_key, _SIMPLE_INTENT_REPLIES["GREETING"]["en"])
    if intent == "ASK_PRICE":
        price = final_price if final_price is not None else base_price
        price_str = _format_naira(price)
        tpl = _SIMPLE_INTENT_REPLIES["ASK_PRICE"].get(lang_key) or _SIMPLE_INTENT_REPLIES["ASK_PRICE"]["en"]
        return tpl.format(product=product_name, price=price_str)
    if intent == "BUY":
        # If there's a final price (or computed), prompt confirmation when offer missing.
        price = final_price if final_price is not None else base_price
        price_str = _format_naira(price)
        # If buyer proactively asserted buy (and we have price), ask to confirm:
        return _SIMPLE_INTENT_REPLIES["BUY_CONFIRM"].get(lang_key, _SIMPLE_INTENT_REPLIES["BUY_CONFIRM"]["en"]).format(product=product_name, price=price_str)
    return None

# ------------------ main phrase() ------------------
def phrase(decision: Dict[str, Any], product: Dict[str, Any], lang: str = "en", context: Optional[str] = None, use_remote_expected: Optional[bool] = None, **kwargs) -> str:
    """
    decision: dict possibly containing 'action', 'price', 'offer', 'meta'
    product: dict with 'name' and 'base_price'
    lang: language key (en|pcm|yo|ig|ha) or variant
    use_remote_expected: optional compatibility flag (ignored by this implementation)
    Returns a user-facing string reply.
    """
    # Normalize incoming language codes
    lang_key = _normalize_lang_code(lang)

    prod_name = product.get("name") or product.get("id") or "product"
    base_price = int(product.get("base_price", 0))

    # optional customer name from kwargs
    customer_name = None
    if "customer_name" in kwargs:
        customer_name = kwargs.get("customer_name")
    elif "customer" in kwargs:
        customer_name = kwargs.get("customer")

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

    # If decision meta includes buyer_text, prefer it for intent classification
    buyer_text = None
    try:
        if isinstance(meta, dict):
            buyer_text = meta.get("buyer_text") or meta.get("text") or None
    except Exception:
        buyer_text = None

    user_text_for_intent = buyer_text or sanitized_ctx or kwargs.get("user_text") or ""

    # classify intent (new)
    intent = classify_intent(user_text_for_intent)
    logger.debug("[llm_phraser] classify_intent -> %s (text preview=%r)", intent, (user_text_for_intent[:120] + "...") if user_text_for_intent else "")

    # Intent-driven quick overrides (greeting, ask_price, buy confirmation)
    try:
        quick = _intent_override_response(intent, lang_key, prod_name, final_price, base_price)
        if quick:
            logger.info("[llm_phraser] intent_override=%s lang=%s quick_reply=%s", intent, lang_key, quick[:120])
            return quick
    except Exception:
        logger.exception("[llm_phraser] intent override failure - continuing with normal flow")

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

    logger.debug("[llm_phraser] phrase() computed_action=%s final_price=%s prod=%s lang=%s floor=%s prev_proposals=%s intent=%s",
                 computed_action, final_price, prod_name, lang_key, floor, prev_proposals, intent)

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
            rendered = _render_template_reply(template_map, key, tpl_price, prod_name, ratio, customer_name)
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
            return _render_template_reply(_TEMPLATES.get("en"), "counter", final_price or base_price, prod_name, ratio, customer_name)

    # --- REMOTE preferred (English or pidgin) ---
    if LLM_MODE == "REMOTE":
        out = None
        try:
            remote_lang_forcing = "en" if lang_key not in ("pcm",) else "pcm"
            out = _call_remote_llm(prompt, lang_key=remote_lang_forcing)
            if out:
                # If PCM mode: handle sentinel <UNABLE_PCM> -> fallback to templates
                if remote_lang_forcing == "pcm" and out.strip() == "<UNABLE_PCM>":
                    logger.info("[llm_phraser] remote signalled UNABLE_PCM; falling back to templates")
                else:
                    # numeric safety: must contain final_price when final_price provided
                    if final_price is None or _reply_contains_price(out, final_price):
                        logger.info("[llm_phraser] remote_generation_ok lang=%s preview=%s", lang_key, (out[:120] + "...") if len(out) > 120 else out)
                        return out.strip()
                    else:
                        logger.warning("[llm_phraser] remote returned no usable text or numeric mismatch; falling back")
            else:
                logger.warning("[llm_phraser] remote returned empty response; falling back")
        except Exception:
            logger.exception("[llm_phraser] remote generation error")

    # --- LOCAL fallback ---
    if LLM_MODE == "LOCAL":
        try:
            out = _run_local_generation(prompt)
            if out and (final_price is None or _reply_contains_price(out, final_price)):
                logger.info("[llm_phraser] local_generation_ok lang=%s preview=%s", lang_key, (out[:120] + "...") if len(out) > 120 else out)
                return out.strip()
        except Exception:
            logger.exception("[llm_phraser] local generation error")

    # --- TEMPLATE fallback (final) ---
    template_map = _TEMPLATES.get(lang_key, _TEMPLATES["en"])
    action_map = {"ACCEPT": "accept", "REJECT": "reject", "COUNTER": "counter", "ASK_CLARIFY": "clarify"}
    key = action_map.get(computed_action, "counter")
    try:
        tpl_price = final_price if final_price is not None else base_price
        logger.info("[llm_phraser] final_template lang=%s action=%s price=%s prod=%s prev_proposals=%s floor=%s customer_name=%s intent=%s",
                    lang_key, computed_action, tpl_price, prod_name, prev_proposals, floor, customer_name, intent)
        return _render_template_reply(template_map, key, tpl_price, prod_name, ratio, customer_name=customer_name, lang_key=lang_key)
    except Exception:
        try:
            if customer_name:
                return f"{customer_name}, Our counter price is {_format_naira(final_price)} for {prod_name}."
            return f"Our counter price is {_format_naira(final_price)} for {prod_name}."
        except Exception:
            return f"Our counter price is ₦{final_price} for {prod_name}."
