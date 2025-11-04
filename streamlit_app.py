"""
Streamlit UI for Conversational Engine

Single-file Streamlit app to exercise the stack you built (FastText service, Language-detection service, Negotiation service, and user language preference endpoints).

Features:
- Send text to language-detection (/detect) and fasttext (/predict) services
- Set / DELETE user language preference (POST /user/{user_id}/lang, DELETE /user/{user_id}/lang)
- Call negotiation service (/decide) with a small payload builder
- Configurable service URLs from sidebar (defaults assume local docker-compose)

Usage:
1. Install dependencies: `pip install streamlit requests`
2. Run: `streamlit run streamlit_app.py`
3. Open the browser UI and interact.

Notes:
- This UI expects your services to be reachable (defaults: language-detection http://localhost:8000, fasttext http://localhost:5000, negotiation http://localhost:9000).
- Timeouts are short by default to keep UI snappy; you can change them in the sidebar.

"""

import json
import requests
import streamlit as st
from typing import Dict, Any

# -----------------------------
# Helpers
# -----------------------------

def safe_post(url: str, json_payload: Dict[str, Any], timeout: float = 3.0):
    try:
        r = requests.post(url, json=json_payload, timeout=timeout)
        r.raise_for_status()
        try:
            return True, r.json()
        except Exception:
            return True, r.text
    except Exception as e:
        return False, str(e)


def safe_get(url: str, timeout: float = 3.0):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        try:
            return True, r.json()
        except Exception:
            return True, r.text
    except Exception as e:
        return False, str(e)


def safe_delete(url: str, timeout: float = 3.0):
    try:
        r = requests.delete(url, timeout=timeout)
        r.raise_for_status()
        try:
            return True, r.json()
        except Exception:
            return True, r.text
    except Exception as e:
        return False, str(e)


# -----------------------------
# Streamlit layout
# -----------------------------

st.set_page_config(page_title="Conversational Engine UI", layout="wide")
st.title("Conversational Engine — Dev UI")

# Sidebar: configuration
st.sidebar.header("Service configuration")
LANG_DETECT_URL = st.sidebar.text_input("Language-detection base URL", "http://localhost:8000")
FASTTEXT_URL = st.sidebar.text_input("FastText service URL", "http://localhost:5000")
NEGOTIATION_URL = st.sidebar.text_input("Negotiation /decide URL", "http://localhost:9000/decide")
TIMEOUT = st.sidebar.slider("Request timeout (s)", min_value=1.0, max_value=10.0, value=3.0)

st.sidebar.markdown("---")
st.sidebar.markdown("Quick actions")
if st.sidebar.button("Ping language-detection"):
    ok, resp = safe_get(f"{LANG_DETECT_URL}/ping", timeout=TIMEOUT)
    if ok:
        st.sidebar.success("language-detection: OK")
        st.sidebar.json(resp)
    else:
        st.sidebar.error(f"Failed: {resp}")

if st.sidebar.button("Ping fasttext"):
    ok, resp = safe_post(f"{FASTTEXT_URL}/predict", {"text":"hello"}, timeout=TIMEOUT)
    if ok:
        st.sidebar.success("fasttext: OK")
        st.sidebar.json(resp)
    else:
        st.sidebar.error(f"Failed: {resp}")

if st.sidebar.button("Ping negotiation"):
    ok, resp = safe_post(NEGOTIATION_URL, {"offer": 1}, timeout=TIMEOUT)
    if ok:
        st.sidebar.success("negotiation: responded")
        st.sidebar.json(resp)
    else:
        st.sidebar.error(f"Failed: {resp}")

# Main panels
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Text / Language Detection")
    user_id = st.text_input("User ID (optional)", value=st.session_state.get("user_id", ""))
    st.session_state["user_id"] = user_id

    text = st.text_area("Message text", value=st.session_state.get("last_text", "Abeg, how much for this?"), height=130)
    st.session_state["last_text"] = text

    btn_detect = st.button("Detect language (via language-detection service)")
    btn_fasttext = st.button("FastText predict (direct)")

    if btn_detect:
        payload = {"text": text}
        if user_id:
            payload["user_id"] = user_id
        ok, resp = safe_post(f"{LANG_DETECT_URL}/detect", payload, timeout=TIMEOUT)
        if ok:
            st.success("Detect OK")
            st.json(resp)
        else:
            st.error(f"Detect failed: {resp}")

    if btn_fasttext:
        ok, resp = safe_post(f"{FASTTEXT_URL}/predict", {"text": text}, timeout=TIMEOUT)
        if ok:
            st.success("FastText OK")
            st.json(resp)
        else:
            st.error(f"FastText failed: {resp}")

    st.markdown("---")
    st.subheader("User language preference (Redis)")
    pref_col1, pref_col2 = st.columns([2, 1])
    with pref_col1:
        pref_lang = st.text_input("Preference language code (e.g. pcm, en, yo)", value="pcm")
    with pref_col2:
        set_pref = st.button("Set preference for user")
        del_pref = st.button("Delete preference for user")

    if set_pref:
        if not user_id:
            st.error("User ID is required to set a preference")
        else:
            ok, resp = safe_post(f"{LANG_DETECT_URL}/user/{user_id}/lang", {"language": pref_lang}, timeout=TIMEOUT)
            if ok:
                st.success("Set preference")
                st.json(resp)
            else:
                st.error(f"Set failed: {resp}")

    if del_pref:
        if not user_id:
            st.error("User ID is required to delete preference")
        else:
            ok, resp = safe_delete(f"{LANG_DETECT_URL}/user/{user_id}/lang", timeout=TIMEOUT)
            if ok:
                st.success("Deleted preference")
                st.json(resp)
            else:
                st.error(f"Delete failed: {resp}")

    st.markdown("---")
    st.subheader("Quick examples")
    with st.expander("Example messages"):
        st.write(
            "\n".join([
                "Abeg, how much for this? (pcm)",
                "How much for this? (en)",
                "Mo le san 5000 (yo)",
                "Zan iya biyan 5000 (ha)",
                "Enwere m ike ikwu 5000 (ig)",
            ])
        )

with col2:
    st.subheader("Negotiation tester")
    st.markdown("Build a negotiation payload and POST to /decide")

    nego_user_id = st.text_input("Negotiation user_id", value=st.session_state.get("nego_user_id", "u1"))
    st.session_state["nego_user_id"] = nego_user_id

    lang_for_buyer = st.selectbox("Buyer language (used only for message text)", ["en", "pcm", "yo", "ha", "ig"], index=0)

    sample_texts = {
        "en": "I can afford 5000",
        "pcm": "I fit pay 5000",
        "yo": "Mo le san 5000",
        "ha": "Zan iya biyan 5000",
        "ig": "Enwere m ike ikwu 5000",
    }

    buyer_text = st.text_area("Buyer text", value=sample_texts[lang_for_buyer], height=80)

    offer = st.number_input("Offer", value=5000, min_value=0)
    base_price = st.number_input("Product base price", value=12000, min_value=0)
    product_id = st.text_input("Product id", value="sku-lip-001")
    product_name = st.text_input("Product name", value="Matte Lipstick - Ruby")

    if st.button("Call negotiation /decide"):
        payload = {
            "offer": offer,
            "product": {"id": product_id, "name": product_name, "base_price": base_price},
            "state": {"conversation_id": "t1", "user_id": nego_user_id, "meta": {"buyer_text": buyer_text}},
        }
        st.write("Request payload:")
        st.json(payload)
        ok, resp = safe_post(NEGOTIATION_URL, payload, timeout=TIMEOUT)
        if ok:
            st.success("Negotiation responded")
            st.json(resp)
        else:
            st.error(f"Negotiation failed: {resp}")


# Footer / notes
st.markdown("---")
st.write(
    "This developer UI is intended for manual testing. For automated integration tests use the `tests/` scripts in the repo.\n"
    "Change service URLs in the sidebar if your services are on different hosts or running in containers." 
)

# keep a small developer info block
with st.expander("Developer info"):
    st.write("Default endpoints used:")
    st.code(f"language-detection: {LANG_DETECT_URL}\nfasttext: {FASTTEXT_URL}\nnegotiation: {NEGOTIATION_URL}")
    st.write("Timeout (s):", TIMEOUT)
