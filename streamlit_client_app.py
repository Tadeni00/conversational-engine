# streamlit_client_app.py
"""
Streamlit client for the conversational-engine stack

Features:
- Test language-detection (/detect) with/without user_id
- Set / Delete user language preference (/user/{user_id}/lang)
- Send negotiation requests (/decide) using the project's schema
- Quick integration test that exercises language-pref cycle and negotiation

How to run:
1. Create a virtualenv and install requirements
   python -m venv .venv && source .venv/bin/activate
   pip install streamlit requests python-dotenv

2. Copy this file to your workspace and run:
   streamlit run streamlit_client_app.py

3. Make sure your services are running (defaults):
   - Language-detection: http://localhost:8000
   - FastText predict: http://localhost:5000
   - Negotiation: http://localhost:9000

You can change endpoints in the sidebar or provide a .env file with keys:
LANG_DETECT_URL, NEGOTIATION_URL, FASTTEXT_URL

This app is intentionally lightweight and safe — it only makes HTTP calls to your local services
and displays the JSON responses.

"""

import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

# Defaults
DEFAULT_LANG_DETECT = os.getenv("LANG_DETECT_URL", "http://localhost:8000")
DEFAULT_NEGOTIATION = os.getenv("NEGOTIATION_URL", "http://localhost:9000")
DEFAULT_FASTTEXT = os.getenv("FASTTEXT_URL", "http://localhost:5000")

st.set_page_config(page_title="Conversational Engine UI", layout="wide")

if "user_id" not in st.session_state:
    st.session_state.user_id = os.getenv("DEFAULT_USER_ID", "demo-user-1")

SUPPORTED_LANGS = ["en", "pcm", "yo", "ha", "ig"]

# Sidebar: configuration
with st.sidebar:
    st.title("Config")
    base_lang_detect = st.text_input("Language-detection base URL", value=DEFAULT_LANG_DETECT)
    base_negotiation = st.text_input("Negotiation base URL", value=DEFAULT_NEGOTIATION)
    base_fasttext = st.text_input("FastText predict URL", value=DEFAULT_FASTTEXT)
    st.session_state.user_id = st.text_input("User ID (session)", value=st.session_state.user_id)
    show_raw = st.checkbox("Show raw JSON responses", value=True)
    st.markdown("---")
    st.markdown("**Quick links**")
    st.caption("Ensure your services are running (language-detection, fasttext, negotiation, redis)")

# Utility functions

def pretty_json(o):
    try:
        return json.dumps(o, indent=2, ensure_ascii=False)
    except Exception:
        return str(o)

def post_json(url: str, payload: Dict, timeout: float = 5.0):
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        try:
            j = resp.json()
        except Exception:
            j = {"http_status": resp.status_code, "text": resp.text}
        return resp.status_code, j
    except Exception as e:
        return None, {"error": str(e)}

# Layout: three columns top
col1, col2 = st.columns([2, 2])

# Language detection panel
with col1:
    st.header("Language detection")
    detect_text = st.text_area("Text to detect", value="Abeg, how much for this?", height=110)
    detect_user = st.text_input("Optional user_id (honor stored preference)", value=st.session_state.user_id)

    if st.button("Detect language"):
        url = f"{base_lang_detect.rstrip('/')}/detect"
        payload = {"text": detect_text}
        if detect_user:
            payload["user_id"] = detect_user
        status, resp = post_json(url, payload)
        st.subheader("Result")
        if status is None:
            st.error(resp.get("error"))
        else:
            if show_raw:
                st.code(pretty_json(resp))
            else:
                st.success(f"language={resp.get('language')} conf={resp.get('confidence')}")
            # quick action: set this as user pref
            if resp.get("language"):
                if st.button("Set detected language as user preference"):
                    # call set pref
                    set_url = f"{base_lang_detect.rstrip('/')}/user/{detect_user}/lang"
                    try:
                        set_payload = {"language": resp.get("language")}
                        s_status, s_resp = post_json(set_url, set_payload)
                        if s_status and s_status == 200:
                            st.success("Preference set: %s" % s_resp.get("language"))
                            if show_raw:
                                st.code(pretty_json(s_resp))
                        else:
                            st.error("Failed to set preference: %s" % pretty_json(s_resp))
                    except Exception as e:
                        st.error(str(e))

# Preference panel
with col2:
    st.header("User language preference")
    pref_user = st.text_input("User ID (for pref actions)", value=st.session_state.user_id, key="pref_user")
    pref_lang = st.selectbox("Choose language", options=SUPPORTED_LANGS, index=0)
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Set preference"):
            set_url = f"{base_lang_detect.rstrip('/')}/user/{pref_user}/lang"
            st.info(f"Calling: POST {set_url}")
            status, resp = post_json(set_url, {"language": pref_lang})
            if status == 200:
                st.success("Preference saved: %s" % resp.get("language"))
                if show_raw:
                    st.code(pretty_json(resp))
            else:
                st.error(f"Failed to set pref: {pretty_json(resp)}")
    with c2:
        if st.button("Delete preference"):
            del_url = f"{base_lang_detect.rstrip('/')}/user/{pref_user}/lang"
            try:
                r = requests.delete(del_url, timeout=5.0)
                try:
                    jr = r.json()
                except Exception:
                    jr = {"status_text": r.text, "code": r.status_code}
                if r.status_code == 200:
                    st.success("Deleted preference")
                    if show_raw:
                        st.code(pretty_json(jr))
                else:
                    st.error("Delete failed: %s" % pretty_json(jr))
            except Exception as e:
                st.error(str(e))
    st.markdown("---")
    if st.button("Show stored preference for user"):
        get_url = f"{base_lang_detect.rstrip('/')}/user/{pref_user}/lang"
        try:
            r = requests.get(get_url, timeout=4.0)
            if r.status_code == 200:
                st.success(f"Stored pref: {r.json().get('language')}")
                if show_raw:
                    st.code(pretty_json(r.json()))
            else:
                st.error(f"Error reading pref: {r.status_code} {r.text}")
        except Exception as e:
            st.error(str(e))

st.markdown("---")

# Negotiation panel
st.header("Negotiation /decide playground")
colA, colB = st.columns([2,1])
with colA:
    offer = st.number_input("Offer (integer)", value=5000, step=100)
    pid = st.text_input("Product ID", value="sku-lip-001")
    pname = st.text_input("Product name", value="Matte Lipstick - Ruby")
    base_price = st.number_input("Product base price", value=12000, step=100)
    buyer_text = st.text_area("Buyer text / buyer message", value="I can afford 5000", height=90)
    conversation_id = st.text_input("conversation_id", value="t1")
    user_id = st.text_input("user_id (for negotiation)", value=st.session_state.user_id)
    if st.button("Send negotiation request"):
        decide_url = f"{base_negotiation.rstrip('/')}/decide"
        payload = {
            "offer": int(offer),
            "product": {"id": pid, "name": pname, "base_price": int(base_price)},
            "state": {"conversation_id": conversation_id, "user_id": user_id, "meta": {"buyer_text": buyer_text}}
        }
        status, resp = post_json(decide_url, payload, timeout=8.0)
        if status is None:
            st.error(resp.get("error"))
        else:
            if show_raw:
                st.code(pretty_json(resp))
            else:
                st.success(f"reply: {resp.get('reply')}")

with colB:
    st.markdown("**Language templates**")
    for L in SUPPORTED_LANGS:
        st.write(f"{L}: {''}")
    st.markdown("---")
    if st.button("Autofill buyer_text for PCM"):
        st.experimental_set_query_params()  # no-op to avoid lint
        st.session_state._autofill = True
        st.experimental_rerun()

# Integration test
st.markdown("---")
st.header("Quick integration test")
if st.button("Run quick pref + detect + negotiation test for all languages"):
    summary = {}
    for L in SUPPORTED_LANGS:
        u = f"test-user-{L}"
        # set pref
        set_url = f"{base_lang_detect.rstrip('/')}/user/{u}/lang"
        s_status, s_resp = post_json(set_url, {"language": L})
        # detect with user id
        det_url = f"{base_lang_detect.rstrip('/')}/detect"
        det_payload = {"text": "Abeg, how much for this?", "user_id": u}
        d_status, d_resp = post_json(det_url, det_payload)
        # negotiation
        decide_url = f"{base_negotiation.rstrip('/')}/decide"
        decide_payload = {"offer": 5000, "product": {"id": "sku-lip-001", "name": "Matte Lipstick - Ruby", "base_price": 12000}, "state": {"conversation_id": "t1", "user_id": u, "meta": {"buyer_text": det_payload['text']}}}
        dec_status, dec_resp = post_json(decide_url, decide_payload)
        summary[L] = {"set": s_resp, "detect": d_resp, "decide": dec_resp}
    st.write("### Results")
    for L, v in summary.items():
        st.write(f"**{L}**")
        st.code(pretty_json(v))

st.markdown("---")

st.caption("Built for local dev. Edit endpoints in the sidebar. Report issues to your dev team.")
import streamlit as st
import requests
import time
import uuid
from typing import List, Dict, Any, Optional

# Simple Streamlit client for continuous negotiations
# Assumptions:
# - negotiation service available at http://localhost:9000/decide
# - optional language-detection service at http://localhost:8000
# - product schema: {"id","name","base_price"}
# - request payload follows user's schema in repo

NEGOTIATION_URL = st.secrets.get("NEGOTIATION_URL", "http://localhost:9000/decide")
LANG_DETECT_URL = st.secrets.get("LANG_DETECT_URL", "http://localhost:8000/detect")
USER_PREF_BASE = st.secrets.get("USER_PREF_BASE", "http://localhost:8000/user")

st.set_page_config(page_title="Negotiation Buyer UI", layout="wide")

if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"conv-{str(uuid.uuid4())[:8]}"
if "user_id" not in st.session_state:
    st.session_state.user_id = f"buyer-{str(uuid.uuid4())[:6]}"
if "meta" not in st.session_state:
    st.session_state.meta = {"prev_proposals": []}
if "agreed" not in st.session_state:
    st.session_state.agreed = False

# Helpful functions

def post_decide(payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(NEGOTIATION_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Negotiation request failed: {e}")
        return None


def detect_language(text: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(LANG_DETECT_URL, json={"text": text}, timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# Sidebar: controls and product selection
with st.sidebar:
    st.title("Buyer settings")
    st.text_input("Conversation ID", key="conversation_id_input", value=st.session_state.conversation_id)
    st.text_input("Buyer user_id", key="user_id_input", value=st.session_state.user_id)

    # simple product picker
    st.markdown("**Product**")
    product_id = st.selectbox("Product id", options=["sku-lip-001", "sku-skin-002", "sku-frag-003"], index=0)
    product_map = {
        "sku-lip-001": {"id": "sku-lip-001", "name": "Matte Lipstick - Ruby", "base_price": 12000},
        "sku-skin-002": {"id": "sku-skin-002", "name": "Glow Serum", "base_price": 8000},
        "sku-frag-003": {"id": "sku-frag-003", "name": "Signature Perfume", "base_price": 25000},
    }
    product = product_map.get(product_id)

    st.write(product["name"] + " — " + f"₦{product['base_price']:,}")

    st.write("---")
    st.checkbox("Automatically attempt language detection of buyer text (best-effort)", key="auto_detect_lang", value=True)

    st.write("Service endpoints")
    NEGOTIATION = st.text_input("Negotiation URL", value=NEGOTIATION_URL)
    LANG_DETECT = st.text_input("Language-detect URL", value=LANG_DETECT_URL)
    st.markdown("---")
    if st.button("Reset conversation"):
        st.session_state.history = []
        st.session_state.meta = {"prev_proposals": []}
        st.session_state.agreed = False
        st.session_state.conversation_id = st.session_state.conversation_id_input or st.session_state.conversation_id
        st.session_state.user_id = st.session_state.user_id_input or st.session_state.user_id
        st.success("Conversation reset")

# Apply updates from sidebar inputs
st.session_state.conversation_id = st.session_state.get("conversation_id_input") or st.session_state.conversation_id
st.session_state.user_id = st.session_state.get("user_id_input") or st.session_state.user_id
NEGOTIATION_URL = NEGOTIATION
LANG_DETECT_URL = LANG_DETECT

# Main UI: history and input
st.title("Negotiation — Buyer UI")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Conversation")
    # show history
    for msg in st.session_state.history:
        who = msg.get("who")
        t = msg.get("text")
        meta = msg.get("meta")
        if who == "buyer":
            st.markdown(f"**You:** {t}")
        else:
            st.markdown(f"**Seller:** {t}")
            if meta:
                st.caption(f"meta: {meta}")
    st.write("\n")

    if st.session_state.agreed:
        st.success("Agreement reached — negotiation finished ✅")
        st.button("Start new negotiation", on_click=lambda: st.session_state.update({"agreed": False, "history": [], "meta": {"prev_proposals": []}}))
    else:
        with st.form(key="offer_form"):
            buyer_text = st.text_area("Message / Offer text", height=100, placeholder="Write a message, e.g. 'I can afford 5000' or 'How much for this?'")
            offer_input = st.text_input("Numeric offer (optional, numbers only)", value="", placeholder="e.g. 5000")
            submitted = st.form_submit_button("Send to seller")
            if submitted:
                # prepare payload
                offer_val = None
                try:
                    if offer_input and offer_input.strip() != "":
                        offer_val = int(re.sub(r"[^0-9]", "", offer_input))
                except Exception:
                    st.warning("Could not parse numeric offer — sending without numeric offer")
                    offer_val = None

                # update buyer message in history
                st.session_state.history.append({"who": "buyer", "text": buyer_text or (f"Offer: {offer_val}" if offer_val else "(no text)"), "ts": time.time()})

                # optional language detection
                if st.session_state.auto_detect_lang and buyer_text:
                    try:
                        det = detect_language(buyer_text)
                        if det and det.get("language"):
                            st.info(f"Detected language: {det.get('language')} (conf={det.get('confidence')})")
                    except Exception:
                        pass

                # build payload following your schema
                payload = {
                    "offer": offer_val,
                    "product": product,
                    "state": {
                        "conversation_id": st.session_state.conversation_id,
                        "user_id": st.session_state.user_id,
                        "meta": {
                            "buyer_text": buyer_text,
                            # carry previous proposals so negotiation service can adapt
                            "prev_proposals": st.session_state.meta.get("prev_proposals", [])
                        }
                    }
                }

                # call negotiation service
                resp = post_decide(payload)
                if resp is None:
                    st.session_state.history.append({"who": "seller", "text": "(no response from negotiation service)", "meta": None, "ts": time.time()})
                else:
                    # append seller reply and update meta tracking
                    seller_text = resp.get("reply") or str(resp)
                    meta = resp.get("meta") or {}
                    action = resp.get("action")
                    price = resp.get("price")

                    # update prev_proposals if seller suggested a counter
                    prev = st.session_state.meta.get("prev_proposals", [])
                    # if meta includes next_proposal or prev_proposals append
                    if isinstance(meta, dict):
                        if meta.get("next_proposal"):
                            prev = list(prev) + [int(meta.get("next_proposal"))]
                        if meta.get("prev_proposals") and isinstance(meta.get("prev_proposals"), list):
                            # merge
                            for p in meta.get("prev_proposals"):
                                try:
                                    prev.append(int(p))
                                except Exception:
                                    pass
                    st.session_state.meta["prev_proposals"] = prev

                    st.session_state.history.append({"who": "seller", "text": seller_text, "meta": meta, "action": action, "price": price, "ts": time.time()})

                    # if seller accepted, mark agreed
                    if action and str(action).upper() in ("ACCEPT",):
                        st.session_state.agreed = True
                        st.success(f"Seller accepted — price: {price}")

                # rerun to show updated history
                st.rerun()

with col2:
    st.subheader("Negotiation state")
    st.markdown(f"**Conversation ID:** {st.session_state.conversation_id}")
    st.markdown(f"**Buyer (user_id):** {st.session_state.user_id}")
    st.markdown(f"**Product:** {product['name']} — ₦{product['base_price']:,}")
    st.write("---")
    st.markdown("**Previous proposals:**")
    prevs = st.session_state.meta.get("prev_proposals", [])
    if prevs:
        for p in prevs:
            st.write(f"- {p}")
    else:
        st.write("(none yet)")

    st.write("---")
    st.markdown("**Quick actions**")
    if st.button("Set user language pref -> pcm (Pidgin)"):
        try:
            r = requests.post(f"{USER_PREF_BASE}/{st.session_state.user_id}/lang", json={"language": "pcm"}, timeout=3)
            if r.ok:
                st.success("Saved user language preference (pcm)")
            else:
                st.error(f"Failed to save pref: {r.text}")
        except Exception as e:
            st.error(f"Error: {e}")
    if st.button("Clear user lang pref"):
        try:
            r = requests.delete(f"{USER_PREF_BASE}/{st.session_state.user_id}/lang", timeout=3)
            if r.ok:
                st.success("Deleted user language preference")
            else:
                st.error(f"Failed to delete pref: {r.text}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.write("---")
    st.markdown("**Tips**")
    st.write("• Type natural messages or short offers like 'I can do 6000' or 'How much for this?'.")
    st.write("• Use the numeric offer field to explicitly send an offer number.")
    st.write("• The UI preserves prev proposals so the negotiation service can produce sensible counters.")
    st.write("• When the seller returns ACCEPT, use the Agree button to finish the flow.")


# bottom: raw debug panel (collapsed)
with st.expander("Debug: raw session_state and last response"):
    st.json({"conversation_id": st.session_state.conversation_id, "user_id": st.session_state.user_id, "meta": st.session_state.meta})
    if st.session_state.history:
        st.write("--- last 5 messages ---")
        for h in st.session_state.history[-5:]:
            st.write(h)


# end of file
