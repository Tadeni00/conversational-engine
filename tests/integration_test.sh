#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

FASTTEXT_URL="http://localhost:5000/predict"
LANG_DETECT_URL="http://localhost:8000/detect"
USER_PREF_URL_BASE="http://localhost:8000/user"
NEGOTIATION_URL="http://localhost:9000/decide"
WAIT_TIMEOUT=60

echo "=== Full integration test: conversational-engine stack ==="

# ensure jq is available
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required for this script. Please install jq."
  exit 1
fi

wait_for_http() {
  local url=$1; local timeout=${2:-$WAIT_TIMEOUT}
  echo "  -> waiting for $url (timeout ${timeout}s)..."
  local start=$(date +%s)
  while true; do
    if curl -sS --connect-timeout 2 --fail -X GET "$url" >/dev/null 2>&1; then
      echo "     $url is responding"
      return 0
    fi
    sleep 1
    now=$(date +%s)
    if (( now - start >= timeout )); then
      echo "ERROR: $url did not respond in ${timeout}s"
      return 1
    fi
  done
}

wait_for_http "http://localhost:8000/ping"

# wait for fasttext predict
echo "  -> waiting for fasttext predict to respond..."
start=$(date +%s)
while true; do
  if curl -sS --connect-timeout 2 --fail -X POST "$FASTTEXT_URL" -H 'Content-Type: application/json' -d '{"text":"hello"}' >/dev/null 2>&1 ; then
    echo "     fasttext predict responding"
    break
  fi
  sleep 1
  now=$(date +%s)
  if (( now - start >= WAIT_TIMEOUT )); then
    echo "ERROR: fasttext predict did not respond in ${WAIT_TIMEOUT}s"
    exit 1
  fi
done

echo
echo "=== Component smoke tests ==="

payload='{"text":"Abeg, how much for this?"}'
echo "- FastText predict (direct): $payload"
ft_resp=$(curl -sS -X POST "$FASTTEXT_URL" -H 'Content-Type: application/json' -d "$payload")
echo "  -> resp: $ft_resp"
echo "$ft_resp" | jq -e '.lang and .score' >/dev/null || { echo "FastText response missing lang/score"; exit 2; }

echo "- Language-detection (no user_id): $payload"
ld_resp=$(curl -sS -X POST "$LANG_DETECT_URL" -H 'Content-Type: application/json' -d "$payload")
echo "  -> resp: $ld_resp"
lang=$(echo "$ld_resp" | jq -r '.language // empty')
conf=$(echo "$ld_resp" | jq -r '.confidence // 0')
if [ -z "$lang" ]; then
  echo "Language detection did not return a language"; exit 3
fi
echo "  -> detected language: $lang (conf=${conf})"

SUPPORTED_LANGS=("en" "pcm" "yo" "ha" "ig")
USER_ID="integration-test-user-$(date +%s)"

for L in "${SUPPORTED_LANGS[@]}"; do
  echo
  echo "=== Testing preference cycle for language: $L ==="
  set_resp=$(curl -sS --fail -X POST "$USER_PREF_URL_BASE/$USER_ID/lang" -H 'Content-Type: application/json' -d "{\"language\":\"$L\"}" || true)
  echo "  -> set response: $set_resp"
  ok=$(echo "$set_resp" | jq -r '.ok // empty' || true)
  if [ "$ok" != "true" ]; then
    echo "Failed to set user pref for $L (response did not contain ok:true). Response: $set_resp"
    exit 4
  fi

  payload_with_user=$(jq -nc --arg t "Abeg, how much for this?" --arg u "$USER_ID" '{text:$t, user_id:$u}')
  ld_pref_resp=$(curl -sS --fail -X POST "$LANG_DETECT_URL" -H 'Content-Type: application/json' -d "$payload_with_user")
  echo "  -> detect with user_id resp: $ld_pref_resp"
  detected_by_pref=$(echo "$ld_pref_resp" | jq -r '.language // empty')
  conf_pref=$(echo "$ld_pref_resp" | jq -r '.confidence // 0')

  if [ "$detected_by_pref" != "$L" ]; then
    echo "User preference NOT honored (expected $L). Response: $ld_pref_resp"
    exit 5
  fi

  # portable float comparison using python
  if ! python - <<PY
conf = float("$conf_pref")
import sys
sys.exit(0 if conf >= 0.9 else 1)
PY
  then
    echo "Preference returned with low confidence ($conf_pref), expected >= 0.9"
    exit 6
  fi
  echo "  -> user pref honored (language=$detected_by_pref conf=$conf_pref)"
done

echo
echo "=== Optional negotiation endpoint test ==="
if curl -sS --connect-timeout 2 -X GET http://localhost:9000/ping >/dev/null 2>&1 || curl -sS --connect-timeout 2 -X GET http://localhost:9000/health >/dev/null 2>&1; then
  echo "Negotiation service appears to be up"
else
  echo "Negotiation service not responding to /ping; attempting a POST /decide (non-fatal)..."
fi

# your negotiation schema provided by you:
declare -A texts
texts[en]="How much for this?"
texts[pcm]="How much for this? (pcm sample)"
texts[yo]="Bawo, elo lo?"
texts[ha]="Nawa?"
texts[ig]="Ego ole?"

# build payload as you specified (per-language buyer_text)
lang="pcm"
decide_payload=$(cat <<JSON
{
  "offer": 5000,
  "product": {"id":"sku-lip-001","name":"Matte Lipstick - Ruby","base_price":12000},
  "state": {"conversation_id":"t1","user_id":"u1", "meta": {"buyer_text": "${texts[$lang]}"}}
}
JSON
)

decide_resp=$(curl -sS -X POST "$NEGOTIATION_URL" -H 'Content-Type: application/json' -d "$decide_payload" || true)
echo "  -> negotiation /decide response: $decide_resp"
if [ -n "$decide_resp" ]; then
  if echo "$decide_resp" | jq -e 'true' >/dev/null 2>&1; then
    echo "  -> negotiation returned valid JSON (OK)"
  else
    echo "  -> negotiation response not valid JSON (continuing)"
  fi
else
  echo "  -> negotiation /decide did not respond (service may not be started) — continuing"
fi

# cleanup: delete pref (best-effort) via DELETE endpoint
curl -sS -X DELETE "$USER_PREF_URL_BASE/$USER_ID/lang" -H 'Content-Type: application/json' || true

echo
echo "=== Full integration checks passed ==="
exit 0
