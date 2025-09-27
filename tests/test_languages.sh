#!/usr/bin/env bash
set -eu

# change host/port if your negotiation service is elsewhere
URL="http://localhost:9000/decide"

declare -A texts
texts[en]="I can afford 5000"
texts[pcm]="I fit pay 5000"
texts[yo]="Mo le san 5000"
texts[ha]="Zan iya biyan 5000"
texts[ig]="Enwere m ike ikwu 5000"

for lang in en pcm yo ha ig; do
  echo
  echo "===== $lang ====="
  payload=$(cat <<JSON
{
  "offer": 5000,
  "product": {"id":"sku-lip-001","name":"Matte Lipstick - Ruby","base_price":12000},
  "state": {"conversation_id":"t1","user_id":"u1", "meta": {"buyer_text": "${texts[$lang]}"}}
}
JSON
)
  # send request and pretty-print
  curl -sS -X POST "$URL" -H "Content-Type: application/json" -d "$payload" | jq
done
