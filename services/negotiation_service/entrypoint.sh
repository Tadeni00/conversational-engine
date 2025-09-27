#!/usr/bin/env sh
set -eu

_mask() {
  s="$1"
  if [ -z "$s" ]; then
    printf "%s" "<missing>"
    return
  fi
  len=$(printf "%s" "$s" | wc -c)
  if [ "$len" -le 10 ]; then
    printf "%s" "<hidden>"
  else
    first=$(printf "%s" "$s" | cut -c1-3)
    last=$(printf "%s" "$s" | rev | cut -c1-3 | rev)
    printf "%s" "${first}...${last}"
  fi
}

echo "[entrypoint] starting container"
echo "[entrypoint] python: $(python -V 2>&1 || true)"
echo "[entrypoint] cwd: $(pwd)"
echo "[entrypoint] ls /app (top):"
ls -la /app | sed -n '1,200p' || true

# Load secrets from /run/secrets/* into environment variables
if [ -d /run/secrets ]; then
  for f in /run/secrets/* ; do
    [ -f "$f" ] || continue
    name=$(basename "$f" | tr '[:lower:]' '[:upper:]' | sed 's/[^A-Z0-9_]/_/g')
    # read and trim CR/LF
    if val="$(tr -d '\r\n' < "$f" 2>/dev/null)"; then
      export "${name}"="${val}"
      export "${name}_FILE"="$f"
    fi
  done
fi

echo "[entrypoint] verifying required python packages..."
python - <<'PY'
import sys
missing = []
for pkg in ("fastapi","uvicorn","requests","pydantic"):
    try:
        __import__(pkg)
    except Exception as e:
        missing.append((pkg, str(e)))
if missing:
    print("[entrypoint] Missing python packages or import errors:", file=sys.stderr)
    for p,err in missing:
        print(f" - {p}: {err}", file=sys.stderr)
    sys.exit(1)
print("[entrypoint] python package check OK")
PY

echo "[entrypoint] Environment (masked) diagnostics:"
for key in GROQ_API_KEY OPENAI_API_KEY HF_TOKEN LLM_API_KEY; do
  # get value of $key safely using eval (POSIX sh)
  val="$(eval "printf '%s' \"\${${key}:-}\"")"
  filevar="${key}_FILE"
  fileval="$(eval "printf '%s' \"\${${filevar}:-<none>}\"")"
  if [ -n "$val" ]; then
    masked=$(_mask "$val")
    echo "  $key = $masked (loaded from: ${fileval})"
  else
    echo "  $key = <missing> (file: ${fileval})"
  fi
done

# Back-compat
if [ -f /app/run.sh ]; then
  echo "[entrypoint] running /app/run.sh"
  exec /app/run.sh
fi

# Exec supplied command
if [ "$#" -gt 0 ]; then
  echo "[entrypoint] launching provided command: $*"
  exec "$@"
fi

UVICORN_MODULE=${UVICORN_MODULE:-"app:app"}
UVICORN_HOST=${UVICORN_HOST:-"0.0.0.0"}
UVICORN_PORT=${UVICORN_PORT:-${PORT:-8000}}
UVICORN_WORKERS=${UVICORN_WORKERS:-1}
UVICORN_CMD="uvicorn ${UVICORN_MODULE} --host ${UVICORN_HOST} --port ${UVICORN_PORT} --workers ${UVICORN_WORKERS}"

echo "[entrypoint] no command provided; launching default: ${UVICORN_CMD}"
exec sh -c "${UVICORN_CMD}"
