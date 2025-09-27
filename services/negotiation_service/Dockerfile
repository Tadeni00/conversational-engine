# services/negotiation_service/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# small apt deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY services/negotiation_service/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the negotiation service into a package directory
COPY services/negotiation_service/ /app/negotiation_service/

# ensure __init__.py exists (safe) and entrypoint is executable
RUN test -f /app/negotiation_service/__init__.py || touch /app/negotiation_service/__init__.py
RUN chmod +x /app/negotiation_service/entrypoint.sh || true

WORKDIR /app/negotiation_service

EXPOSE 9000

ENTRYPOINT ["/app/negotiation_service/entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
