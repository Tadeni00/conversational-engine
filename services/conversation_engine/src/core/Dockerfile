# services/conversation_engine/src/core/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# small apt deps needed for some wheels (fasttext etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY services/conversation_engine/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Put the package sources under /app/conversation_engine so imports work:
COPY services/conversation_engine/src/ /app/conversation_engine/
# copy wrapper to app.py so uvicorn app:app will work
COPY services/conversation_engine/src/app.py /app/app.py

# Also make llm_phraser available at /app/llm_phraser.py (optional helper)
COPY services/negotiation_service/llm_phraser.py /app/llm_phraser.py

EXPOSE 8000

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000","--workers","1"]
