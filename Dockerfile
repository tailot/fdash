FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia entrambi i file: requirements-timesfm.txt include requirements.txt (-r)
COPY requirements.txt requirements-timesfm.txt ./

# 1) torch versione CPU (molto più leggero della versione CUDA)
# 2) dipendenze dell'app + TimesFM-3
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-timesfm.txt

COPY . .

# I pesi del modello (Hugging Face) vengono scaricati qui al primo avvio
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
