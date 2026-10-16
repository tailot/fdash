FROM python:3.12-slim

WORKDIR /app

# Optional build argument to install TimesFM-3 (defaults to false for license compliance)
ARG INSTALL_TIMESFM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-timesfm.txt ./

# Install standard dependencies by default (without TimesFM-3).
# To intentionally install TimesFM-3, build with: --build-arg INSTALL_TIMESFM=true
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt \
 && if [ "$INSTALL_TIMESFM" = "true" ]; then pip install --no-cache-dir -r requirements-timesfm.txt; fi

COPY . .

# Hugging Face model weights cache directory
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
