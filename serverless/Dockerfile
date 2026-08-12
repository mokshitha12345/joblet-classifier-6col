# Runpod serverless image for the six-column classifier (Tier 2).
#
# The model .pkl files and the prediction code are BAKED IN, so a cold start
# only loads the encoder (which Runpod keeps warm between calls). No download of
# models at runtime, no dependency on external storage for the weights.
#
# Build & push (see DEPLOY.md):
#   docker build -t <registry>/joblet-6col:latest -f serverless/Dockerfile .
#   docker push <registry>/joblet-6col:latest
# then point a Runpod serverless endpoint at that image.

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt runpod

# the classifier: model weights + prediction code + guardrails
COPY models/ ./models/
COPY code/ ./code/
COPY serverless/handler.py ./serverless/handler.py

# Pre-download the encoder at BUILD time so the first live call is fast.
# (Same model id every head carries; downloading once bakes ~2 GB into the image.)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('intfloat/multilingual-e5-large')"

# SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are injected as Runpod endpoint
# secrets at runtime -- never baked into the image.
CMD ["python", "-u", "serverless/handler.py"]
