# Multi-stage build.
#
# The legacy Part_2 image was a single stage on `python:3.9` (the full image,
# ~1 GB base) that copied the whole source tree including data, ran pip as
# root, and baked SAS tokens into the layers via the source files.
#
# This one: slim base, wheels built in a throwaway stage, non-root runtime,
# no data and no credentials in any layer.

# ---------------------------------------------------------------- builder
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install ".[app]"

# Bake NLTK corpora into the image so the container never downloads at
# startup. Artefacts (models, index) are deliberately NOT baked -- they are
# fetched from blob storage at runtime, which keeps the image small and lets
# a retrain roll out without a rebuild.
RUN /opt/venv/bin/python -m medsearch.runtime --download-nltk

# ---------------------------------------------------------------- runtime
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="medsearch" \
      org.opencontainers.image.description="Semantic search over COVID-19 clinical trials" \
      org.opencontainers.image.source="https://github.com/<owner>/clinical-trial-search"

# Pin BLAS threads in the image too. A container scheduled onto a many-core
# host would otherwise spawn one thread per host core inside a 1-2 vCPU
# limit and spend its budget on context switches (ADR-008).
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    NLTK_DATA=/home/app/nltk_data \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN useradd --create-home --uid 10001 app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /root/nltk_data /home/app/nltk_data

WORKDIR /home/app
COPY --chown=app:app src ./src
COPY --chown=app:app .streamlit ./.streamlit

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "src/medsearch/app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]


# ------------------------------------------------------------- standalone
# Everything above assumes artefacts arrive at runtime -- a compose volume
# locally, an Azure Storage mount in App Service. Both need somewhere to mount
# *from*, and App Service Free (F1) gives you no storage account.
#
# This target bakes them in instead:
#
#     docker build --target standalone -t medsearch:standalone -f deploy/docker/Dockerfile .
#
# The trade is explicit: a retrain now needs an image rebuild, and the image
# grows by ~110 MB. In exchange the container has no external dependency at
# all, which is the difference between deployable and not on the free tier.
#
# Why all four paths -- none is optional at serve time:
#   models/          the .kv vectors and metadata
#   data/processed/  the document index (ADR-002)
#   data/raw/        the corpus; results render trial titles and abstracts
#   data/interim/    the token cache. Without it the TF-IDF half of the union
#                    re-preprocesses 10,666 documents on first query -- 46 s
#                    out of a 60-minute daily CPU budget, on every cold start,
#                    and F1 has no Always On so cold starts are routine.
FROM runtime AS standalone

COPY --chown=app:app models ./models
COPY --chown=app:app data/raw ./data/raw
COPY --chown=app:app data/interim ./data/interim
COPY --chown=app:app data/processed ./data/processed
