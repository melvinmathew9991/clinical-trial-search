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
COPY deploy/requirements.lock ./deploy/requirements.lock
COPY src ./src

# Installed against the lock as a *constraint*: pyproject decides what is
# needed, the lock decides which versions, and a genuine conflict fails the
# build instead of silently resolving to something new. Ranges alone meant
# every rebuild re-resolved gensim/scipy/pandas and happened to work.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install ".[app]" --constraint deploy/requirements.lock

# Bake NLTK corpora into the image so the container never downloads at
# startup. Artefacts (models, index) are deliberately NOT baked -- they are
# fetched from blob storage at runtime, which keeps the image small and lets
# a retrain roll out without a rebuild.
# The trim that follows keeps English only. punkt ships sentence models for 17
# languages and omw-1.4 for 30-odd; this pipeline calls word_tokenize on English
# abstracts and nothing else. nltk also leaves the downloaded .zip beside the
# tree it unpacked, so that copy is dead weight -- except for wordnet and
# omw-1.4, which nltk reads *from* the zip and which must stay. ~64 MB.
#
# The package directories themselves must survive: ensure_nltk_data() probes
# with nltk.data.find() and re-downloads anything missing, which inside a
# container means a network call on the first query.
RUN /opt/venv/bin/python -m medsearch.runtime --download-nltk \
    && find /root/nltk_data/tokenizers/punkt -name '*.pickle' ! -name 'english.pickle' -delete \
    && find /root/nltk_data/tokenizers/punkt_tab -mindepth 1 -maxdepth 1 -type d ! -name english -exec rm -rf {} + \
    && rm -f /root/nltk_data/tokenizers/punkt.zip \
             /root/nltk_data/tokenizers/punkt_tab.zip \
             /root/nltk_data/corpora/stopwords.zip

# Trim the virtualenv before the runtime stage copies it. Every figure here was
# measured inside this image, not estimated:
#
#   bundled test suites   132 MB   scipy/numpy/pandas ship their pytest suites
#   pip + setuptools       25 MB   nothing installs anything at runtime
#   pyarrow C++ headers     6 MB   for building against Arrow, not importing it
#   272 unstripped .so    341 MB   debug symbols a serving process never reads
#
# Phases.md recorded that the remaining 141 MB over the DoD was 'not reachable
# by packaging' and that meeting it 'means changing the UI stack'. That was
# written without measuring these: the test suites alone very nearly cover it.
#
# __pycache__ is another 141 MB and is deliberately KEPT. PYTHONDONTWRITEBYTECODE
# is set in the runtime stage, so deleting it would make every container start
# recompile the whole dependency tree -- a repeated cost on App Service F1, which
# has no Always On and therefore cold-starts routinely.
RUN find /opt/venv -type d \
         \( -name tests -o -name test \) -prune -exec rm -rf {} + \
    && rm -rf /opt/venv/lib/python3.11/site-packages/pip \
              /opt/venv/lib/python3.11/site-packages/pkg_resources \
              /opt/venv/lib/python3.11/site-packages/setuptools \
              /opt/venv/lib/python3.11/site-packages/pyarrow/include \
    && find /opt/venv -name '*.so*' -type f -exec strip --strip-unneeded {} + || true

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

# Create the artefact directories *before* anything mounts into them. Docker
# creates a missing mount point as root:root, so `data/` ended up root-owned
# and uid 10001 could not mkdir `data/interim` inside it -- the union
# retriever, which is what ships, died with PermissionError on its first
# query while the healthcheck stayed green. Pre-creating them owned by `app`
# means a bind mount lands on an existing directory and the ownership holds.
RUN mkdir -p /home/app/data/raw /home/app/data/interim /home/app/data/processed \
             /home/app/models \
    && chown -R app:app /home/app/data /home/app/models

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
