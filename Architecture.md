# Architecture — Medical Embeddings Search

> Version: `1.1.0` · Last updated: `2026-08-27`
> Companion to [PRD.md](./PRD.md). Requirement ids (`F-xx`) refer to that document.

---

## 1. System context

```
                    ┌──────────────────────────────────────────┐
   Dimensions       │                                          │
   COVID-19    ────▶│  data/raw/dimension-covid.csv            │
   CSV drop         │                                          │
                    └───────────────────┬──────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
                    │  DATA LAYER      medsearch.data          │
                    │  load → validate → hash → cache          │
                    └───────────────────┬──────────────────────┘
                                        │  CorpusFrame
                    ┌───────────────────▼──────────────────────┐
                    │  PREPROCESSING   medsearch.preprocessing │
                    │  normalise → tokenize → stop → lemmatize │
                    └───────────────────┬──────────────────────┘
                                        │  Iterable[list[str]]  (streamed)
                    ┌───────────────────▼──────────────────────┐
                    │  EMBEDDING       medsearch.embeddings    │
                    │  Skip-gram | FastText  →  KeyedVectors   │
                    │  mean-pool → DocumentIndex (float32)     │
                    └───────────────────┬──────────────────────┘
                                        │  models/*.kv + index/*.npy
                    ┌───────────────────▼──────────────────────┐
                    │  SEARCH          medsearch.search        │
                    │  query → vector → normalised matmul      │
                    │  → argpartition top-n → SearchResult     │
                    └───────────────────┬──────────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
      ┌───────────────┐         ┌───────────────┐        ┌───────────────┐
      │ CLI           │         │ Streamlit UI  │        │ Python API    │
      │ medsearch …   │         │ :8501         │        │ import        │
      └───────────────┘         └───────────────┘        └───────────────┘
```

## 2. Layered design

Every module depends **only downward**. A layer never imports from a layer above it.

Listed highest to lowest, matching the `import-linter` contract exactly:

| Layer | Modules | Responsibility |
|-------|---------|----------------|
| L7 Interface | `app` \| `cli` | Presentation and command surface |
| L6 Orchestration | `pipelines` | Train, evaluate, verify artefact integrity |
| L5 Search | `search` | Index persistence, ranking, TF-IDF baseline |
| L4 Embeddings | `embeddings` | Train models, build document vectors |
| L3 Domain | `preprocessing` \| `data` | Text transforms; load, validate, cache |
| L2 Logging | `logging_conf` | Structured logs, stage timing + RSS |
| L1 Runtime | `runtime` | Thread pinning, memory probes, preflight |
| L0 Primitives | `exceptions` \| `config` \| `_typing` | Errors, settings, type aliases |

Modules joined by `|` are **siblings in one layer and may not import each other**.
That rule is why the foundation is four layers rather than one: `runtime` imports
`exceptions`, and `logging_conf` imports `runtime`. Both are correct and intended,
and a single flat foundation layer rejected both.

**Enforcement:** two `import-linter` contracts in `pyproject.toml`, checked in CI
and by a pre-commit hook. A violation fails the build — this is the single rule
that stops the codebase collapsing back into the legacy "everything imports
everything" shape.

## 3. Folder structure

```
medical-embeddings-search/
├── PRD.md                      # what & why
├── Architecture.md             # this file — how
├── Rules.md                    # engineering boundaries
├── Phases.md                   # sprint plan
├── Memory.md                   # running progress log
├── README.md                   # human quickstart
├── pyproject.toml              # deps, tooling, packaging, lint contracts
├── Makefile                    # setup / doctor / data / train / app / test
├── .gitignore
├── .dockerignore
├── .env.example                # every env var, no values
├── .pre-commit-config.yaml
├── .streamlit/
│   └── config.toml             # watcher off, headless, theme
├── .github/
│   └── workflows/
│       ├── ci.yml              # lint → type → test → contract → secret-scan
│       └── release.yml         # build & push container image
│
├── src/
│   └── medsearch/
│       ├── __init__.py         # __version__, public re-exports
│       ├── config.py           # Settings (pydantic-settings), Paths
│       ├── runtime.py          # thread pinning, RSS probe, doctor checks
│       ├── logging_conf.py     # structured logging setup
│       ├── exceptions.py       # MedSearchError hierarchy
│       ├── _typing.py          # FloatArray/IntArray aliases, WordVectors Protocol
│       ├── cli.py              # typer app — the only argparse surface
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── schema.py       # CorpusSchema — column names, dtypes, required set
│       │   └── loader.py       # load_corpus(), corpus_fingerprint()
│       │
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── normalizer.py   # compiled regexes, clean_text()
│       │   └── pipeline.py     # TextPreprocessor, preprocess_corpus()
│       │
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── base.py         # EmbeddingModel protocol, ModelMetadata
│       │   ├── trainer.py      # train_skipgram(), train_fasttext()
│       │   ├── registry.py     # save/load artefacts + metadata sidecar
│       │   ├── document.py     # DocumentEmbedder — mean or SIF pooling
│       │   └── weighting.py    # SIF weights, principal component (ADR-011)
│       │
│       ├── search/
│       │   ├── __init__.py
│       │   ├── index.py        # DocumentIndex — .npy persistence, mmap
│       │   ├── engine.py       # SearchEngine, SearchResult
│       │   └── baseline.py     # TF-IDF keyword baseline (scipy.sparse)
│       │
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── train.py        # end-to-end training orchestration
│       │   ├── evaluate.py     # Recall@k, MRR@k, latency, target check
│       │   └── integrity.py    # fingerprint checks across model/index/corpus
│       │
│       └── app/
│           ├── __init__.py
│           ├── streamlit_app.py    # entrypoint: flow only
│           └── components.py       # sidebar, results table, abstracts, banners
│
├── tests/
│   ├── conftest.py             # fixtures: corpus copies, toy vectors, stub preprocessor
│   ├── fixtures/
│   │   ├── sample_corpus.csv   # 20 rows, committed, six topic clusters
│   │   └── eval_queries.json   # NOT YET WRITTEN — Sprint 8, see note below
│   ├── unit/
│   │   ├── test_config.py      test_schema.py     test_loader.py
│   │   ├── test_normalizer.py  test_preprocessing.py
│   │   ├── test_document.py    test_registry.py
│   │   ├── test_index.py       test_engine.py
│   │   ├── test_runtime.py     test_cli.py
│   │   └── test_regressions.py # one test per defect the legacy code shipped
│   └── integration/
│       └── test_train_pipeline.py   # marked slow: trains real models
│
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile          # multi-stage, non-root, slim
│   │   └── compose.yaml        # local run under a 2 GB limit
│   ├── azure/
│   │   ├── data-factory/
│   │   │   ├── linked-services/   # managed-identity only, no sasUri literals
│   │   │   ├── pipelines/
│   │   │   └── triggers/
│   │   ├── databricks/
│   │   │   ├── run_training.py    # replaces the legacy missing `training_model`
│   │   │   └── run_indexing.py    # separate task: reclaims training memory first
│   │   └── app-service/
│   │       └── site-config.json   # managed identity, B2 tier, no secrets
│   └── README.md               # runbook: revoke → provision → deploy → verify
│
├── notebooks/
│   └── 01-exploration.ipynb    # migrated legacy notebook; never imported by src/
│
├── scripts/
│   ├── migrate_legacy.py       # one-shot: legacy artefacts → new layout
│   └── make_eval_candidates.py # pooled candidate sheet for human labelling
│
├── data/                       # gitignored (.gitkeep only)
│   ├── raw/                    # source CSV, immutable
│   ├── interim/                # preprocessed token cache
│   └── processed/              # document index .npy
│
├── models/                     # gitignored (.gitkeep only)
│   ├── skipgram/
│   └── fasttext/
│
└── reports/                    # gitignored — evaluation.json, profiles
```

### 3.1 The one file this tree names but does not contain

`tests/fixtures/eval_queries.json` is the labelled evaluation set. Everything
around it now exists — `pipelines/evaluate.py`, the TF-IDF baseline, and
`scripts/make_eval_candidates.py`, which pools candidates from Skip-gram,
FastText **and** TF-IDF so no single method's biases shape what a labeller
sees. On the real corpus, **35% of pooled candidates were surfaced only by
TF-IDF**; had candidates come from the embedding models alone, the baseline
would have been scored on documents the labeller never saw.

The eval set itself is not something to invent. Relevance judgements are
what every retrieval metric is measured against; fabricating them, or deriving
them from keyword overlap, would produce numbers that look authoritative and
quietly favour the baseline the project is trying to beat. It needs a human who
can read a clinical abstract and say whether it answers the query.

`medsearch evaluate` is registered as a command so the CLI surface is stable,
but it exits 2 with a pointer to Phases.md rather than printing metrics it
cannot compute.

## 4. Naming conventions

| Kind | Convention | Example |
|------|-----------|---------|
| Repository | `kebab-case` | `medical-embeddings-search` |
| Python package / module | `snake_case`, singular unless a collection | `document.py`, `embeddings/` |
| Class | `PascalCase` | `DocumentIndex`, `SearchEngine` |
| Function / variable | `snake_case`, verb-first for functions | `build_index()`, `corpus_fingerprint()` |
| Constant | `UPPER_SNAKE_CASE` | `DEFAULT_VECTOR_SIZE` |
| Private | single leading underscore | `_compile_patterns()` |
| Test | `test_<module>.py` :: `test_<behaviour>_<condition>` | `test_engine_returns_empty_on_oov_query` |
| Data file | `kebab-case`, dated when versioned | `dimension-covid.csv` |
| Artefact dir | `models/<model-name>/` with fixed inner names | `models/skipgram/model.kv` |
| Env var | `MEDSEARCH_` prefix, `UPPER_SNAKE` | `MEDSEARCH_VECTOR_SIZE` |
| Git branch | `<type>/<sprint>-<slug>` | `feat/s3-fasttext-trainer` |
| Doc file | `PascalCase.md` at root for AI-context docs | `Architecture.md` |

**Banned outright** — all present in the legacy code:
`Modular+Code/` (`+` in a path), `Ipython Notebook/` (spaces), `Medical Embeddings_Final.ipynb`
(spaces + mixed case), `K`/`K1`/`K11`/`KK`/`p`/`x`/`tmp`/`res`/`L` as variable names,
and any name that differs from another only by case (`FastText-vec.csv` vs `Fasttext-vec.csv`
— a real Part 1 bug that broke on Linux).

## 5. Tech stack

| Concern | Choice | Why this and not the alternative |
|---------|--------|----------------------------------|
| Language | Python 3.10–3.12 | 3.10 is installed on the dev machine; 3.13 lacks stable gensim wheels |
| Embeddings | `gensim >= 4.3` | Only mature CPU Word2Vec/FastText with a stable `KeyedVectors` serving format |
| Numerics | `numpy >= 1.24` | Single-matmul ranking; `float32` throughout |
| Tabular | `pandas >= 2.1` | `usecols` + `pyarrow` engine keeps the 29 MB CSV under ~90 MB resident |
| NLP utils | `nltk` | Stopwords + WordNet lemmatizer; already the corpus's known-good transform |
| CLI | `typer` | Type-hint driven, no hand-rolled argparse |
| Config | `pydantic-settings` | Env-var overrides + validation in one declaration |
| UI | `streamlit >= 1.30` | Existing investment; `st.cache_resource` solves the reload problem |
| Tables in UI | `st.dataframe` | Replaces Plotly `go.Table` — Plotly serialised all 10k abstracts to JSON per rerun |
| Tests | `pytest`, `pytest-cov` | Standard |
| Lint / format | `ruff` | Replaces flake8 + isort + black in one fast pass |
| Types | `mypy --strict` on `src/medsearch` | Catches the `K1` vs `K2` class of bug statically |
| Layering | `import-linter` | Enforces §2 mechanically |
| Container | Docker, `python:3.11-slim`, multi-stage | Legacy image was `python:3.9` full — ~1 GB base |
| Cloud | Azure Blob + Data Factory + Databricks + App Service | Already provisioned |
| Secrets | Azure Key Vault + managed identity | Legacy embedded live SAS tokens in source |

**Deliberately rejected:** `plotly` for the results table (JSON bloat per rerun),
`scikit-learn` as a runtime dep (only needed for optional PCA — moved to an extra),
`FAISS` (overkill at 10k docs), `poetry` (PEP 621 + `pip` is enough here).

## 6. Core module contracts

```python
# medsearch/data/loader.py
def load_corpus(path: Path, *, limit: int | None = None) -> pd.DataFrame: ...
def corpus_fingerprint(path: Path) -> str: ...          # sha256 of bytes, 16-char prefix

# medsearch/preprocessing/pipeline.py
class TextPreprocessor:
    def transform(self, text: str) -> list[str]: ...     # pure, no state mutation
    def transform_many(self, texts: Iterable[str]) -> Iterator[list[str]]: ...  # streams

# medsearch/embeddings/trainer.py
def train_model(
    corpus: Iterable[list[str]], *, kind: ModelKind, params: TrainingParams
) -> tuple[KeyedVectors, ModelMetadata]: ...

# medsearch/embeddings/document.py
class DocumentEmbedder:
    def __init__(self, vectors: KeyedVectors) -> None: ...  # builds vocab set ONCE
    def embed(self, tokens: Sequence[str]) -> np.ndarray: ...          # (dim,) float32
    def embed_corpus(self, docs: Iterable[Sequence[str]]) -> np.ndarray: ...  # (n, dim)

# medsearch/search/index.py
@dataclass(frozen=True, slots=True)
class DocumentIndex:
    vectors: np.ndarray          # (n, dim) float32, L2-normalised at build time
    row_ids: np.ndarray          # (n,) int64 — maps back to the corpus frame
    model_fingerprint: str
    def save(self, directory: Path) -> None: ...
    @classmethod
    def load(cls, directory: Path, *, mmap: bool = True) -> "DocumentIndex": ...

# medsearch/search/engine.py
class SearchEngine:
    def search(self, query: str, *, top_n: int = 10) -> list[SearchResult]: ...
```

## 7. Key design decisions (ADRs)

**ADR-001 — Bound the FastText `bucket`.**
gensim's default is `bucket=2_000_000`. At 100 dims × `float32` that is exactly
`2e6 × 100 × 4 = 800 MB`, which is precisely the size of the legacy
`model_Fasttext.bin.wv.vectors_ngrams.npy`. We set `bucket=50_000` → 20 MB, a 40×
reduction. With `min_n=3, max_n=5` over a ~30k-word medical vocabulary, collision rate
stays acceptable. *Consequence:* artefacts fit in a container image and in RAM.

**ADR-002 — Persist the index as `.npy`, not CSV.**
Legacy stored 10,666 × 100 floats as a 21 MB CSV, re-parsed on every process start.
`float32` `.npy` is 4.3 MB, loads by `mmap` in milliseconds, and is exact.

**ADR-003 — Pre-normalise the index; rank with one matmul.**
Legacy ran a Python loop of 10,666 `cos_sim()` calls per query. Storing L2-normalised
rows reduces cosine similarity to `index.vectors @ q_hat`, a single BLAS call — three
orders of magnitude faster and the difference between a laggy and an instant UI.

**ADR-004 — Build the vocabulary set once, in the constructor.**
Legacy called `list(word2vec_model.wv.index_to_key)` *inside* `get_mean_vector`, i.e.
once per document — ~30k-element list construction × 10,666 documents. `DocumentEmbedder`
builds a `frozenset` in `__init__` and does O(1) membership tests.

**ADR-005 — Stream the corpus into the trainer.**
gensim accepts any re-iterable. Materialising all token lists costs ~600 MB on this
corpus; a generator-backed `CorpusStream` keeps it near zero at the cost of a second pass.

**ADR-006 — Serve `KeyedVectors`, not the full model.**
`model.save()` persists `syn1neg`/trainable state needed only to *continue* training.
`model.wv.save()` halves both artefact size and serving RSS. Training state is kept only
under `models/<kind>/checkpoint/` and is gitignored.

**ADR-007 — `--limit` defaults to `None`.**
The legacy `read_data()` hardcoded `df.iloc[:100, :]`, so every "production" artefact was
trained on 100 of 10,666 rows. Sampling is now explicit, opt-in, and logged as a warning.

**ADR-008 — Pin BLAS/OpenMP threads at import.**
numpy + gensim each spawn thread pools sized to core count; combined with `workers=N`
they oversubscribe a 4-thread CPU. `runtime.configure_threads()` sets `OMP_NUM_THREADS`
and friends **before** numpy is imported.

**ADR-009 — `st.dataframe` over Plotly `go.Table`.**
The legacy UI serialised every returned abstract into a Plotly JSON payload on each
rerun. Native `st.dataframe` renders the same data without the round-trip.

**ADR-010 — Refuse a stale index rather than serve it.**
An index records the fingerprint of the corpus it was built from, but nothing
compared that against the *live* corpus file until Sprint 11. Because row ids
are positional, a stale index does not fail — it resolves to the wrong
documents, pairing one trial's title with another trial's score. That is worse
than an outright error, because nothing looks broken. `load_search_engine` now
raises `StaleIndexError`, and `medsearch doctor --full` reports it alongside
the other two mismatch classes. *Consequence:* replacing the corpus without
retraining is a loud failure. This matters most in the Azure path, where the
pipeline is triggered by a CSV drop.

**ADR-011 — SIF weighting is implemented, selectable, and off by default.**
The n=97 evaluation showed TF-IDF beating both embedding models, and the
diagnosis blamed mean pooling. SIF (Arora et al. 2017) was implemented to test
that: frequency weighting plus common-component removal, in
`embeddings/weighting.py`, selected by `--pooling sif` and stored in a separate
index directory so both can be compared without rebuilding either.

**It was measured and it does not work** — no gain for Skip-gram, a significant
loss for FastText (PRD §8.2). The code stays rather than being reverted, for
three reasons: the negative result should be reproducible; the ablation
separating frequency weighting from component removal is reusable; and a future
change to dimensionality or vocabulary could plausibly change the answer.
*Consequence:* `pooling` defaults to `"mean"`, so nothing changes for anyone
who does not opt in, and mean-pooled indexes built before ADR-011 still load.

## 8. Data flow — training run

```
load_corpus(usecols=[title, abstract, trial_id, publication_date], limit=None)
      │  ~90 MB resident (not 700 MB — 4 of 21 columns)
      ▼
corpus_fingerprint()  ──▶  data/interim/<fingerprint>.tokens.jsonl   (cache hit? skip)
      ▼
TextPreprocessor.transform_many()          streamed, generator
      ▼
train_model(kind=skipgram, workers=3, seed=42)   ──▶ models/skipgram/model.kv
train_model(kind=fasttext, bucket=50_000)        ──▶ models/fasttext/model.kv
      │                                              + metadata.json (hparams, hashes, size)
      ▼
DocumentEmbedder(vectors).embed_corpus()   float32, chunked, L2-normalised
      ▼
DocumentIndex.save()  ──▶  data/processed/<kind>-<field>/vectors.npy + manifest.json
```

## 9. Resource budget

**Measured 2026-08-27** on the target machine (4 logical cores, 7.89 GB RAM,
3.8 GB free) over the full 10,666-document corpus. These replace the estimates
this section previously carried.

| Stage | Wall time | Peak RSS | Output |
|-------|-----------|----------|--------|
| `load_corpus` | 0.4 s | 318 MB | 10,666 rows, 4 of 21 columns |
| `preprocess` (cold) | 46.3 s | 313 MB | 22 MB token cache |
| `preprocess` (warm) | 0.0 s | 318 MB | cache hit |
| `train_skipgram` | 30.0 s | 319 MB | **10.2 MB** |
| `train_fasttext` | 56.0 s | 346 MB | **29.3 MB** |
| `build_index` (each) | ~7 s | 152 MB | **4.1 MB** |
| **Full `make train`** | **2 min 22 s** | **≤ 350 MB** | 59 MB total |
| Serving (engine + index + model) | 2.96 s cold load | **342 MB** | — |

Vocabulary: 24,897 words. Out-of-vocabulary documents: **2 of 10,666 (0.02%)**.

### Against the stated budget

| Target | Budget | Measured | |
|--------|--------|----------|---|
| Full pipeline peak RSS | ≤ 2.5 GB | **~350 MB** | 7x headroom |
| Serving RSS | ≤ 1.2 GB | **342 MB** | 3.5x headroom |
| Training wall time | ≤ 15 min | **2 min 22 s** | 6x faster |
| Any single artefact | ≤ 150 MB | **29.3 MB** | 5x headroom |
| `data/` + `models/` total | ≤ 1.5 GB | **59 MB** | 25x headroom |
| Query latency p95 (PRD G2) | < 300 ms | **3.3 ms** embedding · **128 ms** union | 2.3x faster |

Latency measured over 120 queries after warm-up: p50 1.4 ms, p95 3.3 ms,
p99 10.7 ms, max 35.8 ms. That is ADR-003 -- one BLAS matrix-vector product
against a pre-normalised index -- rather than the legacy per-document Python
loop.

**Union retrieval costs 35x that and is still inside budget.** The shipped
default queries both retrievers, so each search adds a sparse matrix product
over 40,012 TF-IDF terms: p95 **128 ms** against the 300 ms target, measured
over the 97 eval queries. The headroom drops from 90x to 2.3x, which is the
one place union retrieval materially spends a budget rather than saving one.
Worth watching if the corpus grows -- the TF-IDF side scales with vocabulary,
the embedding side does not.

### Artefact size against the legacy project

| Artefact | Legacy | v1 | |
|----------|--------|-----|---|
| FastText model | 762.9 MB | **29.3 MB** | 26x smaller |
| Skip-gram doc vectors | 20.7 MB | **4.2 MB** | 5x smaller |
| FastText doc vectors | 20.6 MB | **4.2 MB** | 5x smaller |
| **Total** | **804.3 MB** | **37.6 MB** | **21x smaller** |

ADR-001 (bounded bucket) accounts for the model reduction; ADR-002
(`float32` `.npy` instead of CSV) for the vector reduction.

**Guard rails**
- `medsearch doctor` refuses to start training below **2.0 GB free RAM** and warns below 3.0 GB. The floor scales down for a `--limit` run (`Settings.memory_floor_gb`).
- `runtime.configure_threads()` pins `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` so only gensim's `workers` are parallel.
- Stages are separate CLI invocations, so the OS reclaims each stage's memory before the next begins.
- `--limit 2000` remains the documented fallback for a constrained session (~16 s per model, ~180 MB).
- `.streamlit/config.toml` sets `fileWatcherType = "none"`.

**Why the measurements beat the estimates so widely:** the estimates were
written before the bucket bound and the streaming token cache were built, and
assumed gensim's memory profile scaled with corpus size more steeply than it
does at this vocabulary. The headroom is real, and it means the full corpus
now trains comfortably on the dev laptop rather than needing Databricks.

## 10. Configuration & secrets

Single source of truth: `medsearch.config.Settings`, populated from defaults → `.env` →
environment. Nothing else reads `os.environ` directly.

```
MEDSEARCH_DATA_DIR          MEDSEARCH_VECTOR_SIZE       MEDSEARCH_WORKERS
MEDSEARCH_MODEL_DIR         MEDSEARCH_WINDOW            MEDSEARCH_SEED
MEDSEARCH_LOG_LEVEL         MEDSEARCH_MIN_COUNT         MEDSEARCH_FASTTEXT_BUCKET
MEDSEARCH_TOP_N             MEDSEARCH_EPOCHS            MEDSEARCH_MAX_MEMORY_GB
```

Secrets are **never** in this list and never in a file under version control.
Locally: `.env` (gitignored). In Azure: Key Vault referenced by App Service /
Databricks managed identity. `.env.example` documents every key with an empty value.

## 11. Deployment topology

```
  New CSV lands in Blob container
            │
            ▼
  ADF BlobEventsTrigger  ──▶  ADF pipeline `train-embeddings`
            │
            ▼
  Databricks job (wheel task: medsearch.pipelines.train)
            │  writes models/ + index/ back to Blob
            ▼
  Azure App Service (container) pulls artefacts at cold start
            │
            ▼
  Streamlit UI :8501  ◀── users
```

Auth is managed identity end to end. The container carries **no** credentials; it
resolves a Blob URL via `DefaultAzureCredential` at startup.

## 12. Migration from legacy

`scripts/migrate_legacy.py` performs the one-shot move (idempotent, dry-run by default):

| Legacy | New | Action |
|--------|-----|--------|
| `Part_1/Data/Data/Dimension-covid.csv` | `data/raw/dimension-covid.csv` | move, rename |
| `Part_1/.../output/model_*.bin` | — | **discard** — retrained under ADR-001/006 |
| `Part_1/.../output/*.npy` (800 MB) | — | **discard** — superseded by bounded bucket |
| `Part_1/.../output/*-vec-*.csv` | — | **discard** — rebuilt as `.npy` (ADR-002) |
| `Part_1/Ipython Notebook/*.ipynb` | `notebooks/01-exploration.ipynb` | move, rename, strip outputs |
| `Part_1/src/ML_pipeline/*.py` | `src/medsearch/**` | rewritten, not copied |
| `Part_2/.../src/*.py` | `src/medsearch/**` + `deploy/` | rewritten; SAS tokens removed |
| `Part_2/.../{pipeline,trigger,linkedService}/*.json` | `deploy/azure/data-factory/**` | rewritten with managed identity |

Discarding the legacy artefacts reclaims **~860 MB**.

---

**Change log**

| Date | Version | Change |
|------|---------|--------|
| 2026-08-27 | 1.0.0 | Initial architecture |
| 2026-08-27 | 1.1.0 | Added §9 resource budget, ADR-001/003/004/005/008/009 after laptop profiling |
