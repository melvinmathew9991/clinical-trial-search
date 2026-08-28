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

Five layers, each depending **only downward**. A layer never imports from a layer above it.

| Layer | Package | Responsibility | May import |
|-------|---------|----------------|------------|
| L0 Foundation | `config`, `logging_conf`, `runtime`, `exceptions` | Settings, structured logs, thread pinning, error types | stdlib only |
| L1 Data | `medsearch.data` | Load, validate, hash, cache the corpus | L0 |
| L2 Preprocessing | `medsearch.preprocessing` | Pure text transforms | L0 |
| L3 Embeddings | `medsearch.embeddings` | Train models, build document vectors | L0, L1, L2 |
| L4 Search | `medsearch.search` | Index persistence, ranking | L0, L1, L3 |
| L5 Interface | `cli`, `app`, `pipelines` | Orchestration and presentation | all |

**Enforcement:** `import-linter` contract in `pyproject.toml`, checked in CI. A violation
fails the build — this is the single rule that stops the codebase collapsing back into
the legacy "everything imports everything" shape.

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
│       │   └── document.py     # DocumentEmbedder — mean pooling
│       │
│       ├── search/
│       │   ├── __init__.py
│       │   ├── index.py        # DocumentIndex — .npy persistence, mmap
│       │   └── engine.py       # SearchEngine, SearchResult
│       │
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── train.py        # end-to-end training orchestration
│       │   └── evaluate.py     # Recall@k, MRR, latency, peak RSS
│       │
│       └── app/
│           ├── __init__.py
│           ├── streamlit_app.py    # entrypoint
│           └── components.py       # results table, sidebar, state banners
│
├── tests/
│   ├── conftest.py             # tiny in-memory corpus fixture — no disk, no network
│   ├── fixtures/
│   │   ├── sample_corpus.csv   # 20 rows, committed
│   │   └── eval_queries.json   # labelled query → relevant ids
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_schema.py
│   │   ├── test_loader.py
│   │   ├── test_normalizer.py
│   │   ├── test_document.py
│   │   ├── test_index.py
│   │   └── test_engine.py
│   └── integration/
│       └── test_train_pipeline.py
│
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile          # multi-stage, non-root, slim
│   │   └── compose.yaml
│   ├── azure/
│   │   ├── data-factory/
│   │   │   ├── linked-services/   # managed-identity only, no sasUri literals
│   │   │   ├── pipelines/
│   │   │   └── triggers/
│   │   ├── databricks/
│   │   │   └── job-train.json     # runs medsearch.pipelines.train as a wheel task
│   │   └── app-service/
│   └── README.md               # runbook: provision → deploy → rotate secrets
│
├── notebooks/
│   └── 01-exploration.ipynb    # EDA + PCA only; never imported by src/
│
├── scripts/
│   └── migrate_legacy.py       # one-shot: legacy artefacts → new layout
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

Measured against the target machine: **4 logical cores, 7.89 GB RAM**.

| Stage | Peak RSS | Wall time | Cores used | Notes |
|-------|----------|-----------|-----------|-------|
| `load_corpus` | ~90 MB | 2 s | 1 | `usecols` + pyarrow engine |
| `preprocess` (cold) | ~250 MB | ~4 min | 1 | streamed; cached to `interim/` |
| `preprocess` (warm) | ~40 MB | 3 s | 1 | cache hit |
| `train_skipgram` | ~500 MB | ~3 min | 3 | `workers=3`, one core reserved |
| `train_fasttext` | ~1.1 GB | ~7 min | 3 | dominated by the bounded n-gram matrix |
| `build_index` | ~350 MB | ~40 s | 1 | chunked at 1,000 docs |
| **Full `make train`** | **≤ 2.5 GB** | **≤ 15 min** | 3 | never all stages resident at once |
| Serving (UI + 2 indexes + 2 KV) | ~1.1 GB | — | 1 | mmap index, cached resources |

**Guard rails**
- `medsearch doctor` refuses to start training below **2.0 GB free RAM** and warns below 3.0 GB.
- `runtime.configure_threads()` pins `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` so only gensim's `workers` are parallel.
- Stages are separate CLI invocations, so the OS reclaims each stage's memory before the next.
- `--limit 2000` is the documented fallback profile for a constrained session (~40 s per model).
- `.streamlit/config.toml` sets `fileWatcherType = "none"` — the legacy default polled the whole tree, including the 800 MB artefact.

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
