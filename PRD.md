# PRD — Medical Embeddings Search

> **Project Requirements Document**
> Status: `Active` · Version: `1.1.0` · Last updated: `2026-08-27`

This document defines **what** we are building and **why**. It does not describe
implementation — see [Architecture.md](./Architecture.md) for that. Boundaries for
how the code gets written live in [Rules.md](./Rules.md). Delivery order lives in
[Phases.md](./Phases.md).

---

## 1. Problem statement

Keyword search fails on clinical literature. A researcher searching `lung failure`
gets zero hits on a trial whose abstract says *"acute respiratory distress syndrome"*,
even though it is the single most relevant record in the corpus. General-purpose
embeddings (GloVe, generic Word2Vec) do not fix this: they are trained on news and
web text where `ARDS`, `comorbidity`, and `seroconversion` are rare or absent.

**The gap:** clinical-trial retrieval needs embeddings trained *on clinical trial text*.

## 2. Product vision

A search service that ranks COVID-19 clinical trials by **semantic proximity** to a
free-text query, using word embeddings trained in-domain on the trial corpus itself,
exposed through a Python API, a CLI, and a web UI, and retrainable on a schedule as
new trials are published — **and able to run end-to-end on a 4-core / 8 GB laptop.**

## 3. Target users

| # | Persona | Need | How they use it |
|---|---------|------|-----------------|
| U1 | **Clinical researcher** | Find prior trials on a condition without knowing the exact terminology used | Web UI, free-text search box |
| U2 | **Data scientist** | Reuse the trained vectors as features for downstream models | Python API — `from medsearch.embeddings import load_model` |
| U3 | **ML engineer** | Retrain on new data, compare model variants, ship to production | CLI + CI/CD + Azure pipeline |
| U4 | **Reviewer / analyst** | Batch-screen a list of topics against the corpus | CLI — `medsearch search --queries topics.txt --json` |

**Primary development machine (a hard constraint, not a footnote):**
Intel i5-7300HQ · 4 physical cores / 4 logical threads · 7.89 GB RAM · SATA-class disk.
Every design decision in this project is evaluated against "does this stay responsive
on that machine?" See §7 and [Architecture.md §9](./Architecture.md#9-resource-budget).

## 4. Goals

| ID | Goal | Measure |
|----|------|---------|
| G1 | Semantic retrieval beats keyword baseline | Recall@10 ≥ 0.70 on the labelled eval set (§8) |
| G2 | Query latency is interactive | p95 < 300 ms for a warm index over 10,666 docs |
| G3 | Retraining is one command, reproducible | `medsearch train --all` yields identical artefacts for a fixed seed |
| G4 | The corpus is fully indexed | 10,666 / 10,666 documents embedded — **not** a 100-row sample |
| G5 | Zero secrets in source control | CI secret scan passes on every commit |
| **G6** | **The dev machine stays usable during a full training run** | **Peak RSS ≤ 2.5 GB, ≥ 1 core left idle, no swap storm** |

## 5. Non-goals

Explicitly **out of scope** for v1. Do not build these; do not let scope drift into them.

- ❌ Transformer models (BERT / BioBERT / sentence-transformers). v1 is Word2Vec + FastText by design — it is the comparison baseline, and a transformer will not fit the memory budget in G6.
- ❌ Multi-lingual support. English abstracts only.
- ❌ User accounts, auth, saved searches, or any persistence of user data.
- ❌ A vector database (FAISS / Pinecone / pgvector). Brute-force cosine over 10,666 × 100 `float32` is 4.3 MB and resolves in a single matmul. Revisit above ~1 M documents.
- ❌ Live ingestion from the Dimensions API. Input is a CSV drop.
- ❌ GPU acceleration. The target machine has no usable CUDA device.
- ❌ Ranking by anything other than semantic similarity (no recency boost, no citation weighting) in v1.

## 6. Functional requirements

Priority: **P0** = v1 blocker · **P1** = v1 target · **P2** = post-v1.

### Data
| ID | Requirement | Priority |
|----|-------------|----------|
| F-01 | Load the clinical-trial corpus from a local CSV path or a URI | P0 |
| F-02 | Validate the loaded frame against a declared schema; fail loudly on a missing required column | P0 |
| F-03 | Read **only** the columns the pipeline needs (`usecols`), never all 21 | P0 |
| F-04 | Support an explicit `--limit N` row cap for fast local iteration, **off by default** | P0 |
| F-05 | Cache the preprocessed corpus to `data/processed/` and reuse it when the source hash is unchanged | P0 |
| F-06 | Stream the corpus to the trainer as an iterable, never materialise all token lists at once | P0 |

### Preprocessing
| ID | Requirement | Priority |
|----|-------------|----------|
| F-07 | Normalise text: lowercase, strip URLs, digits, punctuation, and newlines | P0 |
| F-08 | Tokenize, remove English stopwords, lemmatize | P0 |
| F-09 | Apply the **identical** transform to a corpus document and to a user query | P0 |
| F-10 | Preprocessing is a pure function — no mutation of the caller's DataFrame | P0 |
| F-11 | Compile every regex once at module import, not per call | P0 |
| F-12 | Support a domain allowlist so negation terms (`no`, `not`) survive stopword removal | P2 |

### Embeddings
| ID | Requirement | Priority |
|----|-------------|----------|
| F-13 | Train a Skip-gram Word2Vec model with configurable `vector_size`, `window`, `min_count`, `epochs`, `seed` | P0 |
| F-14 | Train a FastText model with configurable `min_n` / `max_n` char n-grams **and an explicit bounded `bucket`** | P0 |
| F-15 | Persist each model with a sidecar `metadata.json` recording hyperparameters, corpus hash, gensim version, artefact size, and training duration | P0 |
| F-16 | Compose a document vector as the mean of its in-vocabulary word vectors, in `float32` | P0 |
| F-17 | Return a zero vector for a document with no in-vocabulary tokens, and count it | P0 |
| F-18 | Resolve vocabulary membership through an O(1) set built **once**, never a list rebuilt per document | P0 |
| F-19 | Save a serving-only artefact (`KeyedVectors`) that drops optimiser/trainable state | P0 |
| F-20 | Expose `most_similar(word)` for qualitative inspection | P1 |
| F-21 | Support PCA projection export for embedding-space visualisation | P2 |

### Search
| ID | Requirement | Priority |
|----|-------------|----------|
| F-22 | Rank documents by cosine similarity to the query vector | P0 |
| F-23 | Compute all similarities as **one matrix–vector product** over an L2-pre-normalised index | P0 |
| F-24 | Return the top *n* results (default 10) with `trial_id`, `title`, `abstract`, `publication_date`, `score` | P0 |
| F-25 | Search over either the `abstract` or the `title` field, selectable at query time | P0 |
| F-26 | Handle an out-of-vocabulary query gracefully — empty result set with a reason, never a crash or a NaN | P0 |
| F-27 | Persist the index as `.npy` (binary `float32`), memory-mappable, **not** CSV | P0 |
| F-28 | Reject an index whose model fingerprint does not match the loaded model | P1 |
| F-29 | Select top *n* with `argpartition` (O(n)), not a full sort | P1 |

### Interfaces
| ID | Requirement | Priority |
|----|-------------|----------|
| F-30 | CLI: `medsearch train`, `index build`, `search`, `evaluate`, `doctor` | P0 |
| F-31 | Streamlit UI: model selector, field selector, search box, results table with scores | P0 |
| F-32 | UI loads models and index once via `st.cache_resource` and never on every rerun | P0 |
| F-33 | UI surfaces "no results" and error states as readable messages, not tracebacks | P1 |
| F-34 | `medsearch doctor` reports free RAM, core count, artefact sizes, and refuses/warns before an over-budget run | P0 |
| F-35 | REST API (`FastAPI`) exposing `/search` | P2 |

### Operations
| ID | Requirement | Priority |
|----|-------------|----------|
| F-36 | All configuration via a `Settings` object, overridable by environment variables | P0 |
| F-37 | Structured logging at every pipeline stage boundary, including elapsed time and peak RSS | P0 |
| F-38 | Container image runs the UI with no build-time secrets | P0 |
| F-39 | Retraining triggers automatically when a new CSV lands in blob storage | P1 |
| F-40 | Credentials resolved at runtime from Key Vault / managed identity — never literals | P0 |
| F-41 | BLAS/OpenMP thread counts pinned at process start to prevent oversubscription | P0 |

## 7. Non-functional requirements

| Area | Requirement |
|------|-------------|
| **Peak memory** | Full pipeline peak RSS ≤ **2.5 GB**. Serving (UI + index + models) ≤ **1.2 GB**. |
| **CPU courtesy** | Training uses `max(1, cores - 1)` = **3 workers**. At least one logical core stays free so the desktop never freezes. |
| **Artefact size** | Any single model artefact ≤ **150 MB**. FastText `bucket` bounded to enforce this. |
| **Training time** | Full run (10,666 docs, both models) ≤ 15 min on the target laptop. |
| **Disk** | Total `data/` + `models/` footprint ≤ 1.5 GB. Repo working tree (code only) ≤ 5 MB. |
| **Reproducibility** | Fixed `seed` + `workers=1` produces identical vectors; hyperparameters recorded in metadata. |
| **Portability** | Runs on Windows and Linux — no OS-specific paths, all path joins via `pathlib`. |
| **Testability** | ≥ 80 % line coverage on `src/medsearch`, excluding the Streamlit layer. |
| **Security** | No credential, SAS token, subscription id, or storage account name committed. |
| **Compatibility** | Python 3.10–3.12, gensim ≥ 4.3, numpy ≥ 1.24. |

### 7.1 Why these numbers

The legacy implementation violated every one of them, and the arithmetic is exact:

| Legacy behaviour | Cost on this laptop | v1 requirement |
|------------------|---------------------|----------------|
| gensim default `bucket=2_000_000` | `2e6 × 100 dims × 4 B` = **800 MB** on disk, and the same again resident when loaded | F-14 — bounded bucket |
| `workers=5` on a 4-thread CPU | 5 training threads + BLAS threads contend for 4 cores → context-thrash, UI freeze | F-41 + 3 workers |
| `list(model.wv.index_to_key)` called **inside** the per-document loop | Rebuilds a ~30k-element Python list 10,666 times ≈ 320 M allocations | F-18 — build the set once |
| Python `for` loop of 10,666 `cos_sim()` calls per query | ~10,666 separate `dot`/`norm` calls per keystroke | F-23 — single matmul |
| Doc vectors stored as CSV text | 21 MB file, full float re-parse on every start | F-27 — `.npy` |
| `pd.read_csv` of all 21 object columns | ~700 MB–1 GB resident for a 29 MB file | F-03 — `usecols` |
| `df[col][i] = ...` chained assignment across 10,666 rows | Quadratic-ish copying, pandas `SettingWithCopyWarning` | F-10 — pure functions |

## 8. Success metrics & evaluation

A held-out eval set of query → relevant-trial-id pairs lives at `tests/fixtures/eval_queries.json`.

| Metric | Baseline (TF-IDF) | v1 target |
|--------|-------------------|-----------|
| Recall@10 | measure first | ≥ 0.70 |
| MRR@10 | measure first | ≥ 0.45 |
| p95 query latency | — | < 300 ms |
| OOV query rate | — | < 5 % |
| Peak training RSS | — | ≤ 2.5 GB |

`medsearch evaluate` computes these and writes `reports/evaluation.json`.

## 9. Constraints & assumptions

- **Corpus is static per training run.** 10,666 trials × 21 columns, ~29 MB CSV. Only `Title` and `Abstract` carry signal; four more columns are display metadata; the other 15 are never read.
- **Mean-pooled word vectors are a known-weak document representation.** Accepted for v1 as the stated learning objective; SIF weighting is the first upgrade candidate (P2).
- **Only ~1 GB of RAM is typically free** on the dev machine at session start. The pipeline must therefore be *streaming-first*, not *load-everything-first*.
- Azure is the target cloud because the existing infrastructure (Blob, Data Factory, Databricks, App Service) is already provisioned. Heavy full-corpus training is expected to move to Databricks; the laptop path stays viable via `--limit` and the bounded bucket.

## 10. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mean pooling washes out long abstracts | Poor Recall@10 | Measure against TF-IDF baseline in Sprint 8; SIF weighting as fallback |
| Legacy Azure SAS tokens were committed **with write+delete rights** (`sp=racwdymeop`) | Storage compromise if the account still exists | Rotate/revoke at account level; v1 uses managed identity only |
| Full FastText training still exceeds the RAM budget | Dev machine swaps, run takes hours | `medsearch doctor` preflight; documented `--limit` fallback; Databricks path for full runs |
| Streamlit file-watcher polls the 1.5 GB `data/` tree | Constant background CPU burn | Watcher disabled and `data/`/`models/` excluded in `.streamlit/config.toml` |
| Developer runs training with a browser + IDE open on 8 GB | OOM kill mid-run | `doctor` warns below a free-RAM floor before training starts |

## 11. Acceptance

v1 is done when a new engineer can run:

```bash
git clone <repo> && cd medical-embeddings-search
make setup
make doctor        # preflight: cores, free RAM, disk, artefact budget
make train         # both models + index, ≤ 15 min, ≤ 2.5 GB peak
make app           # Streamlit on :8501
```

…and search `lung failure` to get ARDS trials ranked above unrelated ones, **without
the laptop becoming unresponsive at any point.**

---

**Change log**

| Date | Version | Change |
|------|---------|--------|
| 2026-08-27 | 1.0.0 | Initial PRD derived from the legacy Part_1/Part_2 codebase |
| 2026-08-27 | 1.1.0 | Added G6, §7.1, and F-03/06/11/18/19/23/27/29/34/41 after profiling the target laptop (4 threads, 7.89 GB RAM) |
