# Memory — Medical Embeddings Search

> **Running progress log. Read this first in any new session.**
> Append-only. Never rewrite or delete a past entry — correct it with a new one.
> Purpose: let a fresh chat or a different AI tool resume work without re-reading the codebase.

---

## Current state — read this block first

| | |
|---|---|
| **Version** | `v0.5.0` |
| **Active sprint** | Sprint 6 — Application layer (Streamlit UI) |
| **Last completed** | Sprint 5 — Search layer |
| **Branch** | `main` |
| **Next action** | Build `app/streamlit_app.py` with `@st.cache_resource`; replace Plotly with `st.dataframe` (ADR-009) |
| **Blocked on** | Nothing |
| **Not yet run** | Full 10,666-doc training. Only the 20-row fixture path is exercised so far. |

### Environment facts (verified 2026-08-27)
- **Dev machine:** Intel i5-7300HQ · 4 physical / **4 logical cores** · **7.89 GB RAM** (~1.07 GB free at profiling time) · `D:` has 164 GB free, `C:` has 34.7 GB free
- **Python:** 3.10.11 at the system level
- **Repo root:** `D:\Word2Vec and FastText Word Embedding with Gensim in Python\medical-embeddings-search`
- **Legacy reference (frozen, do not edit):** `..\Part_1\`, `..\Part_2\`
- Because there are only 4 threads, `workers` is **3** everywhere. Because RAM is 8 GB, artefacts are bounded and the pipeline streams.

### Key decisions already made — do not relitigate
1. `bucket=50_000` on FastText. gensim's default `2_000_000` × 100 dims × 4 B = **exactly the 800 MB** legacy `.npy`. (ADR-001)
2. Index is `.npy` `float32`, L2-normalised at build; ranking is one matmul. (ADR-002, ADR-003)
3. Vocab membership via a `frozenset` built once in `DocumentEmbedder.__init__`. (ADR-004)
4. Serve `KeyedVectors`, not full models. (ADR-006)
5. `--limit` defaults to `None` — the legacy hidden `.iloc[:100]` is gone. (ADR-007)
6. `data/` and `models/` are gitignored; artefacts are regenerated, not committed.
7. Legacy `Part_1`/`Part_2` stay untouched as reference until the user verifies and deletes them.

---

## Session log

### 2026-08-27 — Session 1 · Sprints 0–5

**Sprint 0 — Foundation & Governance ✅ → `v0.0.0`**

Analysed the legacy project, then set up the new repo.

Created: `PRD.md`, `Architecture.md`, `Rules.md`, `Phases.md`, `Memory.md`,
`README.md`, `pyproject.toml`, `Makefile`, `.gitignore`, `.dockerignore`,
`.env.example`, `.pre-commit-config.yaml`, `.streamlit/config.toml`,
`.github/workflows/ci.yml`, plus the full folder skeleton.

Findings from the legacy audit that shaped every doc:
- `engine.py:36,41` wrote **skipgram** vectors into the FastText output files, transposed. Verified: `FastText-vec-abstract.csv` is byte-for-byte `transpose(skipgram-vec-abstract.csv)`.
- `top_n.py:24` read `Fasttext-vec-abstract.csv`; the file on disk is `FastText-...`. Windows-only luck; a `FileNotFoundError` on Linux.
- `utils.py:8` silently truncated every load to 100 rows, so all shipped models were trained on 100 of 10,666 abstracts.
- Part 2's `main.py` `%run ./training_model` — **that file does not exist**. The Databricks notebook could never have run.
- Part 2 committed live Azure SAS tokens with `sp=racwdymeop` (read/write/delete), plus the subscription id and storage account name.

**Sprint 1 — Foundation layer ✅ → `v0.1.0`**
`config.py` (pydantic-settings, `MEDSEARCH_` prefix), `exceptions.py` (full error tree),
`logging_conf.py` (stage-boundary timing + RSS), `runtime.py` (thread pinning, memory
probes, NLTK bootstrap), `cli.py` + `medsearch doctor`.
Note: `configure_threads()` must run **before** numpy is imported — it is called at the
top of `cli.py` and `streamlit_app.py`, not inside a function that runs later.

**Sprint 2 — Data layer ✅ → `v0.2.0`**
`data/schema.py`, `data/loader.py`, `scripts/migrate_legacy.py`.
Corpus columns are renamed to snake_case on load. `usecols` reads 4 of 21 columns.
`limit=None` by default and logs a warning when set.

**Sprint 3 — Preprocessing layer ✅ → `v0.3.0`**
`preprocessing/normalizer.py` (regexes compiled at module level),
`preprocessing/pipeline.py` (`TextPreprocessor`, generator-based `transform_many`,
JSONL token cache keyed by corpus fingerprint).
Documents and queries provably share one code path — asserted in a test.

**Sprint 4 — Embedding layer ✅ → `v0.4.0`**
`embeddings/base.py`, `trainer.py`, `registry.py`, `document.py`.
Bounded bucket, `workers=3`, `float32`, chunked `embed_corpus`, metadata sidecars.

**Sprint 5 — Search layer ✅ → `v0.5.0`**
`search/index.py` (`.npy` + mmap + fingerprint manifest), `search/engine.py`
(single matmul, `argpartition`, typed empty result on OOV).

**Measurements so far:** only on the 20-row fixture. **Real numbers are still to be
taken** — Architecture §9 currently holds *estimates*, clearly labelled as such.
Sprint 8 replaces them with measured values.

**Left undone deliberately:** Sprints 6–11. Full-corpus training has not been run.

---

## How to update this file

Append a new `### YYYY-MM-DD — Session N · Sprint X` block containing:

1. **Sprint + status** — done / partial / blocked
2. **Files created or changed** — paths, not prose
3. **Decisions made** — and the reasoning, so it is not relitigated
4. **Measurements** — peak RSS, wall time, artefact sizes. Write "not measured" if not measured; never guess.
5. **Bugs hit and how they were fixed**
6. **Next action** — the single concrete thing to do next

Then update the **Current state** block at the top. That block is the only part that is
overwritten; everything below it is history.
