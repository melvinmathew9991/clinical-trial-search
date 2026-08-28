# Phases — Medical Embeddings Search

> Version: `1.1.0` · Last updated: `2026-08-27`
> The delivery plan. Work sprints **in order**. One sprint per branch, one tag per sprint.
> Live status lives in [Memory.md](./Memory.md) — this file is the plan, that file is the log.

---

## How to run a sprint

```bash
git checkout main && git pull
git checkout -b feat/s<N>-<slug>

# ... build only what this sprint lists ...

make check                      # ruff + mypy + import-linter + pytest
git add -A && git commit -m "feat(<scope>): <what>"
git checkout main && git merge --no-ff feat/s<N>-<slug>
git tag -a v0.<N>.0 -m "Sprint <N>: <title>"
```

Then append a Memory.md entry. A sprint is not done until its **Definition of Done**
is fully satisfied — including the resource budget check.

**Status legend:** ⬜ not started · 🟨 in progress · ✅ done · ⛔ blocked

---

## Sprint 0 — Foundation & Governance  ✅

**Goal:** A repo an AI agent or a new engineer can pick up cold.

| # | Task |
|---|------|
| 0.1 | Write `PRD.md`, `Architecture.md`, `Rules.md`, `Phases.md`, `Memory.md` |
| 0.2 | Create the folder skeleton from Architecture §3 |
| 0.3 | `pyproject.toml` — PEP 621 metadata, deps, ruff/mypy/pytest/import-linter config |
| 0.4 | `.gitignore` (data, models, reports, .env, artefacts), `.dockerignore` |
| 0.5 | `.env.example`, `.pre-commit-config.yaml`, `.streamlit/config.toml` |
| 0.6 | `Makefile`: `setup doctor data train index app test check clean` |
| 0.7 | `README.md` quickstart |
| 0.8 | `git init`, initial commit, tag `v0.0.0` |

**DoD:** `git log` shows the initial commit · every doc renders · `make help` lists targets.
**Risk:** none. **Est:** 0.5 day.

---

## Sprint 1 — Foundation layer (L0)  ✅

**Goal:** Config, logging, runtime guards, and the error hierarchy — everything else depends on this.

| # | Task | Requirements |
|---|------|--------------|
| 1.1 | `config.py` — `Settings` via pydantic-settings, `Paths` helper | F-36 |
| 1.2 | `exceptions.py` — the full `MedSearchError` tree | Rules §4 |
| 1.3 | `logging_conf.py` — structured logs, stage-boundary helper with elapsed + RSS | F-37 |
| 1.4 | `runtime.py` — `configure_threads()`, `available_memory_gb()`, `peak_rss_mb()`, `ensure_nltk_data()` | F-41 |
| 1.5 | `cli.py` skeleton + `medsearch doctor` | F-30, F-34 |
| 1.6 | Unit tests for config precedence and doctor thresholds | |

**DoD:** `medsearch doctor` prints cores, free RAM, disk, NLTK status, and exits non-zero
below the 2.0 GB floor · `OMP_NUM_THREADS=1` verified set before numpy import.
**Risk:** low. **Est:** 1 day.

---

## Sprint 2 — Data layer (L1)  ✅

**Goal:** Load the corpus correctly, cheaply, and with loud failure on bad input.

| # | Task | Requirements |
|---|------|--------------|
| 2.1 | `data/schema.py` — `CorpusSchema`: canonical names, required set, dtypes | F-02 |
| 2.2 | `data/loader.py` — `load_corpus()` with `usecols`, `limit=None` default | F-01, F-03, F-04 |
| 2.3 | `corpus_fingerprint()` — sha256 of source bytes | F-05 |
| 2.4 | Column normalisation: `Trial ID` → `trial_id`, `Publication date` → `publication_date` | |
| 2.5 | `scripts/migrate_legacy.py` — move the CSV in, dry-run by default | Arch §12 |
| 2.6 | Tests incl. the `--limit` regression test | Rules §5 |

**DoD:** `load_corpus()` on the real 29 MB CSV returns **10,666 rows** at **≤ 120 MB RSS** ·
a missing column raises `SchemaValidationError` naming it · `limit` logs a warning.
**Risk:** the legacy CSV has embedded newlines in abstracts — verify the parser handles them.
**Est:** 1 day.

---

## Sprint 3 — Preprocessing layer (L2)  ✅

**Goal:** One text transform, used identically for documents and queries.

| # | Task | Requirements |
|---|------|--------------|
| 3.1 | `preprocessing/normalizer.py` — module-level compiled regexes, `clean_text()` | F-07, F-11 |
| 3.2 | `preprocessing/pipeline.py` — `TextPreprocessor.transform()` / `.transform_many()` | F-08, F-09 |
| 3.3 | Streaming `transform_many` as a generator | F-06 |
| 3.4 | Token cache to `data/interim/<fingerprint>.tokens.jsonl` | F-05 |
| 3.5 | `preprocess` CLI command | |
| 3.6 | Tests: purity, URL/digit/punct stripping, query≡document equivalence | |

**DoD:** query and document paths provably share one code path (asserted in a test) ·
cold pass over 10,666 abstracts ≤ 250 MB RSS · warm cache load ≤ 3 s.
**Risk:** NLTK `wordnet`/`punkt` download on a cold machine — handled by `ensure_nltk_data()`.
**Est:** 1 day.

---

## Sprint 4 — Embedding layer (L3)  ✅

**Goal:** Train both models within budget and persist them with provenance.

| # | Task | Requirements |
|---|------|--------------|
| 4.1 | `embeddings/base.py` — `ModelKind`, `TrainingParams`, `ModelMetadata` | |
| 4.2 | `embeddings/trainer.py` — `train_model()` for skipgram + fasttext | F-13, F-14 |
| 4.3 | **Bound `bucket=50_000`** (ADR-001) | F-14 |
| 4.4 | `workers = cores - 1` (ADR-008) | F-41 |
| 4.5 | `embeddings/registry.py` — save `KeyedVectors` + `metadata.json` sidecar | F-15, F-19 |
| 4.6 | `embeddings/document.py` — `DocumentEmbedder`, vocab `frozenset` in `__init__` | F-16, F-17, F-18 |
| 4.7 | Chunked `embed_corpus()`, `float32` | |
| 4.8 | `medsearch train` CLI | F-30 |
| 4.9 | Tests: bucket bound, metadata completeness, OOV → zero vector + count |

**DoD:** FastText artefact **≤ 150 MB** (vs. the legacy 800 MB) · both models train in
≤ 12 min combined at ≤ 1.2 GB peak · `metadata.json` records hparams, corpus fingerprint,
gensim version, artefact bytes, duration.
**Risk:** ⚠️ **highest-memory sprint.** Develop with `--limit 2000`; do the full run once, alone.
**Est:** 2 days.

---

## Sprint 5 — Search layer (L4)  ✅

**Goal:** Fast, correct ranking.

| # | Task | Requirements |
|---|------|--------------|
| 5.1 | `search/index.py` — `DocumentIndex`, L2-normalise at build (ADR-003) | F-23, F-27 |
| 5.2 | `.npy` save/load with `mmap_mode="r"` | F-27 |
| 5.3 | `manifest.json` with model fingerprint; reject on mismatch | F-28 |
| 5.4 | `search/engine.py` — `SearchEngine.search()`, single matmul | F-22, F-23 |
| 5.5 | `np.argpartition` top-n | F-29 |
| 5.6 | Zero-norm / all-OOV query → typed empty result with reason | F-26 |
| 5.7 | `medsearch index build` + `medsearch search` CLI | F-24, F-25, F-30 |
| 5.8 | Tests incl. the fingerprint-mismatch and OOV regression tests | |

**DoD:** p95 query latency **< 300 ms** over 10,666 docs, measured and recorded ·
index file ≤ 5 MB · a fingerprint mismatch raises `ArtefactMismatchError`.
**Risk:** low. **Est:** 1.5 days.

---

## Sprint 6 — Application layer (L5)  ✅

**Goal:** The UI a researcher actually uses.

| # | Task | Requirements |
|---|------|--------------|
| 6.1 | `app/streamlit_app.py` — model + field selectors, search box | F-31 |
| 6.2 | `@st.cache_resource` for models and indexes | F-32 |
| 6.3 | `st.dataframe` results table (ADR-009) — replaces Plotly | F-24 |
| 6.4 | `app/components.py` — results table, sidebar, empty/error banners | F-33 |
| 6.5 | `.streamlit/config.toml` — watcher off, headless | Rules §2 |
| 6.6 | Abstract truncation with expand-on-click | |
| 6.7 | `make app` |

**DoD:** first load ≤ 8 s, subsequent queries ≤ 0.5 s · serving RSS ≤ 1.2 GB with both
models loaded · no traceback ever reaches the browser · rerun does **not** reload models.
**Risk:** Streamlit reruns the whole script per interaction — cache discipline is the sprint.
**Est:** 1.5 days.

---

## Sprint 7 — Quality gates & CI  ✅

**Goal:** Make it impossible to merge a regression.

| # | Task |
|---|------|
| 7.1 | `.pre-commit-config.yaml`: ruff, ruff-format, mypy, nbstripout, detect-secrets, large-file guard |
| 7.2 | `.github/workflows/ci.yml`: lint → type → import-linter → pytest+coverage → secret scan |
| 7.3 | `import-linter` layer contract wired to Architecture §2 |
| 7.4 | Coverage gate at 80 % |
| 7.5 | Backfill tests to clear the gate |
| 7.6 | `make check` runs the identical set locally |

**DoD:** CI green on `main` · a deliberate layering violation fails the build · a planted
fake secret is caught · fast suite < 30 s locally.
**Met 2026-08-27:** ruff, ruff-format, mypy --strict, import-linter and 14 pre-commit
hooks all pass; coverage 92.21% against the 80% gate; fast suite 326 tests in 11.6 s.
The layer contract and the pre-commit mypy hook both had to be corrected first — see
Memory.md, Track 1.
**Risk:** low. **Est:** 1 day.

---

## Sprint 8 — Evaluation & tuning  ✅

**Goal:** Numbers, not vibes. Answers "is this actually better than keyword search?"

| # | Task | Requirements |
|---|------|--------------|
| 8.1 | `tests/fixtures/eval_queries.json` — ~30 labelled query → relevant-id pairs | §8 |
| 8.2 | `pipelines/evaluate.py` — Recall@k, MRR@k, latency p50/p95, OOV rate | F-30 |
| 8.3 | TF-IDF baseline for comparison | PRD §8 |
| 8.4 | Hyperparameter sweep — `window`, `vector_size`, `min_count`, `epochs` — `scripts/sweep.py`, **full corpus, not `--limit`** (see below) |
| 8.5 | `reports/evaluation.json` + a results table in the README |
| 8.6 | Record peak RSS and wall time per stage; reconcile against Architecture §9 |
| 8.7 | Decide: does Skip-gram or FastText ship as the UI default? |

**DoD:** Recall@10 ≥ 0.70 and MRR@10 ≥ 0.45, or a written explanation plus a remediation
plan (SIF weighting) · Architecture §9 numbers replaced with measured values.
**Risk:** ⚠️ targets may not be met with mean pooling — that outcome is a finding, not a failure.
**Est:** 2 days.

**DoD met — both targets cleared, by the third remediation.** Recall@10 **0.955**
(target 0.70), MRR@10 **0.852** (target 0.45), p95 128 ms (target 300 ms), via
union retrieval with FastText. Architecture §9 carries measured values including
the union's latency cost. Full detail in [PRD §8](./PRD.md); the short version:

- The risk landed. Mean pooling loses to TF-IDF (0.485 vs 0.648, p = 0.0003).
- SIF weighting (8.4's fallback plan) was built and **made it worse** — killed.
- Rank fusion was measured **before** being built (+0.0005, p = 0.98) — killed.
- The union of both top-10 lists was the survivor: 0.955, +0.240 over
  depth-matched TF-IDF, p < 0.0001. It ships on by default and is switchable.
- 8.7 decided by measurement: **FastText**, because under the union it beats
  Skip-gram (+0.028, p = 0.019) though as a standalone ranker it does not.

**Two corrections to this plan, both from measurement:**

1. **8.4's `--limit` is wrong and the sweep ignores it.** The cap was written
   before Sprint 8 measured training at 30 s / 319 MB. On a 2,000-row sample
   most of the eval set's relevant documents are absent, so recall would score
   the sample rather than the model. The sweep runs the full corpus in ~6 min.
2. **The 2 GB training floor, not RAM, is what blocks a run on a busy machine.**
   Training peaks at 346 MB. With under 2 GB free, `scripts/sweep.py` needs
   `MEDSEARCH_MIN_FREE_MEMORY_GB=1.0`. Third instance of over-conservative
   gating found in this project; the floor itself is left alone for `train`.

---

## Sprint 9 — Containerisation  ✅ *(written, never built)*

**Goal:** One reproducible image.

| # | Task | Requirements |
|---|------|--------------|
| 9.1 | Multi-stage `Dockerfile` on `python:3.11-slim`, non-root user | F-38 |
| 9.2 | Bake NLTK data at build; artefacts fetched at runtime, not baked | |
| 9.3 | `compose.yaml` for local runs | |
| 9.4 | Healthcheck on `/_stcore/health` | |
| 9.5 | Memory limit `--memory=2g` verified | Rules §2 |
| 9.6 | `release.yml` — build and push on tag | |

**DoD:** image < 800 MB · `docker run --memory=2g` serves search successfully · image
contains no secret and no `data/`.
**Risk:** ⚠️ build on the dev laptop is disk-heavy — 34.7 GB free on `C:`; prune between builds.
**Est:** 1 day.

---

## Sprint 10 — Azure cloud pipeline  🟨 *(config written, never deployed)*

**Goal:** Retraining that runs without a laptop.

| # | Task | Requirements |
|---|------|--------------|
| 10.1 | Blob layout: `raw/`, `models/`, `index/`; lifecycle policy | |
| 10.2 | ADF linked services — **managed identity**, zero `sasUri` literals | F-40 |
| 10.3 | ADF pipeline `train-embeddings` → Databricks wheel task | F-39 |
| 10.4 | `deploy/azure/databricks/job-train.json` — replaces the missing legacy `training_model` | |
| 10.5 | `BlobEventsTrigger` on new `raw/*.csv` | F-39 |
| 10.6 | Key Vault + App Service managed identity | F-40 |
| 10.7 | App Service container deployment | |
| 10.8 | **Revoke the legacy SAS tokens** (`sp=racwdymeop`, write+delete) | PRD §10 |
| 10.9 | `deploy/README.md` runbook |

**DoD:** dropping a CSV in Blob produces new artefacts without manual action · no secret in
any tracked file · legacy tokens confirmed revoked · full-corpus training runs on Databricks,
freeing the laptop entirely.
**Risk:** ⚠️ needs a live Azure subscription and permissions. Blocked without them — build
the JSON and runbook regardless.
**Est:** 2–3 days.

---

## Sprint 11 — Hardening & v1.0  ⬜

| # | Task |
|---|------|
| 11.1 | Structured logging review — every stage boundary emits duration + RSS |
| 11.2 | `medsearch doctor --full` including artefact integrity checks |
| 11.3 | README: architecture diagram, benchmarks, troubleshooting |
| 11.4 | ADR review — remove anything superseded |
| 11.5 | Reconcile all five docs against the shipped code |
| 11.6 | Tag `v1.0.0` |

**DoD:** clean-clone → `make setup && make doctor && make train && make app` succeeds on the
dev laptop with no manual steps and no unresponsiveness.
**Est:** 1 day.

---

## Backlog (post-v1, unscheduled)

| Item | Trigger to promote |
|------|--------------------|
| ~~SIF / TF-IDF-weighted document vectors~~ | **Done and rejected** — built in Sprint 8, measurably worse (PRD §8.2) |
| FastAPI `/search` endpoint (F-35) | A second consumer beyond the UI appears |
| BioBERT comparison | Budget for GPU or a cloud-only path exists |
| Approximate NN (hnswlib) | Corpus exceeds ~1 M documents |
| Domain stopword allowlist (F-12) | Negation errors show up in evaluation |
| Incremental training on new trials | CSV drops become frequent |

---

## Critical path

```
S0 ─▶ S1 ─▶ S2 ─▶ S3 ─▶ S4 ─▶ S5 ─▶ S6 ─▶ S8 ─▶ S11
                              └────▶ S7 (parallel from S5)
                                     S9 ─▶ S10 (parallel from S6)
```

S7 can run alongside S5+. S9/S10 can run alongside S6+. Everything else is strictly serial.

**Total estimate:** ~15 working days. **Highest-risk sprints:** 4 (memory), 8 (quality
targets), 10 (external Azure dependency).

---

**Change log**

| Date | Version | Change |
|------|---------|--------|
| 2026-08-27 | 1.0.0 | Initial 12-sprint plan |
| 2026-08-27 | 1.1.0 | Added per-sprint resource DoD criteria and memory-risk flags |
