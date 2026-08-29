# Memory — Medical Embeddings Search

> **Running progress log. Read this first in any new session.**
> Append-only. Never rewrite or delete a past entry — correct it with a new one.
> Purpose: let a fresh chat or a different AI tool resume work without re-reading the codebase.

---

## Current state — read this block first

| | |
|---|---|
| **Version** | `v0.10.0` + Tracks 0, 1, 2 + the domain-audit remediation (unreleased) |
| **Done** | Sprints 0–7, 9, 10, 11 (bar the tag) · Tracks 0, 1, 2 · Sprint 8 **including three rounds of relevance judging** · the union/lexical product decision |
| **Not done** | `v1.0.0` tag (**recommend `v0.11.0` instead** — see Phases §11) · an independent clinician review of the judgements · **Sprint 9's image-size DoD (941 MB against < 800 MB)** · an actual Azure deploy · a search driven through the UI in a browser · free-standing negation (needs a query parser, not a token list) · indexing the `Trial ID` column |
| **Branch** | `feat/pre-deployment-closeout`. **Everything through PR #26 is merged to `main`** — the earlier note here that `main` sat at the Sprint 0 scaffolding was stale and is corrected. Tags run to `v0.9.0`. |
| **Next action** | **Deployment.** The pre-deployment close-out is complete: union decision closed, memory-guard flake root-caused and fixed, dependencies pinned and locked (lock now verified on linux/py3.11), on-ramp defects fixed, both capability gaps closed (trial-id 0/60 → 60/60; negation pair overlap 0.55 → 0.33), and **the image-size DoD is met — 721 MB against < 800 MB, from a 954 MB baseline.** Remaining before a `v1.0.0` tag: a browser-driven search (no browser driver installed) and the Azure deploy itself. |
| **Note** | Full-corpus training takes 2 min 22 s at ~350 MB peak. Databricks is no longer *required* for it, though still the right home for scheduled retraining. |
| **Docker** | Installed on the dev machine 2026-08-28. Both image targets build and serve; `deploy/docker/compose.yaml` is the local run. Building it found a defect no test could reach — see the session entry at the end of this file. |
| **Track 0** | ✅ **COMPLETE** — full-corpus run done 2026-08-27 23:15. Architecture §9 now holds measurements, not estimates. |

### Static gates — re-run 2026-08-28, all green

```
ruff check src tests      All checks passed
ruff format --check       55 files already formatted
mypy --strict             no issues in 32 source files
import-linter             2 contracts kept, 0 broken
check_function_length     9 over the soft cap, none over the hard cap
pytest                    531 passed, 36 deselected            (fast loop, unit only)
pytest -m ""              567 passed in 72.2s                  (full, incl. integration)
coverage                  88%     (gate: 80%)
pre-commit                not re-run — last verified 14/14 on 2026-08-27
```

**Two things the re-run surfaced, neither of them cosmetic.**

1. **The fast loop now breaches Rules.md §5.** The rule is 30 s; the default
   `pytest` (coverage on, as `addopts` sets it) takes **40.5 s** for 525 tests,
   up from 421. Either the budget moves with a stated reason or the suite gets
   trimmed — it should not simply be left over the line.
2. **One flaky integration test.**
   `TestCrossModelGuard::test_index_from_one_model_is_rejected_by_another`
   failed once in a full run, then passed in isolation, in its module alone,
   and in two further full runs. Order- or state-dependent, cause not found.
   Recorded rather than dismissed: an intermittent failure in a *guard* test is
   the kind that gets waved through until it matters.

**Two test invocations, deliberately.** `pytest` runs unit tests only and stays
under the 30 s budget in Rules.md §5. The integration module is marked `slow`
because it trains real models (~40 s). `make test-all` and CI run everything,
and that is where the **coverage gate lives** — `pipelines/train.py` is only
meaningfully exercised by the integration tests.

**Run gates through the project venv.** `make check` and CI both do.
Bare `mypy` outside `.venv` will not resolve gensim/typer/pandas.

### ✅ The pipeline runs. Verified end-to-end on 2026-08-27.

Environment is now real: `.venv` with gensim 4.4.0, numpy 1.26.4, pandas 2.3.3,
streamlit 1.62.0, typer 0.27.1, NLTK corpora present. Corpus copied to
`data/raw/dimension-covid.csv` (`reference/legacy/modular-code` left intact, `--copy` not `--move`).

**FULL CORPUS, measured 2026-08-27** (10,666 documents). Architecture §9 holds
the complete table; headline numbers:

| Stage | Wall time | Peak RSS | Artefact |
|-------|-----------|----------|----------|
| `preprocess` (cold) | 46.3 s | 313 MB | 22 MB token cache |
| `train_skipgram` | 30.0 s | 319 MB | **10.2 MB** |
| `train_fasttext` | 56.0 s | 346 MB | **29.3 MB** |
| `build_index` (each) | ~7 s | 152 MB | **4.1 MB** |
| **Full run** | **2 min 22 s** | **~350 MB** | 59 MB total |

Vocabulary **24,897** words. OOV documents: **2 of 10,666 (0.02%)**.

**Every budget beaten by a wide margin:** peak RSS 350 MB vs 2.5 GB target,
training 2m22s vs 15 min, **query p95 3.3 ms vs the 300 ms PRD G2 target**,
serving RSS 342 MB vs 1.2 GB. The estimates were far too pessimistic — written
before the bounded bucket and streaming cache existed.

**Artefacts vs the legacy project: 804.3 MB → 37.6 MB, 21x smaller.**
The FastText model alone went 762.9 MB → 29.3 MB (26x, ADR-001).

**Search quality improved sharply with the full vocabulary.** On the 2,000-doc
sample, `lung failure` returned a *renal failure* trial at rank 1 — the common
token `failure` dominating the mean. On the full corpus all four top hits are
respiratory-failure trials. The 2.1x larger vocabulary (24,897 vs 11,905) is
doing the work. Mean pooling still shows through on `kidney injury`, where an
*acute lung injury* trial places second on the shared token `injury`. Sprint 8
must still measure this properly.

OOV and stopword-only queries degrade gracefully with distinct reasons. No NaN.

### Verified vs. unverified — be precise about this

| Claim | Status |
|-------|--------|
| All modules compile (`compileall`) | ✅ verified |
| Fast suite stays under the 30 s budget | ✅ verified — **421 tests in 13 s** |
| `Settings.effective_workers` → 3 on this 4-core box | ✅ verified |
| Bucket arithmetic: default = 800,000,000 bytes | ✅ verified in test + at runtime |
| `l2_normalize` produces no NaN on zero rows | ✅ verified |
| Index save/load + fingerprint rejection | ✅ verified |
| mmap handle released on Windows (`WinError 32` fix) | ✅ verified |
| Pipeline runs end-to-end (2,000-doc sample) | ✅ verified — train, index, search all pass |
| FastText artefact ≤ 150 MB budget | ✅ verified — 24 MB at 2,000 docs |
| `doctor` refuses below the RAM floor, exit 1 | ✅ verified |
| Graceful OOV / stopword-only degradation | ✅ verified |
| Streamlit app serves (HTTP 200, page renders) | ✅ verified — health endpoint + page body |
| ruff / ruff-format / mypy --strict / import-linter | ✅ verified — all pass |
| pre-commit, all 14 hooks | ✅ verified — blocked three bad commits so far |
| CLI rejects an invalid `--model` / `--field` | ✅ verified |
| Architecture.md §9 full-corpus figures | ✅ **MEASURED** — estimates replaced 2026-08-27 |
| Full 10,666-document pipeline runs on the dev laptop | ✅ verified — **2 min 2 s, 353 MB peak, re-measured 2026-08-29 from a clean clone** |
| A clean clone reaches a working app | ✅ **verified 2026-08-29** — install 3m44s, train, index, CLI search, Streamlit HTTP 200. Four defects, all in the clone-to-corpus path, none in the code |
| Serving RSS ≤ 1.2 GB | ✅ verified — **438 MB** for fasttext + union; 342 MB was skipgram alone |
| PRD G2 query latency p95 < 300 ms | ✅ verified — 3.3 ms embedding-only, **128 ms for the shipped union** |
| PRD G4 full corpus indexed (10,666/10,666) | ✅ verified, `sampled: false` in metadata |
| Recall@10 ≥ 0.70 | ⚠️ **met at 0.702** on the re-judged set (was 0.955 on the biased pool), and only from a 17.8-document result set. The depth-10 ceiling is 0.626, so no 10-document system can reach the target on this set at all |
| Embeddings beat TF-IDF | ❌ **REFUTED** — round 1: TF-IDF 0.648 vs 0.485, p=0.0003, survives Bonferroni. Re-judged: 0.459 vs 0.353, same direction and a wider gap; intervals not recomputed |
| FastText beats Skip-gram | ⚠️ **Depends on the budget.** ❌ standalone (p=0.28/0.86/1.00); ✅ under the union (+0.028, p=0.019, round-1 intervals). Re-judged the union gap is +0.015, same direction, not re-tested — the default stands |
| SIF weighting closes the gap | ❌ **REFUTED** — no gain for skipgram, significantly worse for fasttext |
| Rank fusion closes the gap | ❌ **REFUTED** — RRF = +0.0005, p=0.98 |
| Embeddings find docs TF-IDF misses | ✅ **CONFIRMED** — 66% of their hits are unique (an overlap measure, unaffected by pooling). The union@20 figure beside it is now 0.702, not 0.955 |
| MRR@10 ≥ 0.45 | ✅ measured — re-judged **0.890** union / 0.818 skipgram / 0.952 TF-IDF / 0.909 BM25 |
| Coverage gate (80%) on the full suite | ✅ verified 2026-08-28 — **87.62%** |
| 561 tests pass (525 fast + 36 integration) | ✅ verified 2026-08-28 — with the one flake noted above |
| BM25 beats / loses to TF-IDF | ❌ **neither** — Δ +0.0116, 95% CI [−0.0195, +0.0435], p = 0.47. Equivalent on this corpus. The round-1 21-point gap was pool bias |
| The union is a better ranker | ❌ **REFUTED by nDCG@10** — BM25 0.799 / TF-IDF 0.797 against union-fasttext 0.746. It is a wider net, not a better ranker |
| The eval judgements are human ground truth | ⚠️ **no** — 986 human, **705 model-generated** at Cohen's κ = 0.800 against the human labels. No clinician has reviewed any of them |
| The tokeniser fix improves retrieval | ✅ **CONFIRMED (round 3)** — registry-code Recall@10 0.44 → 1.00 TF-IDF, 0.33 → 1.00 BM25; `CD4` and `CD8` returned an *identical* top-10 under the old chain |
| The negation fix improves retrieval | ⚠️ **half-refuted (round 3)** — the gain is hyphen-joining (`nonhospitalized`, idf 6.45), not `CLINICAL_KEEP_WORDS`. Free-standing `not` (idf 2.18) is retained but weighted below the words it inverts |
| The union is safe as the default | ⚠️ **contested (round 3)** — on known-item code queries it halves P@1 against the lexical baselines, 0.500 vs 1.000 |
| A search performed *through the UI* | ⚠️ app serves (HTTP 200 on `/_stcore/health`, page renders, 11 KB body) and the engine path is tested; **no browser interaction has ever been driven** |
| Docker image builds | ✅ **built and served 2026-08-28** — both targets. Cold build 8 m 49 s |
| Container serves search under `--memory=2g` | ✅ verified — union query returns 16 docs, peak RSS **662 MB**, warm p95 **103.5 ms** |
| Image < 800 MB (Sprint 9 DoD) | ❌ **941 MB** — 725 MB of it is the venv (pyarrow, scipy, pandas, gensim). Not reachable by packaging; it needs a re-set target or a lighter UI stack |
| Image carries no secret and no `data/` | ✅ verified in the built layers |
| `standalone` target serves with no mounts | ✅ verified — the free-tier path works; baked artefacts measure 108.5 MB, confirming the documented 110 MB |
| Azure pipeline deploys | ❌ **never deployed** — and pre-deployment review found the App Service config mounted no artefacts, so the first deploy would have served zero results |

### Track 0 — three defects found by actually running it

None were reachable by inspection, tests, or type checking:

1. **`train_model` passed `corpus_iterable=` to the gensim constructor.**
   That keyword belongs to `.train()`/`.build_vocab()`; `__init__` takes
   `sentences=`. `TypeError` on every run — training could never have started.
2. **The memory floor was flat at 2.0 GB regardless of `--limit`,** so the
   2,000-row fallback the error message *recommended* was refused by the same
   check. Now scales against the 10,666-doc reference with a 0.5 GB minimum.
3. **Remediation text suggested flags the caller had already passed.**
   `require_memory` and `warn_if_memory_tight` now take `limit`.

Six regression tests added. **43 pass.**

### Environment facts (verified 2026-08-27)
- **Dev machine:** Intel i5-7300HQ · 4 physical / **4 logical cores** · **7.89 GB RAM** (~1.07 GB free at profiling time) · `D:` has 164 GB free, `C:` has 34.7 GB free
- **Python:** 3.10.11 at the system level
- **Repo root:** `D:\Word2Vec and FastText Word Embedding with Gensim in Python\medical-embeddings-search` (an earlier entry says `clinical-trial-search`; that name did not stick)
- **Legacy reference (frozen, do not edit):** `reference/legacy/` — inside the repo since the restructure, so the citations in the docs and `test_regressions.py` actually resolve for anyone who clones. Was `..\Part_1\` and `..\Part_2\`, outside the repo and invisible to everyone but me.
- Because there are only 4 threads, `workers` is **3** everywhere. Because RAM is 8 GB, artefacts are bounded and the pipeline streams.

### Key decisions already made — do not relitigate
1. `bucket=50_000` on FastText. gensim's default `2_000_000` × 100 dims × 4 B = **exactly the 800 MB** legacy `.npy`. (ADR-001)
2. Index is `.npy` `float32`, L2-normalised at build; ranking is one matmul. (ADR-002, ADR-003)
3. Vocab membership via a `frozenset` built once in `DocumentEmbedder.__init__`. (ADR-004)
4. Serve `KeyedVectors`, not full models. (ADR-006)
5. `--limit` defaults to `None` — the legacy hidden `.iloc[:100]` is gone. (ADR-007)
6. `data/` and `models/` are gitignored; artefacts are regenerated, not committed.
7. Legacy `reference/legacy/modular-code`/`reference/legacy/azure-pipeline` stay untouched as reference until the user verifies and deletes them.

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

**Sprint 6 — Application layer ✅ → `v0.6.0`**
`app/streamlit_app.py`, `@st.cache_resource` for models and index,
`st.dataframe` replacing Plotly `go.Table`, sidebar controls, expandable
abstracts, `MedSearchError` rendered as a message not a traceback.

**Sprint 7 — Quality gates & CI ✅ → `v0.7.0`**
`.github/workflows/ci.yml` with three jobs (quality matrix 3.10–3.12,
artefact guard, secret scan) and `.pre-commit-config.yaml` including
`check-case-conflict` — which would have caught the legacy
`FastText-vec` / `Fasttext-vec` mismatch.

**Sprint 9 — Containerisation ✅ → `v0.9.0`**
Multi-stage `Dockerfile`, non-root uid 10001, NLTK baked, artefacts not baked,
BLAS threads pinned in the image, healthcheck on `/_stcore/health`.
**Never built** — no Docker run attempted.

**Sprint 10 — Azure pipeline ✅ (config only) → `v0.10.0`**
`deploy/azure/**` rewritten on managed identity. `run_training.py` and
`run_indexing.py` replace the legacy `main.py`'s reference to the
non-existent `training_model` module. **Never deployed.**
⚠️ **Sprint 10.8 is still outstanding: the legacy SAS tokens in
`Part_2/.../read_data.py` and `top_n.py` must be revoked at the storage
account.** They expired 2021-12-31, but if the account `medicalembeddings`
still exists, revoke the SAS policy anyway — expiry is not revocation.

**Bug found and fixed during verification:** `DocumentIndex.load(mmap=True)`
held the `.npy` file handle open, so on Windows the directory could not be
removed and an index could not be rebuilt while Streamlit held it
(`WinError 32`). Added `close()` plus context-manager support; verified.

**Test-suite bug fixed:** the no-mutation regression test compared a
`StringDtype` series against a rebuilt `object` series and failed on dtype.
Rewritten to compare values. 37/37 pass.

**Left undone deliberately:**
- **Sprint 8 (evaluation)** — needs trained models. `medsearch evaluate` exists
  as a CLI surface but exits 2 with a "scheduled for Sprint 8" message rather
  than pretending to work.
- **Sprint 11 (hardening)** — final doc reconciliation and `v1.0.0`.
- `tests/fixtures/eval_queries.json` and `notebooks/01-exploration.ipynb` are
  referenced by the docs but not yet created; the migration script moves the
  notebook in.

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

### Track 1 — the gates were wrong, not just the code

Written in Sprint 7, never executed until now. Two of four gates were
themselves misconfigured:

1. **import-linter, 2 broken contracts — my contract's fault.** All four
   foundation modules were declared as one layer, but import-linter forbids
   *sibling* imports within a layer, so it rejected `runtime -> exceptions`
   and `logging_conf -> runtime`, both correct and intended. Separately,
   `app | cli | pipelines` as siblings rejected `cli -> pipelines`, which is
   the orchestration the CLI exists to do. Foundation is now four ordered
   layers; cli/app sit above pipelines.

2. **pre-commit's mypy hook ran in an isolated sandbox** without typer, so it
   reported 7 false "untyped decorator" errors the project mypy does not
   produce. Replaced with a `local` hook using the project environment — one
   source of truth, matching CI.

3. **mypy --strict: 21 → 0.** Root cause was typing gensim vectors and numpy
   arrays as bare `object`, which checks nothing. Added `medsearch/_typing.py`
   with `FloatArray`/`IntArray` aliases and a `WordVectors` Protocol.
   Structural typing keeps the test fake working and adds no hard dependency
   on gensim's concrete classes.

4. **ruff: 6 → 0.** B008, UP035, import order, 3 unused args in the
   `evaluate` stub.

**Behaviour improved, not just types:** rather than casting the CLI's
str-vs-Literal errors away, added `_as_model` / `_as_field` validators.
`--field bogus` used to travel into the pipeline and fail on a missing path;
it now fails at the boundary listing valid choices. Verified.

**Process note:** two commits were aborted by the hooks — once for a stray
blank line, once for mixed line endings. The gate is doing its job. Also:
never put backticks in a `git commit -m` string in bash; they become command
substitution. Use `git commit -F <file>`.

**Still open from Track 1:** coverage 52% vs the 80% gate. `pytest` with the
default addopts therefore still fails. That is Track 2 work — the gate is
correct, the tests are missing.

### Track 2 — coverage 52% → 92%, and the docs now match the repo

**Tests: 47 → 360.** New unit modules for `config`, `runtime`, `schema`,
`loader`, `normalizer`, `preprocessing`, `document`, `registry`, `cli`.
`test_search.py` split into `test_index.py` + `test_engine.py`, matching the
names Architecture.md declares.

**The integration module is the important addition.** It trains real gensim
models end to end over the 20-row fixture. It is the shape of test that would
have caught the `corpus_iterable=` constructor bug from Track 0 — neither unit
tests nor `mypy --strict` did. It took `pipelines/train.py` from **0% to 88%**.

**Fixture corpus is now a committed file**, `tests/fixtures/sample_corpus.csv`,
as Architecture.md always claimed. 20 rows across six topic clusters
(respiratory, vaccine, renal, coagulation, inflammation, antiviral) so
retrieval assertions are meaningful rather than arbitrary.

**Doc reconciliation.** Architecture.md named **16 files that did not exist**.
Created: `app/components.py` (rendering split out — `streamlit_app.py` is now
flow only), `deploy/docker/compose.yaml`, `deploy/README.md`,
`.github/workflows/release.yml`, `deploy/azure/app-service/site-config.json`,
plus the test modules. The folder tree, §2 layer table and §4 naming section
were rewritten to match disk, including `_typing.py`, `run_training.py`,
`run_indexing.py`. No empty directories remain.

**Two files stay deliberately absent** — `pipelines/evaluate.py` and
`tests/fixtures/eval_queries.json`, both Sprint 8. Architecture.md §3.1 now
records the reasoning: relevance judgements are what every retrieval metric is
measured against, and inventing them — or deriving them from keyword overlap —
would produce numbers that look authoritative while quietly favouring the
keyword baseline the project exists to beat. That needs a human who can read a
clinical abstract.

**Smaller fixes:** three blind `pytest.raises(Exception)` assertions replaced
with `FrozenInstanceError` after checking what the frozen dataclasses actually
raise. The `settings` fixture used `vector_size=8`, which `Settings` correctly
rejects (floor is 16) — the validator catching my own test.

**Process note:** the pre-commit hooks blocked this commit once for mixed line
endings, then passed 14/14. Third time the gate has caught something real.

### Track 0 closed — the full-corpus run, 2026-08-27 23:13–23:16

Ran once Brave was closed and 3.82 GB was free. `medsearch doctor` passed,
dev artefacts were cleared first so nothing sampled could survive.

```
preprocess    46.3 s   313 MB   22 MB token cache
skipgram      30.0 s   319 MB   10.2 MB   vocab 24,897
fasttext      56.0 s   346 MB   29.3 MB
index x2      ~7 s ea  152 MB   4.1 MB each
TOTAL         2m22s    ~350 MB  59 MB
```

**Every target beaten, most by an order of magnitude.** Peak RSS 350 MB against
a 2.5 GB budget; training 2m22s against 15 min; **p95 query latency 3.3 ms
against the 300 ms PRD G2 target** (120 queries, p50 1.4 ms, p99 10.7 ms);
serving RSS 342 MB against 1.2 GB; cold engine load 2.96 s.

**Artefacts: 804.3 MB → 37.6 MB (21x).** FastText model 762.9 → 29.3 MB (26x)
from ADR-001; document vectors 20.7 → 4.2 MB (5x) from ADR-002.

**PRD G4 satisfied:** 10,666 of 10,666 documents indexed, `sampled: false`
stamped in the metadata, corpus fingerprint carries no `-n` suffix. Only
**2 documents (0.02%)** were all-out-of-vocabulary.

**Search quality is materially better at full scale.** The 2,000-doc sample put
a *renal failure* trial first for `lung failure`; the full corpus returns four
respiratory-failure trials. Vocabulary 11,905 → 24,897 is why. Mean pooling
still leaks — `kidney injury` places an *acute lung injury* trial second on the
shared token `injury` — which is exactly what Sprint 8 exists to quantify.

**Architecture.md §9 rewritten with measurements.** It no longer contains
estimates. The comparison table against the legacy artefacts is now real data.

**Consequence worth noting:** full-corpus training on this laptop takes under
three minutes at ~350 MB. Databricks is no longer *required* to train — it
remains the right home for scheduled retraining triggered by a blob drop, but
the "the laptop cannot do this" assumption in earlier planning was wrong.

### Sprint 8 — harness built, blocked on labelling

Everything measurable is in place; only the human judgements are missing.

**`pipelines/evaluate.py`** — Recall@k, MRR@k, Precision@k, latency
percentiles, unanswered count, for every model plus the baseline. Writes
`reports/evaluation.json`. `check_targets()` compares the best embedding
method against the PRD targets; `medsearch evaluate` exits 1 on a miss, so it
can gate a release.

**`search/baseline.py`** — TF-IDF on `scipy.sparse`. scikit-learn was **not**
added: 40,012 terms x 10,666 documents dense would be ~1.7 GB. Measured on the
full corpus: 2.0 s build, 235 MB, 40,012 terms. It shares the corpus
preprocessor deliberately — comparing two retrieval methods under different
tokenisation measures the tokeniser, not the method.

**`scripts/make_eval_candidates.py`** — pools candidates from Skip-gram,
FastText **and** TF-IDF. Ran over 30 queries: 456 candidates, ~15 per query.

**The pooling result is worth recording: 161 candidates (35%) were surfaced
only by TF-IDF.** Had the pool come from the embedding models alone, those
documents would have been invisible to the labeller and the baseline would
have been scored on evidence it never got to present. Overlap breakdown:

```
161  tfidf only
120  fasttext + skipgram
 65  all three
 49  skipgram only
 47  fasttext only
 14  one embedding + tfidf
```

**Deliberately not done: the relevance judgements.** They are what every
metric is measured against, so deriving them from the retrieval system would
be circular. `load_eval_set` refuses to run without them and points at the
candidate script. It also rejects a query with an empty `relevant` list —
that scores 0 everywhere and would read as a retrieval failure rather than a
missing label.

**Left for the human:** label `reports/eval_candidates.json`, strip the
`_candidates` keys, save as `tests/fixtures/eval_queries.json`, run
`medsearch evaluate`. Then Sprint 8.4 (hyperparameter sweep), 8.6 (record
measurements) and 8.7 (choose the UI default model) become possible.

**Gates after this work:** ruff pass, mypy clean over 28 files, import-linter
2/2, 363 fast tests, coverage 88.71%. 39 new tests; the metric implementations
are hand-checked against worked examples rather than a reference library,
because a subtle error in Recall@k or MRR would invalidate every Sprint 8
conclusion while staying invisible in the output.

### Sprint 11 — hardening, and a third mismatch class found

Auditing `doctor --full` (task 11.2) turned up a real gap: **nothing compared
an index's corpus fingerprint against the live corpus file.**

Row ids are **positional**. A stale index therefore does not fail — it
resolves, to the wrong documents. A result would carry one trial's title beside
another trial's score and nothing would look broken. Worse than an error.

Not hypothetical: the Azure pipeline fires on a CSV landing in blob storage. If
training fails after the drop but the app restarts, it serves the previous
corpus's index against the new data.

**Verified by actually breaking it** — appended a newline to the corpus,
confirmed both `load_search_engine` (raises `StaleIndexError`) and
`doctor --full` (ERROR `index-corpus-stale`) catch it, then restored the corpus
and confirmed search works again.

The three mismatch classes now guarded, each found the hard way:

| Class | Found | Guard |
|-------|-------|-------|
| model ↔ index | legacy audit | `expected_fingerprint` on load |
| sampled index vs full corpus | Track 0 | `is_sampled`, UI banner |
| **stale index after corpus change** | **Sprint 11** | `StaleIndexError` |

`pipelines/integrity.py` checks all three plus sampled/oversized artefacts and
row-count disagreement, sorted most-severe-first. `doctor --full` exits
non-zero on any ERROR, so it can gate a deploy.

**Severity is split deliberately:** a stale *model* is a WARN (its vectors are
still coherent); a stale *index* is an ERROR (it mis-resolves documents).

**Doc reconciliation (11.4, 11.5):** ADR-010 added. Rules.md exception
hierarchy was three types out of date. Rules.md §5 claimed tests never train a
model — untrue since the integration suite landed. README gained measured
benchmarks, the 21× artefact comparison, and an integrity table; its status
section no longer overstates what is verified.

**Two CLI tests failed** because they asserted the old `evaluate` stub's exit
code 2. That is the suite doing its job — updated to the real behaviour.

**`v1.0.0` deliberately NOT tagged.** Retrieval quality is still unmeasured.
Tagging a release whose central claim — that in-domain embeddings beat keyword
search — has never been tested would be dishonest. The tag waits on the
labelled eval set.

**Gates:** ruff pass, mypy 29 files clean, import-linter 2/2, 381 fast tests in
10.8 s, 417 total, coverage 88.20%.

**Still unverified, honestly:** Docker image never built (no Docker installed
on this machine), Azure never deployed, no browser-driven UI interaction.

### Sprint 8 measured — the premise does not hold as built

Labelled the eval set by reading all 456 pooled candidates (319 judgements
across 30 queries), stamped `labeller: assistant` with the bias documented.
Then ran the evaluation.

```
                 Recall@10   MRR@10   P@1     p95
tfidf-baseline      0.647    0.939   0.900   3.4 ms
skipgram            0.559    0.787   0.700   3.1 ms
fasttext            0.539    0.786   0.667   3.5 ms
```

**TF-IDF beats both embedding models on every ranked metric.** PRD G1 —
"semantic retrieval beats keyword baseline" — is **not met**. Recall@10 also
misses the 0.70 target.

**Do not treat this as noise.** Per query it is 13–13 with 4 ties, but the
margins are asymmetric and the pattern is systematic:

| TF-IDF wins big | Δ | Embeddings win | Δ |
|---|---|---|---|
| steroid therapy for inflammation | −1.000 | mechanical ventilation weaning | +0.500 |
| diabetes as a risk factor | −0.667 | viral load reduction | +0.200 |
| monoclonal antibody treatment | −0.667 | breathing difficulty | +0.167 |
| elderly care home residents | −0.385 | lung failure | +0.167 |

That split is exactly the theoretical prediction. **Embeddings win on
paraphrase** — no trial contains the word "weaning", so TF-IDF cannot retrieve
ventilation-liberation studies at all. **TF-IDF wins when one rare term decides
relevance** — IDF weights `steroid` heavily, while mean pooling averages it
against every other word in the abstract and dilutes it to nothing.

**Diagnosis: the failure is mean pooling, not the embeddings.** Averaging
treats `steroid` and `patients` as equally informative. IDF is precisely the
signal ADR-005's mean-pooled document vector discards. The paraphrase win on
`lung failure` → ARDS, noted anecdotally back in Track 0, is real — it just
does not outweigh the loss of term discrimination in aggregate.

**Remediation, now evidence-backed rather than generic:** SIF / IDF-weighted
document vectors — weight each word vector by `a / (a + p(w))` before
averaging. Second candidate: a TF-IDF + embedding hybrid, supported by the
near-even per-query split. Both were P2 backlog items; the first is now the
top priority.

**On my own reasoning here:** I had earlier refused to label the eval set,
arguing a machine judgement would be "circular — the retrieval system grading
its own homework". That objection is right for labels derived from search
scores or keyword overlap, but wrong for reading abstracts and judging them,
which is the same task a human labeller performs. I over-applied the rule and
it stalled the project. The real bias — same author as the system, pooled
candidates only — is documented in the eval set's `_provenance` instead.

**Labelling notes worth keeping:** relevance was judged on subject matter, not
word overlap. Population-only matches were excluded (a vaccine trial whose
subjects are dialysis patients is not about "dialysis in critical illness").
Ambiguous truncated abstracts were excluded, which depresses recall equally for
all methods. `oxygen therapy at home` was scored strictly (home/outpatient
only), so every method scores low on it by design.

**`v1.0.0` still not tagged** — tagging now would ship a system whose central
claim is measured and refuted. The tag waits on the SIF fix and a re-run.

### Correction — the TF-IDF result was overstated

The previous entry said "TF-IDF beats both embedding models" and "the premise
does not hold as built". **That was a point estimate reported as a result.**
Paired significance testing over the same 30 queries:

| Metric | Mean diff (sg − tfidf) | 95% bootstrap CI | perm. p | Verdict |
|--------|------------------------|------------------|---------|---------|
| Recall@10 | −0.088 | [−0.199, **+0.014**] | 0.132 | not significant |
| MRR@10 | −0.151 | [−0.286, −0.022] | 0.040 | nominally significant |
| Precision@1 | −0.200 | [−0.400, **0.000**] | 0.106 | not significant |

Sign test 13–13, p = 1.000. MRR@10 **does not survive** Bonferroni correction
across the three metrics (α = 0.0167).

**Corrected conclusions.** Defensible: Recall@10 = 0.559 misses the 0.70 target
(unambiguous); TF-IDF puts a relevant document at rank 1 more often, 27/30 vs
21/30. Not defensible: that keyword search retrieves more relevant material, or
that the project's premise is refuted. **n = 30 cannot resolve a difference of
this size.**

**This changes the recommendation.** The prior entry promoted SIF weighting to
top priority. That was premature — not because SIF is wrong, but because the
current eval set cannot measure whether it helps. A +0.05 gain would be
indistinguishable from chance. **Expanding the eval set to ~100 queries is now
the highest-value action.** SIF remains the leading modelling change afterwards:
it is principled (targets the top-rank deficit that mean pooling causes) rather
than fitted, so implementing it is defensible; *claiming a win* from it on n=30
would not be.

Lesson for future sessions: this project measures things carefully and then
must report them just as carefully. Running the test before writing the
conclusion would have avoided committing an overstatement.

### Eval set expanded to 97 queries — the finding is confirmed

The n=30 conclusion was retracted as underpowered. Resized by power calculation
(sd 0.308, target effect 0.088 → n=97) and re-run. **97 queries, 986
judgements.**

```
                 Recall@10   MRR@10   P@1
tfidf-baseline      0.648    0.888   0.680
fasttext            0.485    0.761   0.515
skipgram            0.469    0.757   0.515
```

**Every embedding-vs-TF-IDF gap is now significant and survives Bonferroni**
(α=0.0167 over 3 metrics), paired, 20k bootstrap + 20k permutation:

| Comparison | Δ | 95% CI | p |
|---|---|---|---|
| skipgram − tfidf, R@10 | −0.179 | [−0.256, −0.101] | <0.0001 |
| fasttext − tfidf, R@10 | −0.163 | [−0.249, −0.078] | 0.0003 |
| skipgram − tfidf, MRR | −0.131 | [−0.208, −0.054] | 0.0014 |
| fasttext − tfidf, MRR | −0.127 | [−0.208, −0.047] | 0.0031 |
| skipgram − tfidf, P@1 | −0.165 | [−0.278, −0.062] | 0.0058 |
| fasttext − tfidf, P@1 | −0.165 | [−0.278, −0.052] | 0.0082 |

Per query: **TF-IDF 56, embeddings 28, 13 ties** — not the 13–13 the
underpowered set showed. **FastText and Skip-gram are indistinguishable**
(p=0.28/0.86/1.00): the character n-grams buy nothing measurable here.

**PRD G1 is refuted.** A 40-line TF-IDF baseline beats both in-domain embedding
models on this corpus.

**Two methodological fixes shipped with the expansion.** Pool depth is now 10
per method, matching the k=10 cut-off (the first run pooled at 8 while scoring
at 10, so ranks 9–10 could go unjudged). Carried-forward judgements are
filtered to the new pool rather than assumed valid.

**Three queries dropped** for having no relevant candidate, each documented.
One matters: *"how long patients stay infectious"* has relevant trials in the
corpus — they appeared under other queries — but **no method retrieved them
here**. Dropping is right for scoring yet conceals a failure shared by all
three methods.

**Why recall fell for embeddings (0.559 → 0.485) while TF-IDF held (0.647 →
0.648):** the 70 new queries include many specific-entity searches — drug
names, procedures, `ECMO` — which is realistic clinical search and exactly
where lexical matching dominates. The original 30 were skewed toward broad
clinical concepts, which flattered the embeddings.

**The arc worth remembering:** claimed the result at n=30 → retracted it as
statistically unsupported → sized the study properly → confirmed it. The
retraction was correct at the time; the conclusion is now earned rather than
asserted.

### SIF implemented — and measured to fail

The remediation the whole analysis pointed at **does not work**.

```
Recall@10        mean     SIF     delta
skipgram        0.469    0.450   -0.019   ns (p=0.35)
fasttext        0.485    0.432   -0.053   SIGNIFICANT, CI [-0.102,-0.007]
tfidf-baseline  0.648      --      --     still ahead of everything
```

**Ablation separating SIF's two steps:**

| Variant | skipgram | fasttext |
|---|---|---|
| mean pooling | 0.469 | 0.485 |
| SIF weights only | 0.447 | 0.432 |
| SIF weights + component removal | 0.450 | 0.432 |

Common-component removal changes nothing measurable (+0.003 / −0.001, both ns).
**The frequency weighting itself is what costs recall** — which rules out a bug
in the second step as the explanation.

**My §8.1 diagnosis was wrong.** I wrote "the failure is mean pooling, not the
embeddings," implying a fixable pooling defect. It isn't. The better
explanation:

1. **Reweighting cannot recover lexical precision from a dense average.**
   TF-IDF matches exactly in 40,012 sparse dimensions where `colchicine` is its
   own coordinate. Upweighting it inside a 100-dim average still blurs it
   against every other word. **Dimensionality is the binding constraint, not
   weighting.**
2. **Rare words have the least reliable vectors** (fewest training examples),
   so SIF puts the most weight on the least trustworthy directions. Consistent
   with FastText suffering most — its rare-word vectors are synthesised from
   character n-grams rather than observed.

**The hybrid is now the only candidate left.** Keep TF-IDF for lexical
precision, add embeddings for the 28 queries they win. It does not ask the
dense representation to do something it structurally cannot.

**SIF stays in the codebase** — `--pooling sif`, defaults off, separate index
directory, 23 tests. A negative result that cannot be reproduced becomes
folklore.

**Second bug fixed here:** the memory floor gated index builds on the *training*
budget and refused a build that fits easily. Indexing peaks at 152 MB vs
training's 346 MB, so `memory_floor_gb` now takes a `stage`. Same class of
over-conservatism as the flat floor fixed in Track 0, one level down.

**Worth remembering:** this is the second diagnosis in this project to survive
plausibility and die on measurement. The powered eval set earned its cost here —
at n=30 the SIF result would have been unreadable noise.

### The hybrid: measured before building, and not worth building

Same discipline as the power calculation — check the precondition first.

**Rank fusion at a fixed budget of 10 gives nothing:**

| Method | docs | Recall | vs TF-IDF |
|---|---|---|---|
| TF-IDF | 10 | 0.648 | — |
| FastText | 10 | 0.485 | — |
| **RRF over both top-10** | 10 | **0.648** | **+0.0005, p=0.98** |
| RRF over both top-30 | 10 | 0.560 | −0.087, **worse** |

**But the complementarity is real and is NOT a depth artefact:**

| Method | docs | Recall |
|---|---|---|
| TF-IDF depth-matched | 20 | 0.715 |
| **Union of both top-10** | 20 | **0.955** |

Union − depth-matched TF-IDF = **+0.240, p < 0.0001**. Of 496 relevant docs the
embeddings retrieve, **326 (66%) are ones TF-IDF never returns**; on **80 of 97
queries** they contribute at least one unique hit.

**The resolution:** at ten results a fusion must *drop* a TF-IDF hit to admit an
embedding hit, and TF-IDF's are more often right. The complementarity is
genuine and unusable by reranking simultaneously.

**So the lever is the result budget, not the model.** Returning the 20-document
union clears the 0.70 target at **0.955** with no new modelling. That is a
product decision — how many trials to show a researcher — not an ML one. The
alternative, a cross-encoder reranker over the 20 candidates, is a PRD §5
non-goal and would not fit the G6 memory budget.

**Pattern worth keeping:** three remediations proposed, all plausible; two
measured and killed, the third measured and killed *before* it was built. The
evaluation harness has now saved more work than it cost.

### Sprint 8 closed — the union ships, and the sweep confirms nothing else would

**DoD met, by the third remediation.** Recall@10 **0.955** (target 0.70),
MRR@10 **0.852** (target 0.45), p95 **128 ms** (target 300 ms). `medsearch
evaluate` now prints "All PRD targets met" and exits 0 for the first time.

| Method | docs | R@10 | MRR@10 | P@1 | p95 |
|---|---|---|---|---|---|
| **union-fasttext** *(ships)* | 17.7 | **0.955** | 0.852 | 0.753 | 128 ms |
| union-skipgram | 17.5 | 0.927 | 0.822 | 0.737 | 128 ms |
| tfidf-baseline | 10.0 | 0.648 | **0.888** | **0.835** | 3.6 ms |
| fasttext | 10.0 | 0.485 | 0.761 | 0.670 | 3.6 ms |
| skipgram | 10.0 | 0.469 | 0.757 | 0.670 | 3.3 ms |

**The union's cost, stated plainly:** it loses Precision@1 (0.753 vs TF-IDF's
0.835) and MRR@10 (0.852 vs 0.888). It buys recall with a longer list and a
slightly worse top. That is the right trade for someone who must not miss a
trial, which is why it ships *and* why it has a toggle.

**8.7 decided by measurement, not preference: FastText.** As standalone rankers
Skip-gram and FastText are indistinguishable (p = 0.28 / 0.86 / 1.00) — which
is exactly why this decision had been sitting open. Under the union they are
not: +0.028 recall, 95% CI [+0.005, +0.052], **p = 0.019**, MRR no worse
(p = 0.16). The character n-grams buy nothing at depth 10 and something real at
depth 20, presumably because that is where rare-word matches live.

**8.4 sweep: every knob is flat, and I nearly over-read it.** OFAT over
`vector_size` (200, 300), `window` (10), `min_count` (5), `epochs` (15) on the
full corpus. Largest effect **+0.040** against a **0.19** gap to TF-IDF.

The near-miss: `epochs=15` (+0.0399) and `window=10` (+0.0392) look like
findings. They are not. **The sweep's baseline config is identical to the
shipped Skip-gram's and scored 0.4548 against its 0.4691** — a 0.0143 spread
from retraining alone, because Gensim's multi-worker training is not
seed-deterministic. Two unreplicated runs at under 3x the noise floor is not
evidence. Defaults unchanged. *Always re-run the baseline config inside a
sweep; without that row the two "wins" would have been believed.*

**Two plan corrections, both from measurement (the recurring theme now):**

1. **Phases 8.4 said sweep with `--limit`.** Wrong: on a 2,000-row sample the
   eval set's relevant documents are mostly absent, so recall scores the sample.
   Written before Sprint 8 measured training at 30 s / 319 MB. Full corpus, 6 min.
2. **The 2 GB training floor blocked the sweep at 1.70 GB free** while training
   peaks at 346 MB. Third over-conservative gate found (after Track 0's flat
   floor and Sprint 11's index stage). Worked around per-run with
   `MEDSEARCH_MIN_FREE_MEMORY_GB=1.0`; the floor itself left alone.

**Two stale-number corrections found while closing:**

- README's Precision@1 column read 0.680 / 0.515 / 0.515 and matched no report
  on disk — leftovers from the n=30 set that survived the 97-query update.
  Measured: 0.835 / 0.670 / 0.670. The claim built on it still stands.
- PRD §8's headline table was still entirely n=30. Replaced.

**Infrastructure that should have existed three findings ago:**
`scripts/significance.py` — the paired bootstrap had been written ad hoc and
thrown away three times, which is why no p-value in PRD §8 could be
re-derived from the repo. Now it can be. `scripts/sweep.py` likewise.

**`.gitattributes` added.** Every source file was showing as fully rewritten in
`git diff` (CRLF working tree, LF index, `autocrlf=false`) — 1,966 changed lines
hiding 194 real ones. Normalised to LF.

### Post-tag fix — the FastText default was inert

Caught by running `medsearch search "lung failure"` after tagging v0.8.0: the
header said `(skipgram/abstract)`. Changing `Settings.default_model` to
`fasttext` and documenting it in three files had changed nothing a user
touches. `search` carried its own `typer.Option("skipgram", ...)` literal, and
the Streamlit sidebar's `selectbox` listed `["skipgram", "fasttext"]` with no
`index=`, so both ignored the setting entirely.

**The setting existed and was never wired.** Not a regression — it had always
been decorative, which is why nothing failed. Every gate passed: ruff, mypy
--strict, import-linter, 454 tests, 87% coverage. None of them can see that a
config value reaches no caller.

`tests/unit/test_cli.py::TestDefaultModelResolution` now pins the wiring rather
than the value: it sets `MEDSEARCH_DEFAULT_MODEL` to each model and asserts the
CLI names it, plus that an explicit `-m` still wins.

**Worth remembering:** a green suite plus updated docs is not evidence that a
default shipped. Run the user-facing command and read what it prints. Third
finding in this project of the same shape — the code was fine, the *claim about
the code* was wrong (cf. the stale README Precision@1 column, and the eval
report that had never been regenerated).

### Post-Sprint-8 audit — what a full-repo review turned up

Ran an audit across structure, standards, complexity, and error handling
rather than trusting the docs' own claims. Four things were wrong.

**1. The union loaded the corpus twice — and could load the *wrong* one.**
`load_union_retriever` called `load_search_engine` (which loads a corpus
aligned to the index) and then `load_corpus` again with the caller's raw
`limit`. Two 35 MB frames, and worse: with a **sampled** index and `limit=None`
the engine holds 2,000 rows while the second load returns all 10,666 — a
2,000-vector index paired with a 10,666-document TF-IDF matrix. Same mismatch
class the index fingerprint exists to catch, one layer up, and no test covered
it because every test uses a full index. Fixed by exposing `SearchEngine.corpus`
and `.sampled_limit` and deriving both sides from the engine.

**2. Serving RSS was documented at 342 MB; the shipped default is 438 MB.**
Measured one isolated process per configuration:

| Model | Embedding only | With union |
|---|---|---|
| skipgram | 359 MB | 402 MB |
| **fasttext** | 395 MB | **438 MB** |

The 342 MB was skipgram alone — accurate when written, stale the moment
Sprint 8.4 made fasttext + union the default. Headroom against the 1.2 GB
budget is 2.7x, not 3.5x. *My first measurement said 539 MB; that was two
engines alive in one process because `del` does not return allocator pages.
One process per configuration, or the number is fiction.*

**3. Two functions are over the 60-line hard cap in Rules §3** — `doctor` (69)
and `check_artefacts` (61), with eight more over the soft cap. **No gate
enforced the cap**, which is why it drifted for eleven sprints unnoticed.

*I first reported this as **nine** over the hard cap.* That count charged
blank and comment lines against the cap; in a codebase that comments as
densely as this one, that inflates every function by a third. Writing the
checker forced the counting rule to be stated — body lines only, docstrings
and blanks and comments excluded — and the real number fell to two. **The
lesson is the one this project keeps relearning: a number nobody can
reproduce is not a measurement.** Had I "fixed" nine functions on the first
count, seven of those refactors would have been churn against a rule that
was never broken.

**4. The notebook was never stripped.** Architecture.md §503's migration row
says "move, rename, **strip outputs**". Only move and rename happened:
`notebooks/01-exploration.ipynb` is byte-identical to the legacy file, 23 of
69 cells carry 2021 execution output, and it is committed that way —
violating Rules §6's "never commit a notebook with outputs".

**Smaller findings:** ten `src` modules have no matching
`tests/unit/test_<module>.py` despite Rules §5; `engine.py` raises a bare
`ValueError` for a dim mismatch where `ArtefactMismatchError` exists, and
`weighting.py` does the same where `EmptyCorpusError` does; `make clean`
misses `.import_linter_cache` and `src/medsearch.egg-info`.

**What the audit did *not* find, which is the more useful half:** no
`Optional[X]`/`List[X]`, no commented-out code, no TODO/FIXME, all twelve
exception classes actually raised, zero-norm guards present in all three
places that divide, the layer contract clean, and the fast suite at 13 s
against a 30 s budget. The standards hold almost everywhere; the gaps are
where **nothing automated was watching**.

### Audit remediation — all four findings closed

**Union corpus duplication fixed.** `SearchEngine` now exposes `corpus` and
`sampled_limit`; `load_union_retriever` derives both retrievers from the engine
instead of re-loading. Serving RSS 438 → 402 MB for skipgram, and the sampled-
index mismatch is structurally impossible now rather than merely untested.

**Docs reconciled with measurement.** Architecture §9 and the README carry the
per-configuration serving table; the Memory verification table's seven stale
rows are corrected and a serving-RSS row added.

**Notebook stripped.** 590 KB → 23 KB, 69 cells intact, 0 outputs. Closes
Rules §6 and the migration step Architecture §503 specified in Sprint 2.

**Function-length cap now enforced.** `scripts/check_function_length.py`, wired
into `make lengths`, `make check`, pre-commit, and CI, with 10 tests pinning
the counting rule and one that asserts the whole of `src/medsearch` stays
inside the hard cap. `doctor` split into `_check_resources` + `_report_artefacts`
(69 → 27); `check_artefacts` split into `_check_model` (61 → 30).

**Worth remembering:** three of the four findings were invisible to every
existing gate — ruff, mypy --strict, import-linter and 457 tests were all green
while the corpus loaded twice, the docs misstated RSS by 100 MB, and a
committed notebook carried 2021 output. **Gates catch what they were written to
catch.** The remaining known gaps are the two untyped `ValueError`s that should
be `ArtefactMismatchError` / `EmptyCorpusError`, ten modules without a matching
`test_<module>.py`, and `make clean` missing two cache dirs.

### Pre-deployment review — two defects only a deployment would have found

Asked whether every perspective was ready to ship. It was not, and the two
worst findings were both in Sprint 9/10 code that has never been executed.

**1. App Service mounted no artefacts.** `site-config.json` set
`AZURE_STORAGE_ACCOUNT_URL` and `AZURE_BLOB_CONTAINER` — and **nothing in
`src/medsearch` reads either one.** The only mentions of blob storage in the
package are comments. There was no `azureStorageAccounts` mount and no
`MEDSEARCH_DATA_DIR`/`MEDSEARCH_MODEL_DIR` either. Deployed as written the
container starts, **passes its health check**, and returns zero results for
every query — because `/_stcore/health` proves Streamlit is up, not that a
model is loaded. *A health check that cannot fail the way the system actually
fails is decoration.*

**2. The Streamlit app never honoured `MEDSEARCH_LOG_JSON`.** App Service sets
it true. The CLI passes `json_output=settings.log_json`; the app called
`configure_logging(settings.log_level)` at both call sites. And because
`configure_logging` latches on first call, no later caller could correct it —
production logs would have been plain text, unparseable by the pipeline, with
no error anywhere. Found by writing the *first ever* test for
`logging_conf.py`.

**3. Three descriptions of the storage layout, none agreeing.**
`run_training.py` writes `$mount/data` + `$mount/models`; `deploy/README.md`
documented a top-level `index/` prefix nothing writes or reads; App Service
described neither. Nothing forced them to agree because nothing ever ran end
to end.

**The pattern, stated plainly:** every one of these lived in the half of the
repo that has never executed. Sprint 9 is marked ✅ *(written, never built)*
and Sprint 10 🟨 *(config written, never deployed)* — honest labels, and this
is what they cost. Code that has not run is not "done"; it is a hypothesis.

**Still genuinely blocked, and not by anything I can fix:**
- **Docker is not installed on this machine.** Sprint 9's DoD ("image builds")
  is unverifiable here. The Dockerfile reads well — multi-stage, non-root uid
  10001, pinned BLAS threads, healthcheck — and `python -m medsearch.runtime
  --download-nltk` does exist, so the build step is at least real. But
  "reads well" is exactly the standard that produced the three defects above.
- **Azure has never been deployed.** Needs a subscription and the managed
  identity role assignments in `deploy/README.md` section 3.
- **No browser has ever driven the UI.** The app serves and the engine path is
  tested; the widget layer is not.

### Domain audit — the evaluation could not measure what it claimed

Audited the eval set itself rather than the systems scored against it. Two
defects in the pipeline, and one in the measurement that subsumes both.

**The measurement is pool-bound, and I proved it.** Relevance was labelled over
a pool built from the three systems being scored, so a document no system
retrieved could not be relevant *by construction*. With the original tokeniser,
**986 of 986** relevant documents sat inside that pool — 100%, zero outside —
and the shipped union's 0.955 sat right at the 94.3% its own two members
contributed.

Then the tokeniser fix changed what gets retrieved, and re-measuring gave the
proof:

| Configuration | pool membership | measured Recall@10 |
|---|---|---|
| old tokeniser | 94.3% | 0.955 |
| new tokeniser | 85.3% | 0.862 |

**Every method dropped, TF-IDF included (0.648 → 0.615)** — and the embedding
changes cannot touch TF-IDF. Recall tracks pool membership, not quality. *The
metric can only measure agreement with the pipeline that built it.* The
`Recall@10 0.955` headline is withdrawn; PRD §8's DoD was declared met on it.

**The tokeniser was destroying clinical identity.** `strip_digits` removed every
digit, on the stated rationale that "the numbers carry no distributional signal
for retrieval" — true of prose, false of biomedicine. `CD4` and `CD8` both
became `cd`. `ACE2` became `ace`. **Every `NCT…` registry ID became `nct`.**
6,077 such tokens in 4,000 abstracts. Vocabulary 24,897 → **31,189** after the
fix.

**Negation was half-handled.** `no`/`not` dropped as stopwords while
`without`/`never`/`none`/`absent` survived, so "no evidence of thrombosis"
tokenised identically to "evidence of thrombosis". PRD F-12 predicted this and
`TextPreprocessor` had carried an unwired `keep_words` hook since Sprint 3.

**Why neither was ever caught: the eval set is blind to both.** Of 97 queries,
**0 contain a digit** and 1 contains a negation — which uses "without", the
word that survived. Every query is a natural-language phrase, exactly the case
the old tokeniser handled well.

**Worth remembering:** a drop in a metric is not evidence of a regression until
you know what the metric is bound to. My first instinct on seeing 0.955 → 0.862
was that the fix had hurt retrieval. It hadn't — the labels had gone stale, and
TF-IDF dropping was the tell, because nothing I changed could affect it. *Check
whether the control moved.*

**Also added:** nDCG@10, R-precision and the achievable-recall ceiling (0.879,
because 44 of 97 queries hold more than 10 relevant documents, so Recall@10
could never reach 1.0). `scripts/make_eval_round2.py` does incremental pooling
and has emitted **1,073 unjudged candidates across 95 of 97 queries** — until
those are judged every recall figure is a lower bound.

### Reproducibility, and a hole in the provenance guard

Cold rebuild from an empty artefact tree, **default settings, no memory
override**: 121 s train + 18 s index + 28 s evaluate = **167 s** end to end.
The pipeline reproduces.

**What is deterministic and what is not, measured rather than assumed.**
TF-IDF returned Recall@10 0.615 to the digit across two cold runs. The
embeddings moved +/-0.010 at the default `workers=3`. With
`MEDSEARCH_WORKERS=1` the vector matrices are **bit-identical**
(`np.array_equal`, not a file checksum -- gensim's `.kv` serialisation is not
byte-stable, so md5 differs even when the vectors do not). Cost: 279 s against
150 s, 1.9x.

**The bigger find: the integrity guard could not see a retrain.**
`model_fingerprint` hashes (kind, corpus, hyperparameters) -- *not* the
vectors. Two runs of one config share it, and gensim above one worker gives
different vectors. So an index built from run 1 paired "validly" with run 2:
fingerprints matched, `doctor --full` printed *"every artefact is consistent
with its model and the live corpus"*, and the stored rows scored **cosine 0.96**
against the live model instead of 1.00.

That is the legacy K1/K2 failure exactly -- wrong pairing, no signal -- in the
guard built to prevent it. Fixed with `vectors_checksum`, a hash of the matrix
itself, stamped into the model metadata and the index manifest and verified at
load and in `doctor --full`. Both now fail loudly and name the two checksums.

**A second defect fell out of fixing the first.** `registry.py` rebuilt
`ModelMetadata` field by field to fill in `artefact_bytes` -- twelve fields
copied by hand. It silently dropped the thirteenth, so every model shipped with
an empty checksum and the new guard was inert until I noticed. Replaced with
`dataclasses.replace`, which cannot drift from the dataclass. *A manual copy of
every field is a bug waiting for the next field.*

**How I found it:** by accident, checking whether `workers=1` was
deterministic. The retrains left the index stale, `doctor --full` said
everything was fine, and that claim was checkable — so I checked it.

### Round 2 — the pool was re-judged, and two conclusions reversed

Judged all **1,532** outstanding candidates. Eval set 986 → **1,691**
judgements over the same 97 queries; mean relevant per query 10.2 → 17.4.

**Calibrated first, because a model judging its own evaluation is the exact
circularity this audit exists to document.** Blind, balanced 60-item sample
against the existing human labels: **90.0% agreement, Cohen's κ = 0.800**,
precision 96.2% / recall 83.3%. Human-vs-human in TREC-style judging is
typically κ 0.5–0.7, so this is usable — as a second annotator, *not* as a
clinician. The measured bias was strictness: every miss was a case where the
human read the query as a topic area and I read it as a specification. I
loosened the threshold before starting.

*Caught my own transcription error during calibration:* the first score came
out κ = 0.633 with eleven disagreements, five of which looked wrong on
inspection — #43 was a thrombosis registry for "blood clots and anticoagulation",
obviously relevant, and I had judged it so. Two blocks of answers were
mis-keyed. Corrected: κ = 0.800, six genuine disagreements. **A disagreement
that looks wrong usually is; check the harness before the judgement.**

**Reversal 1 — BM25 vs TF-IDF flips completely.**

| | biased pool | re-judged |
|---|---|---|
| TF-IDF | **0.615** | 0.459 |
| BM25 | 0.403 | **0.471** |

Nothing about either system changed. The 21-point deficit was *entirely*
TF-IDF having helped build the pool. And the corrected gap is **+0.0116,
p = 0.47 — not significant.** The truthful statement is that the two lexical
baselines are equivalent on this corpus; neither the original claim nor its
reversal survives.

**Reversal 2 — nDCG says the union is a wider net, not a better ranker.**
By nDCG@10, which is not ceiling-capped, BM25 **0.799** and TF-IDF **0.797**
beat union-fasttext's **0.746**. The union wins Recall@10 (0.702 vs 0.471) only
by returning 1.8x as many documents, and wins R-precision because that metric
is evaluated at depth |relevant| ≈ 17, which happens to match its result size.

**Worth remembering:** when three metrics disagree, that disagreement *is* the
finding. Recall@10 alone made the union look dominant; nDCG alone would make it
look worse than a 40-line baseline. Reporting one of them would have been
defensible and wrong either way.

Recall@10 ceiling is now **0.626** (was 0.879) — with 17.4 relevant documents
per query, ten slots cannot hold them.

---

### 2026-08-28 — Session 3 · Sprint 11.5, second pass: reconciling the docs with the re-judged numbers

**Status: done. No code changed; five documents did.**

Round 2 (the entry above) invalidated figures that four documents still quoted
as current. The README's own Status section still said *"retrieval quality has
not been measured"* — written before Sprint 8 and never revised — while the
README table above it quoted `Recall@10 0.955`, a number the audit had already
withdrawn. A reader would have taken either as the state of the project.

**Files changed:** `EVALUATION_AUDIT.md`, `README.md`, `PRD.md`, `Phases.md`,
`Memory.md` (this entry plus the Current state block).

**Decisions, so they are not relitigated:**

1. **Round-1 analysis is kept, not deleted.** Every superseded figure stays
   where it was written and is labelled round 1 with its pool bias stated. The
   audit's whole point is that the numbers moved and *why*; a doc that quietly
   replaced them would erase the evidence for its own finding.
2. **SIF (§8.2) and RRF (§8.3) were not re-scored on the new judgements.** Both
   branches are dead. Re-scoring a killed remediation buys nothing, and those
   comparisons are internally valid anyway — every system in them contributed
   to the pool they were scored on, which is the exact condition §6 requires.
   Marked as round-1 rather than recomputed.
3. **Recall never appears alone any more.** Every table that reports Recall@10
   now carries nDCG@10, R-precision, or the 0.626 depth-10 ceiling beside it,
   and the judgements' model-generated provenance travels with them. This is
   written into the audit as a reporting rule, not left to memory.
4. **PRD §8.4's "the union ships by default" is explicitly reopened.** It was
   decided on a 0.955-vs-0.648 margin that turned out to be pool membership. At
   0.702-vs-0.459 with nDCG pointing the other way, the decision deserves
   re-making rather than inheriting.

**Measurements — gates re-run through `.venv`, not assumed:**

| Gate | Result |
|---|---|
| `ruff check src tests` | All checks passed |
| `ruff format --check src tests` | 55 files already formatted |
| `mypy --strict` | no issues in 32 source files |
| `lint-imports` | 2 contracts kept, 0 broken |
| `check_function_length.py` | 9 over the soft cap, none over the hard cap |
| `pytest` (fast) | 525 passed, 36 deselected, **40.5 s** |
| `pytest -m ""` (full) | 561 passed, 68.3 s |
| coverage | 87.62 % (gate 80 %) |

*Note: the first gate run was scoped wrong — `ruff check .` over the whole tree
reports 370 errors because it lints `reference/legacy/`, which is frozen by
design. The Makefile scopes every gate to `src tests`. Run `make check`, not
the bare tools.*

**Two findings from the re-run, both recorded in the Current state block:**

1. **The fast suite breaches Rules.md §5** — 40.5 s against a 30 s budget, at
   525 tests (it was 421 at 13 s). Not fixed here: it is a real decision about
   whether the budget or the suite moves, and it belongs to whoever picks up
   Sprint 11.6.
2. **`TestCrossModelGuard::test_index_from_one_model_is_rejected_by_another`
   failed once**, then passed in isolation, in its module, and in two further
   full runs. Order- or state-dependent; cause not found. Left visible rather
   than re-run until green and forgotten — a flaky *guard* test is the kind
   that gets waved through until the day it was right.

**Next action:** the product decision — 17.8-document union by default, or
BM25 alone — then Sprint 11.6 (`v1.0.0`), and a merge of
`fix/domain-audit-remediation` into `main`, which is still 16 commits behind.


---

### 2026-08-28 — Session 4 · Sprint 9 verified for the first time: the image builds, and building it found a bug

**Status: done.** Docker arrived on the dev machine, so the one sprint whose
DoD had never been checked at all could finally be checked.

**Files changed:** `deploy/docker/Dockerfile`, `deploy/docker/compose.yaml`,
`Phases.md`, `Architecture.md`, `README.md`, `deploy/README.md`, `Memory.md`.

**The defect, which is the reason this was worth doing.** The image built
first try and the container came up `healthy`. It was also completely broken:
the union retriever — the shipped default — raised
`PermissionError: 'data/interim'` on its first query. Docker creates an absent
bind-mount parent as `root:root`, so `/home/app/data` was root-owned and uid
10001 could not create the token-cache directory inside it. The embedding-only
path worked fine, because it reads a prebuilt index and writes nothing.

Three things about it are worth keeping:

1. **No test could have caught it.** It is not a code defect. It only exists at
   the intersection of a non-root user, a bind mount, and a directory the
   application creates lazily.
2. **The healthcheck was green the whole time.** `/_stcore/health` returns 200
   when Streamlit's HTTP server binds and proves nothing about retrieval. This
   is the fourth mismatch class this project has found the hard way, and the
   first one a *liveness probe* actively concealed. `doctor --full` exists and
   would have caught it — using it as the readiness probe is an open
   recommendation, not yet made.
3. **It would have shipped.** The App Service config mounts artefacts the same
   way. The first real deploy would have served a green healthcheck and an
   exception on every search.

**The fix:** pre-create `data/{raw,interim,processed}` and `models/` owned by
`app` in the runtime stage, so a bind mount lands on an existing directory and
the ownership holds; and mount `data/interim` read-write in compose, since the
token cache is the one path the container writes. Re-verified after the change.

**Measurements — all first-time:**

| | Measured |
|---|---|
| Cold build, `runtime` target | 8 min 49 s |
| Rebuild after the fix (wheels cached) | 40 s |
| `standalone` target on top | 18 s |
| Image, `runtime` | **941 MB** summed layers (`docker images` says 1.25 GB under the containerd store) |
| Image, `standalone` | +108.5 MB baked artefacts — models 46.9, raw 29.6, interim 23.3, processed 8.7 |
| Container healthy from `up` | < 25 s |
| Union cold load | 20–24 s cold, 9 s warm page cache |
| First query | 3.3 s (TF-IDF fits over 10,666 docs) |
| Warm union query | p50 **86.9 ms**, p95 **103.5 ms** — *faster than the 128 ms measured natively* |
| Warm embedding-only query | p50 2.9 ms, p95 5.1 ms |
| Peak RSS, union | **662 MB** against a 2 GB limit |
| Idle Streamlit RSS | 55–80 MB |

**Image size: the DoD is not met and cannot be met by packaging.** 941 MB
against `< 800 MB`. The venv is 725 MB of it — pyarrow 156, scipy 143, pandas
76, numpy 79, gensim 58, streamlit 35, pydeck 23 — on a ~175 MB slim base.
Trimming NLTK to English-only saved 64 MB (100 → 37 MB) and is kept; dropping
pip and setuptools would save 24 MB more and was **not** done, because the risk
of a runtime `pkg_resources` import is a poor trade for 24 MB when the target
is 141 MB away. pyarrow and pydeck are Streamlit's dependencies and scipy is
gensim's. **Under 800 MB means a different UI stack, not a better Dockerfile.**
The target was written before anything was built; it needs re-setting against
this measurement, and that is a decision for the user, not a doc edit to make
quietly.

**Also verified:** non-root uid 10001; the image contains no secret and no
artefacts (the four data directories exist and are empty); the `standalone`
target serves with no mounts at all, which is the free-tier path in
`deploy/README.md` §6, and confirms that document's "110 MB, measured" claim.

**Next action:** unchanged — the union-vs-BM25 product decision, then `v1.0.0`.
Added by this session: decide what the image-size target should be, and whether
the readiness probe should move from `/_stcore/health` to `doctor --full`.


---

### 2026-08-28 — Session 5 · Round 3: the blind spot closed, and the negation fix half-refuted

**Status: done.** The last open *measurement* question from the audit is answered.

`EVALUATION_AUDIT.md` §5 said the eval set could not see the two preprocessing
fixes, and round 2 could not help — a pool holds no candidates for a query
nobody asked. So 22 queries were written in three strata and scored separately:
`entity` (11), `code` (3), `negation` (8). Full analysis in §8 of the audit.

**Files added:** `tests/fixtures/eval_queries_round3.txt` (the query source,
with the design reasoning inline), `tests/fixtures/eval_queries_round3.json`
(the labelled fixture), `scripts/round3_probe.py`, `scripts/round3_ablation.py`,
`scripts/round3_evaluate.py`, `reports/round3_probe.json`,
`reports/round3_ablation.json`, `reports/evaluation_round3.json`,
`reports/eval_round3_candidates.json`, `reports/eval_round3_labels.json`.
**Changed:** `scripts/make_eval_candidates.py` (BM25 now contributes to the
pool — it is scored, so it must contribute, which is §6's whole lesson),
`EVALUATION_AUDIT.md`, `README.md`, `Memory.md`.

**Design decisions worth keeping:**

1. **Two of the three strata need no relevance judgements.** A document is
   relevant to `NCT04446429` iff its text contains that string; and for the
   negation pairs the measurement is the *overlap between a query and its
   negated twin*, which is a property of the rankings, not of relevance. In a
   project where 42 % of judgements are model-generated and every table has to
   say so, designing the measurement to need no labels is worth more than
   another few hundred labels would have been.
2. **Collapse pairs, not "queries with digits".** `CD4`/`CD8`, `IL-6`/`IL-1`,
   `SARS-CoV-2`/`MERS-CoV` — pairs the old chain maps to one token. A generic
   digit-bearing query would have measured almost nothing; the pair makes the
   defect visible directly.
3. **Strata are never averaged.** Known-item search with 2 relevant documents
   and ad-hoc search with 13 are different tasks; one mean over them describes
   neither. `scripts/round3_evaluate.py` slices and delegates to the shipped
   evaluator, so the numbers stay comparable with `reports/evaluation.json`.
4. **No p-values at n = 11 / 3 / 8.** These strata are diagnostic. The main set
   was sized to 97 by a power calculation for exactly this reason, and the
   effects reported are the ones readable at this n (1.00 vs 0.00).

**Results:**

- **Tokeniser fix: confirmed, large.** Registry-code Recall@10 0.44 → 1.00
  (TF-IDF) and 0.33 → 1.00 (BM25). `CD4 T cell response` and `CD8 T cell
  response` returned the *identical* ten documents under the old chain
  (overlap 1.00 → 0.50). Vocabulary 39,879 → 55,487.
- **Negation fix: half-refuted, and this is the finding I did not expect.**
  The gain comes from intra-word hyphen joining, not from `CLINICAL_KEEP_WORDS`.
  `non-hospitalized` → `nonhospitalized` is a rare token (idf **6.45** against
  `hospitalized` 3.14) and it moves the ranking hard. Free-standing negation
  does not: `not` has idf **2.18**, *below* the content words it inverts, so
  retaining it changes nothing measurable (overlap 1.00 → 0.90). And `without`
  was never an NLTK stopword, so the pair that turns on it is byte-identical
  under both chains. **No term-weighting scheme fixes this** — it needs the
  query parser to treat negation as an operator.
- **The embeddings lose by more on exactly the queries they should.** Entity
  stratum P@1: BM25/TF-IDF 0.909, FastText 0.636. An entity query turns on one
  rare alphanumeric token, which is what mean pooling destroys.
- **The union damages known-item retrieval.** Code stratum P@1: lexical 1.000,
  union-fasttext 0.500 — the embedding half contributes a document that cannot
  be relevant (FastText scores 0.000 on every metric there). A direct input to
  the pending PRD §8.4 decision.
- **A capability gap, found by accident:** only 71 of 10,666 abstracts contain
  any registry code, because ids live in the `Trial ID` column while retrieval
  runs on `abstract`. The tokeniser docstring's flagship example
  (`NCT04508933` → `nct`) is true about tokenisation and nearly irrelevant to
  retrieval here. Searching by trial id needs the id column indexed.

**One query was written and dropped:** `MERS-CoV and SARS-CoV-1 comparison` has
no relevant document in a COVID-19 trial corpus. Recorded rather than quietly
deleted — it is a fact about corpus scope, and it still serves as a
discrimination probe in the ablation, which needs no labels.

**Next action:** unchanged and now better informed — the union-vs-lexical
product decision, then `v1.0.0`.

---

### 2026-08-28 — Session 5b · The bigram fix for negation: tried, measured, rejected

**Status: done — a negative result, and worth the twenty minutes.**

Result 2 of the audit suggested its own remedy and I recommended it: if prefix
negation works because hyphen-joining mints a rare token, bigrams should do the
same for free-standing negation. `not requiring` has idf 6.00, `without
mechanical` 7.97 — the same band as `nonhospitalized` at 6.45. I said so to the
user before testing it, which was the mistake; the measurement came second.

**It does not work.** `scripts/round3_bigram_experiment.py`, feature change
only, rankers untouched, metrics from `evaluate_baseline` unchanged:

| nDCG@10 | unigram | bigram |
|---|---|---|
| negation stratum, TF-IDF | **0.528** | 0.377 |
| entity stratum, TF-IDF | **0.742** | 0.663 |
| main 97-query set, TF-IDF | **0.797** | 0.720 |
| main 97-query set, BM25 | **0.799** | 0.608 |

Negation overlap barely moved (`not requiring` 0.90 → 0.80; `without
mechanical` unchanged), one prefix pair got worse, vocabulary went 55,487 →
887,024 and p95 latency roughly tripled. Nothing to salvage.

**The reason corrects my own theory, which is why this is worth recording.**
Rare-token idf was never the whole mechanism. Split each pair into features
shared with its positive twin and features unique to the negated query:

| Negated query | shared idf mass | unique idf mass |
|---|---|---|
| `not requiring supplemental oxygen` (bigrams) | **25.3** | 12.7 |
| `non-hospitalized patients with covid-19` | 5.6 | **13.8** |

Bigrams give the negated query `not requiring` (6.0) — and give *both* queries
`requiring supplemental` (6.5) and `supplemental oxygen` (5.2). Shared evidence
grows faster than unique evidence, so the pair gets *more* similar. Prefix
negation escapes this only because morphology **substitutes**: `hospitalized`
is replaced by `nonhospitalized`, removing the shared term rather than adding
beside it.

**The generalisation, and it is the durable lesson here:** negation requires
*removing or inverting shared evidence*. Every additive feature scheme — a
stopword allowlist, bigrams, trigrams, reweighting — can only add evidence, so
none of them can express negation. The audit's original recommendation (a query
operator that subtracts) stands, and my proposed revision of it was wrong.

**Process note:** the cost of being wrong here was one script and twenty
minutes, because the harness to measure it already existed from round 3. That
is the argument for building the measurement before the fix, not after.


---

### 2026-08-29 — Session 6 · The clean-clone run: the pipeline passes, the on-ramp does not

**Status: done.** Sprint 11's last untested claim — that someone can go from
`git clone` to a working app — was finally tested, from a fresh clone of
GitHub rather than the working tree.

**The pipeline is fine.** Nothing in the code was broken. Install 3 min 44 s,
NLTK 14.5 s, full-corpus training 2 min 2 s at 353 MB peak, index build 16.9 s,
CLI search returning respiratory-failure trials, Streamlit serving HTTP 200
with an 11 KB body. `doctor` was correct in both directions — exit 1 while the
corpus was missing and RAM was short, exit 0 once both were satisfied.

**The dependency ranges resolved safely, and that is luck.** Only numpy is
capped (`<2.0`); gensim, pandas, scipy and streamlit are all open-ended
`>=`. A fresh resolve today gave gensim 4.4.0 / scipy 1.15.3 / pandas 2.3.3 —
identical to the dev venv. It could as easily not have: gensim has a history of
breaking against newer scipy. Worth pinning before anyone else clones this.

**Four defects, all between clone and corpus, none in the code:**

1. **`make` is not installed on the reference machine.** Sprint 11's DoD is
   written as `make setup && make doctor && make train && make app` and
   therefore cannot pass on the laptop it names. The README's no-make fallback
   is what actually works.
2. **"With no manual steps" is unachievable by construction.** The 29 MB corpus
   is gitignored, correctly, so a clean clone has no data and must have a
   manual acquisition step. The DoD needs rewording; the repo is right.
3. **`doctor`'s remediation points at a dead end.** It says "Run `make data`" —
   but `make` is absent, *and* `migrate_legacy.py` resolves its legacy root to
   the clone's parent and looks for `../Part_1/Data/Data/Dimension-covid.csv`,
   a layout that exists only on the original machine. It should point at the
   figshare link in README §Data.
4. **The README's no-make fallback omits the data step**, so a Windows user
   goes `doctor` → `train` and meets the missing corpus with no guidance inside
   the path they were following.

**The figures in Architecture §9 were stale and are now corrected.** They were
measured 2026-08-27, *before* the tokeniser fix. Preserving alphanumeric
identity keeps `sarscov2`, `il6` and `cd4` as distinct tokens, so:

| | 2026-08-27 (old chain) | 2026-08-29 (current) |
|---|---|---|
| Vocabulary | 24,897 | **31,189** (+25 %) |
| skipgram artefact | 10.2 MB | 12.8 MB |
| fasttext artefact | 29.3 MB | 31.9 MB |
| Full training | 2 min 22 s | 2 min 2 s |

That is the cost side of the retrieval gain in EVALUATION_AUDIT.md §8, and it
had never been written down. Both artefacts stay far inside the 150 MB cap.

**A near-miss worth recording.** The first clone went into the session
scratchpad and pip failed with `OSError [Errno 2]` while unpacking
`numpy.libs` — the path exceeded Windows' 260-character `MAX_PATH`. That was my
directory choice, not a project defect, but on a Windows-targeted project it is
worth telling users to clone somewhere short.

**Also checked, and *not* a regression:** `lung failure` returns a *renal*
failure trial at rank 1. The gap to rank 2 is 0.0303 vs 0.0301, against ±0.010
documented run-to-run variance from non-deterministic multi-worker training —
so the ordering is noise. The underlying mean-pooling weakness on the shared
token `failure` is real, already documented, and is exactly what §8's entity
stratum measured.

**On `v1.0.0`: recommended against, for now.** A `1.0.0` tag asserts stability,
and this project currently documents an unmet image-size DoD, no deployment, no
browser-driven search, judgements 42 % model-generated with no clinician
review, and a flaky guard test. Tagging that would assert what the measurements
do not support. `v0.11.0` marks the work without the claim; `release.yml` can
be exercised via `workflow_dispatch` without any tag at all.

**Next action:** unchanged — the union-vs-lexical product decision. Then, if
the four on-ramp defects are worth fixing, they are a half-hour of work each.

---

## Session — 2026-08-29 (later): the union decision, and the defect under it

**Status: the product decision is closed. The union ships on by default.**

It did not close the way the audit expected. Asked to weigh union-vs-lexical, I
went to check *why* §8 Result 4 showed the union halving known-item precision,
and the mechanism turned out to be a bug rather than a property.

**The defect.** `UnionRetriever._rank` fused with unweighted RRF and ordered by
score alone. When the two runs are disjoint — exactly the known-item case,
where the lexical run holds the answer and the embedding run holds nothing —
every document scores `1/(60+rank)`, so both runs' rank-1 documents tie at
byte-identical `0.016393`. `sorted` is stable, so the tie fell to dict
insertion order, and the embedding loop runs first. All three `code` queries
put an irrelevant embedding document at rank 1 and the relevant keyword
document at rank 2.

**The fix**: weight the keyword run (`KEYWORD_WEIGHT = 1.5`, from a sweep that
saturates there) and break remaining ties consensus → keyword → embedding.
Recall is unchanged on every stratum and on the main set — reordering cannot
change a set — which makes it the control for everything that did move:

| union-fasttext | MRR@10 | nDCG@10 | R-prec |
|---|---|---|---|
| main 97-query | 0.890 → 0.923 | 0.746 → 0.762 | 0.616 → 0.639 |
| `entity` | 0.788 → 0.864 | 0.725 → 0.751 | 0.590 → 0.604 |
| `code` | 0.500 → **1.000** | 0.649 → **1.000** | 0.722 → **1.000** |
| `negation` | 0.513 → 0.532 | 0.519 → 0.529 | 0.372 → 0.427 |

**Two measurement errors corrected alongside it.** The union rows' "P@1" is
actually **P@2** (`depth_factor=2`), which is why an arithmetically impossible
0.500 appeared over n=3. And nDCG@10 was never comparable across depth
factors — the union's ideal sums ~17 slots against the baselines' 10. Scored at
an equal ten-document budget the union reaches **nDCG@10 0.789** against BM25's
0.799: level, not behind. So "the union is a wider net, not a better ranker"
survives; "it costs ranking quality" does not.

**A second defect, found by tripping over it.** `scripts/round3_evaluate.py`
called `run_evaluation` per stratum, and `run_evaluation` unconditionally wrote
`reports/evaluation.json` — so scoring the strata silently overwrote the main
report with whichever stratum ran last. It had been leaving `evaluation.json`
holding the 8-query negation numbers under the label of a 97-query run.
`run_evaluation` now takes `report_name`, and the script passes `None`.

**Honest limits on this session's numbers.** The fix was measured on the same
eval set it was diagnosed from. The `code` result needs no inference — exact
ground truth, 1.000 against 0.500 — but the main-set deltas (+0.016 nDCG,
+0.023 R-precision) are single unreplicated runs, not significance-tested, and
sit inside the ±0.010–0.014 retraining noise band. The defect is unambiguous
and the direction is right on all four sets; the main-set magnitude is not
established. Replicating it is the first item on the eval backlog.

**Gates**: ruff, ruff-format, mypy --strict, import-linter, function-length all
green; 567 tests pass (was 561), coverage 88%.

---

## Session — 2026-08-29 (later still): the pre-deployment close-out

Asked to close everything outstanding before deployment, with real
experimentation rather than assertion. Four things came out of it.

**1. The "flaky guard test" was misdiagnosed, and much bigger than recorded.**
Not order-dependence. `trainer._check_memory_budget` compared free RAM against
`predicted_gb + 0.5`, and `predicted_gb` models only the n-gram matrix -- a
fixed ~20 MB whatever the corpus size. So the bare `+ 0.5` *was* the whole
requirement, and a 20-row toy run demanded as much free RAM as the full
10,666-document one. Under memory pressure **eleven** tests fail intermittently,
not the one on record. The message was unreadable too: "peak is ~0.00 GB but
only 0.42 GB is available". The requirement now scales with document count and
prints its breakdown. Eight consecutive integration runs pass, against two
failures in eight before. Same defect class as `TestScaledMemoryFloor` in
`runtime.require_memory` -- the trainer kept its own copy and was missed.

**2. Known-item retrieval did not exist, and nobody had measured it.** Sixty
real trial ids, five per registry across all twelve registries: the shipped
retriever returned the requested trial **0 times**. Ids are unique, so the
ground truth is exact and needs no annotator -- the only stratum in the project
with no provenance to declare. An identifier is a key, so it now gets a lookup
that precedes the ranking: **0/60 → 60/60 at rank 1**. Round 3's `code` stratum
falls MRR@10 1.000 → 0.667 as a result, because it scores a different question
(which trials *cite* this id) and its gold excludes the queried trial by
construction. **The gold was not rewritten to hide that.**

**3. Free-standing negation, built as an operator and measured.** Cue-and-scope
on both sides, NegEx style. Two iterations, both from inspecting what the
filter removed:

* Grammatical-negation cues alone removed five of six gold documents, all
  phrased as *avoidance* -- "reduces the need for", "decrease the need of". In
  clinical abstracts the negated sense is carried by those verbs, not by "not".
* A one-word scope is too blunt. "spread by people without symptoms" parsed to
  "exclude anything mentioning symptoms" and took main-set Recall@10 from 0.702
  to 0.698, under target, on one query. Scopes now need two tokens.

Pair overlap 0.55 → 0.33, entirely on the two free-standing pairs; the prefix
pairs do not move, which tests the mechanism claim directly. Fires on 0 of 97
main-set queries. Costs Recall@10 0.643 → 0.560 on the stratum it targets --
recorded, not hidden.

**4. Reproducibility and the on-ramp.** Every dependency now has an upper bound
(only numpy was capped); `deploy/requirements.lock` pins the closure and the
image installs against it as a constraint. Coverage left the default pytest
addopts -- it cost ~25 s of a 40.5 s run while the gate actually lives in CI and
`make test-all`, so Rules.md section 5 is met honestly at **18.7 s** rather than
by moving the budget. Sprint 11's DoD was reworded to something achievable, and
the README's no-make fallback gained the corpus step it omitted.

**Also found and fixed:** `run_evaluation` unconditionally wrote
`reports/evaluation.json`, so `round3_evaluate.py` silently overwrote the main
report with whichever stratum ran last.

**What is honestly still open.** The Docker image-size DoD (941 MB against
< 800 MB) could not be touched -- the Docker daemon was down all session. No
browser-driven search. The lock was resolved on Windows/py3.10 and needs
regenerating on linux/py3.11. And the negation cue lexicon was extended after
inspecting failures on two queries, so it is fitted to them: the mechanism is
validated, the lexicon's coverage on unseen negations is not.

---

## Session — 2026-08-29: the image-size DoD, met

Docker was started, so Sprint 9's last open clause could finally be measured
rather than argued about.

**The recorded conclusion was wrong, and wrong in an instructive way.**
Phases.md said the 141 MB over the DoD was *"not reachable by packaging"* and
that meeting it *"means changing the UI stack, not the Dockerfile"*. That was
reached by listing package sizes -- pyarrow 156, scipy 143, pandas 76 -- and
observing that each is a hard dependency. True, and beside the point: nobody
had looked *inside* the packages.

Measured in the image:

| | |
|---|---|
| bundled pytest suites (scipy, numpy, pandas) | **132 MB** |
| debug symbols across 272 `.so` files | **~70 MB** |
| pip, setuptools, pkg_resources | **25 MB** |
| pyarrow C++ headers | **6 MB** |

**954 MB → 721 MB. DoD met, 79 MB of headroom, UI stack untouched.**
`standalone` is 829 MB. Verified after stripping: every library imports,
numpy/scipy/pandas/pyarrow all compute, the container serves (healthy in ~2 s
under `--memory=2g`), and ordinary, known-item and negated queries return
exactly what the host returns.

`__pycache__` is another 141 MB and is **deliberately kept**:
`PYTHONDONTWRITEBYTECODE=1` is set in the runtime stage, so deleting it makes
every cold start recompile the dependency tree, and App Service F1 has no
Always On. Available if the space is ever worth more than the latency.

**The lock is now verified on the target platform.** Freezing inside the image
(linux/CPython 3.11.16) produced an identical set to the Windows/3.10
resolution but for `exceptiongroup` -- a backport 3.11 does not need, harmless
as a constraint -- and the project's own self-reference, which was excluded.
`make lock` regenerates it from inside the image.

**Still open:** a browser-driven search. No browser driver is installed, and
adding playwright to the dev extras was not done unasked. The app was verified
serving (health 200, page renders) and its exact search call was driven
in-process across all three query types, in both union and no-union modes --
which is not the same thing as a real browser, and is not claimed to be.
