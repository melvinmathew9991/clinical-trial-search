# Rules — Medical Embeddings Search

> Version: `1.1.0` · Last updated: `2026-08-27`
> Boundaries for any contributor, human or AI. **Read this before writing code.**
> When this file and a prompt disagree, this file wins — say so and ask.

---

## 0. The five rules that matter most

1. **Never train, index, or load with more than 3 workers.** The dev machine has 4 logical cores. One stays free. Always.
2. **Never set or leave `bucket` unbounded on FastText.** It is an 800 MB artefact by default. See ADR-001.
3. **Never put a secret, SAS token, connection string, subscription id, or storage account name in a tracked file.**
4. **Never mutate a caller's DataFrame.** Return a new object.
5. **Never widen scope.** If the task is Sprint 3, do not "helpfully" also build Sprint 5.

---

## 1. Libraries

### Use
| Purpose | Library | Notes |
|---------|---------|-------|
| Embeddings | `gensim>=4.3` | The `KeyedVectors` API, not the deprecated `Word2Vec.wv` attribute juggling |
| Arrays | `numpy>=1.24` | `float32` everywhere; `float64` doubles memory for no retrieval benefit |
| Tabular | `pandas>=2.1` | Always with `usecols=`; prefer `engine="pyarrow"` |
| Text | `nltk` | Stopwords + `WordNetLemmatizer` only |
| CLI | `typer` | |
| Config | `pydantic-settings` | |
| UI | `streamlit>=1.30` | |
| Logging | stdlib `logging` | Configured once in `logging_conf.py` |
| Sparse arrays | `scipy` | TF-IDF baseline only. Dense would be ~1.7 GB at 40k terms x 10.6k docs |
| Tests | `pytest`, `pytest-cov` | |
| Tooling | `ruff`, `mypy`, `import-linter`, `pre-commit` | |

### Do not use
| Banned | Reason | Use instead |
|--------|--------|-------------|
| `plotly` in the UI hot path | Serialises 10k abstracts to JSON on every rerun | `st.dataframe` |
| `scikit-learn` at runtime | Needed only for optional PCA; its TF-IDF was not worth the dependency when `scipy.sparse` does the job in 40 lines | `[viz]` extra; `search/baseline.py` |
| `faiss`, `pinecone`, `chromadb` | Overkill at 10k docs; both memory and a dependency we do not need | numpy matmul |
| `torch`, `transformers`, `sentence-transformers` | Explicit PRD non-goal; will not fit the memory budget | — |
| `tensorflow`, `keras` | Same | — |
| `os.path` string joins | Breaks across OS; caused a real reference/legacy/modular-code bug | `pathlib.Path` |
| `pickle` for model artefacts | Unsafe, version-fragile; Part 2 tried to `pickle.load` a gensim `.bin` | `KeyedVectors.save/load` |
| `requests` / `urllib` to fetch from Blob | No auth story | `azure-storage-blob` + `DefaultAzureCredential` |
| Bare `print()` in `src/` | Unstructured, unfilterable | `logger.info()` |
| `from x import *` | Shadows names silently | Explicit imports |
| Global mutable state | Untestable | Dependency injection via constructor |

**Adding a dependency requires:** a line in `pyproject.toml` with a lower bound, a note in
[Architecture.md §5](./Architecture.md#5-tech-stack) saying what it replaces, and a check
that it does not pull in a >100 MB transitive wheel.

---

## 2. System-resource rules

These are **hard** limits, not guidance. The target machine is 4 threads / 7.89 GB RAM.

### Must
- Call `runtime.configure_threads()` **before** the first numpy import in every entrypoint.
- Default `workers = max(1, os.cpu_count() - 1)`, capped at `settings.workers`.
- Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` to `1`.
- Read only required columns: `pd.read_csv(path, usecols=CorpusSchema.required())`.
- Stream corpora as generators. If you write `list(...)` around a 10k-document generator, justify it in a comment or delete it.
- Store vectors as `float32`. Cast explicitly: `.astype(np.float32, copy=False)`.
- Chunk any loop that allocates per document; flush every 1,000 items.
- Persist arrays as `.npy`; load large ones with `mmap_mode="r"`.
- Free big intermediates: `del frame; gc.collect()` at stage boundaries.
- Log elapsed time and peak RSS at every stage boundary.

### Must not
- Load a full gensim model at serving time when `KeyedVectors` suffices.
- Rebuild a vocabulary list inside a per-document loop. Build a `frozenset` once.
- Run a Python `for` loop to compute similarity across the corpus. One matmul.
- Sort the full similarity array to take the top 10. Use `np.argpartition`.
- Hold the raw frame, the token cache, and the vector matrix alive simultaneously.
- Start a training run without `medsearch doctor` passing.
- Leave the Streamlit file watcher on.

### Memory checklist before merging anything that touches the pipeline
```
[ ] Peak RSS measured and recorded in the PR description
[ ] Stage stays inside the Architecture.md §9 budget
[ ] No new artefact over 150 MB
[ ] `--limit 2000` profile still completes in under 2 minutes
```

---

## 3. Code standards

- **Python 3.10+ syntax.** `X | None`, not `Optional[X]`. `list[str]`, not `List[str]`.
- **Type-annotate every public function.** `mypy --strict` must pass on `src/medsearch`.
- **Docstrings** on every public module, class, and function. Google style. State units and shapes: `(n_docs, dim) float32`.
- **Functions do one thing.** Soft cap 40 lines; hard cap 60. `top_n()` in the legacy code did seven things in 45 lines — that is the anti-pattern.
  Enforced by `scripts/check_function_length.py` (`make lengths`, pre-commit, CI). It counts body lines only — docstrings, blanks, and comment-only lines are excluded, so documenting a function well never costs against the cap. Soft-cap hits warn; hard-cap hits fail. Unenforced from Sprint 0 to Sprint 8, during which two functions crossed the hard cap unnoticed.
- **Descriptive names.** No `K`, `K1`, `K11`, `KK`, `p`, `x`, `tmp`, `res`, `L`. If a reviewer must scroll up to learn what a variable holds, rename it.
- **`@dataclass(frozen=True, slots=True)`** for value objects. `slots` measurably cuts per-instance memory.
- **Pure functions by default.** Side effects live in `pipelines/` and `cli.py`.
- **No commented-out code.** Git remembers.
- **No notebook code in `src/`.** No `# In[24]:`, no `%run`, no `!pip install`, no `nltk.download()` at import time — downloads happen once in `runtime.ensure_nltk_data()`.
- **Line length 100.** Enforced by `ruff format`.

---

## 4. Error handling

**Exception hierarchy** — every raised error is one of these:

```python
MedSearchError                 # base — never raised directly
├── ConfigurationError         # bad/missing settings, invalid CLI choice
├── DataError
│   ├── CorpusNotFoundError
│   ├── SchemaValidationError  # names the offending columns
│   └── EmptyCorpusError       # loaded, but no usable text
├── ModelError
│   ├── ModelNotTrainedError
│   ├── ArtefactMismatchError  # index fingerprint ≠ model fingerprint
│   └── StaleIndexError        # corpus changed after the index was built
├── IndexBuildError            # index absent, incomplete, or malformed
└── ResourceError              # insufficient RAM before a stage
```

**Rules**
- Fail fast and loudly at boundaries: validate on load, not three layers down.
- Never `except:` or `except Exception:` without re-raising or logging with `exc_info=True`.
- Never swallow an error to return a default. The legacy `get_mean_vector` returned `np.array([0]*100)` for an all-OOV document, which then produced a `0/0` NaN in cosine similarity that silently ranked as `nan`. Return a typed result that says *why* it is empty.
- Error messages state **what failed, what was expected, and what to do**:
  ```
  SchemaValidationError: corpus is missing required column 'Abstract'.
    Found: ['Date added', 'Trial ID', 'Title', ...]
    Expected at least: ['Trial ID', 'Title', 'Abstract', 'Publication date']
    Fix: check that data/raw/dimension-covid.csv is the Dimensions export, not a filtered subset.
  ```
- User-facing surfaces (CLI, Streamlit) catch `MedSearchError` and render `str(e)`. They never show a traceback. Everything else propagates.
- Guard divide-by-zero explicitly: if `norm(v) == 0`, return no results with a reason.
- `ResourceError` is raised *before* an expensive stage, never mid-run.

---

## 5. Testing

- Every `src/medsearch` module has a `tests/unit/test_<module>.py`, with two stated exemptions: `_typing.py` (type aliases only — nothing to execute) and `app/` (Streamlit callbacks, excluded from the coverage gate and covered by the container health check instead). Everything else needs its own file; `exceptions.py`, `logging_conf.py`, and `search/baseline.py` went eleven sprints on transitive coverage alone, and a pre-deployment audit found real defects in two of the three.
- **Unit tests never touch the network, the 29 MB corpus, or a trained model.** Use `tests/fixtures/sample_corpus.csv` (20 rows) and the toy vectors built in `conftest.py`. The default `pytest` run is unit-only and must stay under 30 seconds.
- **Integration tests may train**, over the 20-row fixture at 16 dimensions and one epoch. They are marked `slow` so the fast loop skips them; `make test-all` and CI run them, and that is where the coverage gate is enforced. Anything that trains a model belongs there, not in `tests/unit/`.
- Test names describe behaviour: `test_engine_returns_empty_when_query_is_all_oov`.
- Every bug fix ships with a regression test. The four legacy bugs each get one:
  - `test_index_written_for_fasttext_uses_fasttext_vectors` (the `K1`/`K2` swap)
  - `test_artefact_paths_are_case_consistent` (the `FastText`/`Fasttext` mismatch)
  - `test_load_corpus_returns_all_rows_by_default` (the hidden `.iloc[:100]`)
  - `test_preprocess_does_not_mutate_input_frame` (chained assignment)
  - `test_load_raises_stale_index_error` (corpus replaced after indexing)
  - `test_the_advertised_fallback_is_actually_permitted` (flat memory floor)
  - `test_sampled_limit_parsing` (sampled index paired with the full corpus)
- Coverage floor 80 % on `src/medsearch`, excluding `app/`. CI fails below it.
- Mark anything slow `@pytest.mark.slow`; excluded from the default run.

---

## 6. Git

- **Trunk-based.** `main` is always releasable. No long-lived branches.
- **Branch:** `<type>/<sprint>-<slug>` — `feat/s3-fasttext-trainer`, `fix/s4-index-fingerprint`.
- **Conventional Commits**, imperative mood, scoped:
  ```
  feat(embeddings): bound fasttext bucket to 50k
  fix(search): guard zero-norm query vectors
  perf(search): replace per-doc cosine loop with single matmul
  docs(memory): log sprint 3 completion
  chore(deps): pin gensim to >=4.3
  ```
- Body explains **why**, not what. The diff shows what.
- One logical change per commit. No "misc fixes".
- **Never commit:** anything under `data/`, `models/`, `reports/`, `.env`, `*.bin`, `*.npy`, `*.kv`, `__pycache__/`, `.ipynb_checkpoints/`, or a notebook with outputs.
- Every commit must pass `pre-commit` locally. Do not `--no-verify`.
- Tag each completed sprint: `v0.<sprint>.0`.

---

## 7. Documentation

These five files are the project's contract. Keep them true.

| File | Update when |
|------|-------------|
| `PRD.md` | Requirements change, scope shifts, a non-goal becomes a goal |
| `Architecture.md` | A module moves, a dependency changes, a design decision is made (add an ADR) |
| `Rules.md` | A convention is agreed or a library is banned/adopted |
| `Phases.md` | A sprint's scope, status, or ordering changes |
| `Memory.md` | **Every working session.** Append; never rewrite history |

A PR that changes structure without updating `Architecture.md` is incomplete.

---

## 8. AI agent boundaries

### Always
- Read `PRD.md` → `Architecture.md` → `Rules.md` → `Phases.md` → `Memory.md` before the first edit of a session.
- Work on **one sprint at a time**, in the order in `Phases.md`.
- Check `Memory.md` for what is already done before writing anything.
- Update `Memory.md` at the end of the session with: sprint, files touched, decisions, measured RSS/timing, and the next step.
- State assumptions explicitly when a requirement is ambiguous.
- Run `ruff check`, `mypy`, and `pytest` before declaring work complete.
- Report honestly. If tests fail, say so and paste the output.

### Never
- Install a dependency not listed in §1 without asking.
- Refactor code outside the current sprint's scope.
- Rewrite `Memory.md` history or delete past entries.
- Commit, push, or tag unless explicitly asked.
- Touch anything under `Part_1/` or `Part_2/` — they are frozen legacy reference.
- Delete anything under `data/raw/`.
- Invent benchmark numbers. Measure, or say "not measured".
- Claim a sprint is done when only part of it is. Say what is left.
- Bypass §2 resource rules for convenience.

### When blocked
State what is blocking, what you tried, and the options — then ask. Do not guess at a
requirement and build the wrong thing for an hour.

---

**Change log**

| Date | Version | Change |
|------|---------|--------|
| 2026-08-27 | 1.0.0 | Initial rules |
| 2026-08-27 | 1.1.0 | Added §0 and §2 system-resource rules after profiling the dev laptop |
