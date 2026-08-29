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
| G1 | Semantic retrieval beats keyword baseline | ❌ **REFUTED.** On the re-judged set, FastText Recall@10 0.353 against TF-IDF 0.459 and BM25 0.471; nDCG@10 0.662 against 0.797 / 0.799. See §8.1 and [EVALUATION_AUDIT.md](./EVALUATION_AUDIT.md) |
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
| F-42 | Answer a registry-id query with that trial, at rank 1 | **P0** |
| F-43 | Honour free-standing negation (`not X`, `without X`) as an exclusion, not a term | P2 |

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

> ⚠️ **These numbers were superseded by a re-judged evaluation set.** The
> original pool was built from the systems being scored, so `Recall@10 0.955`
> measured pool membership, not retrieval quality. All 1,532 outstanding
> candidates have since been judged (986 → 1,691 judgements). On the corrected
> set: **BM25 0.471, TF-IDF 0.459 — statistically indistinguishable (p = 0.47)**,
> and by nDCG@10 the lexical baselines *beat* the union (0.799 vs 0.746). The
> judgements are model-generated, calibrated at **Cohen's κ = 0.800** against
> the original human labels. Full analysis:
> **[EVALUATION_AUDIT.md](./EVALUATION_AUDIT.md)**.


A held-out eval set of query → relevant-trial-id pairs lives at `tests/fixtures/eval_queries.json`.

Measured over **97 labelled queries / 1,691 relevance judgements** — the
round-2, re-judged set. The union column is what ships (§8.4); the
single-ranker columns are why.

| Metric | TF-IDF | BM25 | Skip-gram | FastText | **Union (FastText + TF-IDF)** | Target | Met? |
|--------|--------|------|-----------|----------|-------------------------------|--------|------|
| Recall@10 | 0.459 | 0.471 | 0.351 | 0.353 | **0.702** | ≥ 0.70 | ⚠️ *via the union only, at a 20-document budget* |
| nDCG@10 | 0.797 | **0.799** | 0.655 | 0.662 | 0.746 | — | — |
| R-precision | 0.449 | 0.458 | 0.346 | 0.351 | **0.616** | — | — |
| MRR@10 | **0.952** | 0.909 | 0.818 | 0.818 | 0.890 | ≥ 0.45 | ✅ |
| Precision@1 | **0.918** | 0.856 | 0.742 | 0.732 | 0.830 | — | — |
| p95 latency | 3.5 ms | 13.4 ms | 2.4 ms | 2.5 ms | 122 ms | < 300 ms | ✅ |
| Docs returned | 10.0 | 10.0 | 10.0 | 10.0 | 17.8 | — | — |
| Unanswered | 0 | 0 | 0 | 0 | 0 | < 5 % | ✅ |
| Peak training RSS | — | — | ~350 MB | ~350 MB | — | ≤ 2.5 GB | ✅ |

**Read the recall row against the docs row, and against the ceiling.** The
union is scored to depth 20 because a union of two top-10 lists *is* a
20-document budget. It is not a like-for-like win over TF-IDF@10; it is a
deliberately larger result set, and it costs Precision@1 (0.830 against
TF-IDF's 0.918) to buy the recall. With 17.4 relevant documents per query the
attainable Recall@10 is **0.626** at depth 10 and 0.951 at depth 20, so no
depth-10 system can approach 0.70 on this set at all. And on nDCG@10 — the one
metric here not capped by that ceiling — both lexical baselines *beat* the
union. **The DoD is met on the number and not on the reading:** what the union
provides is a wider net, not better ranking. See
[EVALUATION_AUDIT.md](./EVALUATION_AUDIT.md) §7.

An earlier version of this table carried the n=30 numbers (Recall@10 0.647 /
0.559 / 0.539, Precision@1 0.900 / 0.700 / 0.667). Those are superseded — the
eval set was resized to 97 by a power calculation in §8.1.

`medsearch evaluate` computes these and writes `reports/evaluation.json`.

### 8.1 G1 is not met: the keyword baseline wins, confirmed at adequate power

Measured 2026-08-28 over **97 labelled queries / 986 relevance judgements**
— *round 1, on the pool these systems helped build. The direction survives
re-judging (TF-IDF 0.459 against FastText 0.353) but the intervals below have
not been recomputed; see [EVALUATION_AUDIT.md](./EVALUATION_AUDIT.md) §7.*
The set was sized by a power calculation after a first attempt at n=30 proved
underpowered (95% CI on the recall difference then: [−0.199, **+0.014**]).

| Method | Recall@10 | MRR@10 | Precision@1 |
|--------|-----------|--------|-------------|
| **TF-IDF baseline** | **0.648** | **0.888** | **0.680** |
| FastText | 0.485 | 0.761 | 0.515 |
| Skip-gram | 0.469 | 0.757 | 0.515 |

**Every gap is statistically significant and survives Bonferroni correction**
(α = 0.0167 across three metrics), paired over the same queries with 20,000
bootstrap resamples and 20,000 permutations:

| Comparison | Δ | 95% CI | p |
|------------|---|--------|---|
| Skip-gram − TF-IDF, Recall@10 | −0.179 | [−0.256, −0.101] | <0.0001 |
| FastText − TF-IDF, Recall@10 | −0.163 | [−0.249, −0.078] | 0.0003 |
| Skip-gram − TF-IDF, MRR@10 | −0.131 | [−0.208, −0.054] | 0.0014 |
| FastText − TF-IDF, MRR@10 | −0.127 | [−0.208, −0.047] | 0.0031 |
| Skip-gram − TF-IDF, P@1 | −0.165 | [−0.278, −0.062] | 0.0058 |
| FastText − TF-IDF, P@1 | −0.165 | [−0.278, −0.052] | 0.0082 |

Per query, TF-IDF wins 56, the embeddings win 28, 13 ties — not the 13–13 tie
the underpowered set suggested. **FastText and Skip-gram are statistically
indistinguishable from each other** on all three metrics (p = 0.28, 0.86, 1.00),
so the character n-grams buy nothing measurable here.

**Conclusion: the project's central premise does not hold as built.** In-domain
mean-pooled word embeddings are beaten by a 40-line TF-IDF baseline on this
corpus and this query set.

**The paraphrase advantage is real but narrow.** Embeddings still win where no
lexical overlap exists — `mechanical ventilation weaning` retrieves
ventilation-liberation trials although no abstract contains "weaning". That
capability simply does not compensate, because most realistic clinical queries
contain a rare, decisive term (`colchicine`, `remdesivir`, `steroid`,
`ECMO`) that IDF weights heavily and mean pooling averages away.

**Diagnosis at the time: mean pooling.** Averaging treats `colchicine` and
`patients` as equally informative, discarding the signal IDF supplies.
**This diagnosis was tested and refuted — see §8.2.** Reweighting the
average does not recover what a 100-dimensional dense vector loses.

**Remediation candidates, in the order they were tried:**
1. ~~**SIF / IDF-weighted document vectors**~~ — implemented and **measured to
   fail**. See §8.2. The diagnosis above turned out to be wrong.
2. ~~**Hybrid retrieval**~~ — rank fusion measured at **+0.0005 (p = 0.98)**
   before being built. See §8.3. The complementarity is real but cannot be
   cashed in by reranking at a fixed result budget.

The powered eval set did its job: it was able to say that SIF does not work,
which the n=30 set could not have.

### 8.2 SIF weighting was implemented and does NOT work

The §8.1 diagnosis said the failure was mean pooling, and named SIF /
IDF-weighted document vectors as the fix. That was implemented (ADR-011,
`embeddings/weighting.py`) and measured on the same 97 queries. **It does not
close the gap, and for FastText it makes things significantly worse.**

> *Round-1 numbers throughout §8.2 and §8.3, on the pool the scored systems
> helped build. SIF and RRF were not re-scored on the re-judged set — both were
> killed, and re-scoring a dead branch buys nothing. The comparisons here are
> internally consistent (every system in them contributed to that pool), which
> is exactly the condition §6 of the audit says must hold.*

| Method | Recall@10 mean | Recall@10 SIF | Δ | |
|--------|----------------|---------------|---|---|
| Skip-gram | 0.469 | 0.450 | −0.019 | ns (p = 0.35) |
| FastText | 0.485 | 0.432 | −0.053 | **significant** (CI [−0.102, −0.007]) |
| TF-IDF baseline | 0.648 | — | — | still ahead of everything |

An ablation separates SIF's two steps and shows the weighting is what fails,
not the implementation of the second step:

| Variant | Skip-gram R@10 | FastText R@10 |
|---------|----------------|----------------|
| mean pooling | 0.469 | 0.485 |
| SIF weights only | 0.447 | 0.432 |
| SIF weights + common-component removal | 0.450 | 0.432 |

Common-component removal changes nothing measurable either way
(Δ = +0.003 / −0.001, both ns). The frequency weighting itself is what costs
recall.

**The §8.1 diagnosis was wrong.** "The failure is mean pooling, not the
embeddings" implied a fixable pooling defect. The evidence says otherwise, and
the better explanation is:

1. **Reweighting cannot recover lexical precision from a dense average.**
   TF-IDF matches exactly in a 40,012-dimensional sparse space where
   `colchicine` is its own coordinate. Upweighting `colchicine` inside a
   100-dimensional average still blurs it against every other word in the
   abstract. Dimensionality, not weighting, is the binding constraint.
2. **Upweighting rare words amplifies their noise.** Rare words have the
   fewest training examples and therefore the least reliable vectors. SIF puts
   the most weight on exactly the least trustworthy directions — which also
   explains why FastText suffers most, since its rare-word vectors are
   synthesised from character n-grams rather than observed directly.

**The remaining candidate is the hybrid**, previously second on the list and
now the only one still standing: keep TF-IDF for lexical precision and add
embeddings for the 28 queries they win. That combination does not require the
dense representation to do something it structurally cannot.

SIF stays in the codebase — selectable via `--pooling sif`, defaulting off,
and covered by tests — so the negative result stays reproducible rather than
becoming folklore.

### 8.3 The hybrid was measured before being built — and is not worth building

SIF failed (§8.2), leaving the TF-IDF + embedding hybrid as the last candidate.
Rather than build it, rank fusion was measured directly. It does not work
either, but the reason is worth recording because it points somewhere useful.

**Reciprocal Rank Fusion at a fixed budget of 10 documents:**

| Method | Docs returned | Recall@10 | vs TF-IDF |
|--------|---------------|-----------|-----------|
| TF-IDF | 10 | 0.648 | — |
| FastText | 10 | 0.485 | — |
| **RRF over both top-10 lists** | 10 | **0.648** | **+0.0005, p = 0.98** |
| RRF over both top-30 lists | 10 | 0.560 | −0.087, **significantly worse** |

Fusion delivers *nothing*. Going deeper into the embedding ranking actively
hurts, because it trades reliable TF-IDF hits for unreliable embedding ones.

**But the complementarity is real, and it is not a depth artefact:**

| Method | Docs returned | Recall |
|--------|---------------|--------|
| TF-IDF, depth-matched | 20 | 0.715 |
| **Union of both top-10 lists** | 20 | **0.955** |

Union beats depth-matched TF-IDF by **+0.240 (p < 0.0001)**. Of the 496
relevant documents the embeddings retrieve, **326 (66%) are ones TF-IDF never
returns**, and on **80 of 97 queries** the embeddings contribute at least one
document TF-IDF misses.

**The resolution.** The two methods genuinely find different relevant
documents, but at a fixed budget of ten results a fusion must *drop* a TF-IDF
hit to admit an embedding hit, and TF-IDF's are more often correct. The
complementarity is real and simultaneously unusable by reranking alone.

**What this actually implies — a product decision, not a modelling one.**
Cashing in the complementarity requires either

1. **A larger result budget.** Returning the 20-document union instead of
   TF-IDF's top 10 lifts recall from 0.648 to **0.955** — comfortably past the
   0.70 target — at the cost of showing twice as many results. For a research
   tool whose user scans a list of trials, that trade is probably right, and it
   requires no new modelling at all. *On the re-judged set the same move reads
   0.459 → 0.702: still past the target, but only just, and the union's nDCG@10
   (0.746) falls below TF-IDF's (0.797). The trade is real; it is a coverage
   gain paid for with ranking quality.*
2. **A second-stage reranker** over the 20-candidate union. That means a
   cross-encoder, which PRD §5 excludes as a non-goal and which will not fit
   the memory budget in G6.

**Recommendation: do not build the fusion hybrid.** It is measured at +0.0005.
Take route 1 if the product can show 20 results; otherwise ship TF-IDF as the
ranker and keep the embedding index for the paraphrase queries it uniquely
serves.

### 8.4 What ships: the union, FastText, and the defaults left alone

**The union ships, on by default — decision re-made 2026-08-29 and confirmed.**
On the re-judged set: Recall@10 **0.702** against the 0.70 target, R-precision
**0.639**, MRR@10 0.923 against the 0.45 target, nDCG@10 0.762, p95 128 ms
against 300 ms. It is switchable (`--no-union`, and a sidebar toggle) because it
buys that recall with ~17.8 results instead of 10.

The decision was reopened when re-judging cut the union's margin from 0.955-vs-0.648
to 0.702-vs-0.459 and the ranking metrics appeared to point the other way. Two
things settled it.

**First, the strongest argument against the union was a defect, not a property.**
The known-item damage recorded in EVALUATION_AUDIT §8 Result 4 — the union
halving MRR@10 to 0.500 where the lexical ranker scored 1.000 — came from an
exact tie in the fusion: two runs sharing no documents award identical RRF
scores at identical ranks, and `sorted` settled it by dict insertion order,
which put the embedding run first. Weighting the keyword run (`KEYWORD_WEIGHT
= 1.5`) and breaking remaining ties explicitly takes the `code` stratum to
**MRR@10 1.000 and nDCG@10 1.000**, level with the lexical baselines, and lifts
every other stratum too. Recall is unchanged throughout, because reordering
cannot change a set.

**Second, the nDCG comparison was never like-for-like.** The union is scored to
depth 20, so its nDCG ideal sums ~17 slots against the baselines' 10
(`evaluate.py`). Truncated to an equal ten-document budget the union scores
**nDCG@10 0.789** against BM25's 0.799 and TF-IDF's 0.797 — level, not behind.

| method | docs | nDCG@10 | MRR@10 | R@10 | R-prec |
|---|---|---|---|---|---|
| BM25 | 10 | **0.799** | 0.909 | 0.471 | 0.458 |
| TF-IDF | 10 | 0.797 | **0.952** | 0.459 | 0.449 |
| union-fasttext, truncated to 10 | 10 | 0.789 | 0.923 | 0.459 | 0.451 |
| **union-fasttext (shipped)** | 17.8 | 0.762 | 0.923 | **0.702** | **0.639** |
| fasttext alone | 10 | 0.662 | 0.818 | 0.353 | 0.351 |

So the honest statement is the one this section always should have made: **the
union is not a better ranker, it is a wider net, and it no longer costs ranking
quality to use.** The remaining trade is one-directional — 17.8 documents for
Recall@10 0.702 and R-precision 0.639, against 10 documents for 0.471 and
0.458. For the researcher §2 describes, who must not miss a relevant trial,
that is the right default; for anyone wanting the tightest list, the toggle is
there. See [EVALUATION_AUDIT.md](./EVALUATION_AUDIT.md) §§7–9.

**FastText is the default model, not Skip-gram.** As standalone rankers the two
are indistinguishable (p = 0.28 / 0.86 / 1.00, §8.1), which is why the choice
looked arbitrary. Under the union it is not:

| Comparison | Skip-gram | FastText | Effect | 95% CI | p |
|---|---|---|---|---|---|
| Union Recall@10 | 0.927 | **0.955** | +0.028 | [+0.005, +0.052] | **0.019** |
| Union MRR@10 | 0.822 | 0.852 | +0.030 | [−0.011, +0.073] | 0.157 |

*Round-1 intervals. On the re-judged set the same gap is 0.702 against 0.687
(+0.015) and 0.746 against 0.733 on nDCG@10 — same direction, not re-tested for
significance. FastText stays the default; nothing here argues for switching.*

Better on recall, no worse on ranking, for a 29.3 MB artefact against 10.2 MB —
both far inside the 150 MB cap. Reproduce with
`python scripts/significance.py union-skipgram union-fasttext`, which is
committed precisely so these numbers stop being ad-hoc.

**The hyperparameter sweep changes nothing, and the noise floor is why.**
One-factor-at-a-time around the shipped defaults, full corpus, 97 queries
(`python scripts/sweep.py` → `reports/sweep.json`):

| Config | Recall@10 | Δ | Union Recall@10 | Δ |
|---|---|---|---|---|
| baseline (100 / 5 / 2 / 5) | 0.4548 | — | 0.9089 | — |
| `vector_size=200` | 0.4722 | +0.017 | 0.9042 | −0.005 |
| `vector_size=300` | 0.4715 | +0.017 | 0.9005 | −0.009 |
| `window=10` | 0.4940 | +0.039 | 0.9223 | +0.013 |
| `min_count=5` | 0.4793 | +0.025 | 0.9183 | +0.009 |
| `epochs=15` | 0.4946 | +0.040 | 0.9040 | −0.005 |

**The largest effect is +0.040, and the gap to TF-IDF is 0.19.** No knob is
within a factor of four of closing it, which is the answer the sweep was run to
get.

**Read those deltas against retraining noise.** The sweep's baseline config is
byte-identical to the shipped Skip-gram's, and scores 0.4548 against the
shipped model's 0.4691 — a spread of **0.0143 from retraining alone**, because
Gensim's multi-worker training is not deterministic even at a fixed seed. The
two largest effects (+0.039, +0.040) are under 3x that, from a single
unreplicated run each. **Defaults stay as they are.** Promoting `window=10` or
`epochs=15` on this evidence would be reading noise as signal — the same
mistake §8.2 and §8.3 were run to avoid, and the union numbers they produce
(+0.013 at best) are inside the noise band anyway.

### 8.5 Two capabilities added before deployment

**F-42 — search by trial id.** Measured over 60 real ids drawn five per registry
from all twelve registries, the system returned the requested trial **0 times**:
not at rank 1, not in the top 10, not at all. Every row carries a unique
`Trial ID`, retrieval ran on `abstract`, and nothing indexed the identifier, so
the most basic known-item operation a trial-search tool offers did not work.

An identifier is a key, so it gets a lookup that precedes the ranking, and
trials whose abstracts cite the id follow directly below. **0/60 → 60/60 at
rank 1.** The ground truth is exact — ids are unique — so this is the only
measurement in the project with no annotator provenance to declare.

The cost is stated rather than removed: round 3's `code` stratum, which scores
the *different* question of which trials cite an id, falls from MRR@10 1.000 to
0.667, because its gold lists citing trials only and excludes the queried trial
by construction. Its judgements were **not** rewritten to hide that. See
EVALUATION_AUDIT §10.1.

**F-43 — free-standing negation.** EVALUATION_AUDIT §8 Result 5 established that
no additive feature scheme can express negation, and bigrams were built and
rejected on measurement. The query now carries an operator instead: cue-and-scope
detection in the NegEx style, run on the query to find what is excluded and on
the *document* to separate an abstract that asserts the concept from one that
denies it. Overlap between a query and its negated twin falls **0.55 → 0.33**,
entirely on the two free-standing pairs; the two prefix pairs, which already
worked by morphological substitution, do not move.

It fires on **0 of the 97 main-set queries** and leaves every headline number
unchanged. On the stratum it targets it buys P@1 0.375 → 0.438 and R-precision
0.427 → 0.448 at a cost of Recall@10 0.643 → 0.560, because filtering removes
documents. Both halves of that trade are recorded in EVALUATION_AUDIT §10.2.

**Limits.** Two queries drive the negation measurement, and the cue lexicon was
extended after inspecting failures on them, so its coverage on unseen negations
is unmeasured. The known-item result carries no such caveat: n = 60, exact
ground truth.

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
git clone <repo> && cd clinical-trial-search
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
