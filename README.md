# Medical Embeddings Search

Semantic search over COVID-19 clinical trials, using word embeddings trained
**in-domain** on the trial corpus itself rather than general web text.

Search `lung failure` and get back trials whose abstracts say *"acute
respiratory distress syndrome"* — records a keyword search would miss entirely.

> **Measured caveat:** that paraphrase capability is real and demonstrable, but
> plain keyword baselines still **outperform** both embedding models on their
> own (TF-IDF 0.459 and BM25 0.471 against FastText 0.353 Recall@10, over 97
> queries). What ships is therefore neither one alone but the **union** of
> embeddings and TF-IDF — the two methods miss different trials, and returning
> everything either finds lifts Recall@10 to **0.702** from a 20-document
> budget. By nDCG@10 the plain baselines still rank better (0.799 against
> 0.746): the union is a wider net, not a better ranker. See
> [Retrieval quality](#retrieval-quality--the-keyword-baseline-wins).

---

## Why in-domain embeddings

General-purpose embeddings are trained on news and web text, where `ARDS`,
`seroconversion`, and `comorbidity` are rare or absent. Training Word2Vec
(Skip-gram) and FastText directly on 10,666 clinical-trial abstracts produces
a vector space where clinical terms cluster the way clinicians use them.

FastText additionally learns character n-grams, so it can produce a vector for
a morphological variant it never saw — useful in a domain full of them.

## Quickstart

```bash
git clone <repo> && cd clinical-trial-search

make setup      # venv + install + NLTK corpora
make doctor     # preflight: cores, free RAM, disk, artefact budget
make data       # migrate the corpus into data/raw/ (dry run, then --apply)
make train      # train both models, build both indexes
make app        # Streamlit UI on http://localhost:8501
```

No `make` on Windows? Each recipe is a plain command — open the `Makefile` and
run the line you need, or:

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev,app]"
python -m medsearch.runtime --download-nltk
medsearch doctor
medsearch train --model all
medsearch index build --model all
streamlit run src/medsearch/app/streamlit_app.py
```

### Low-memory profile

On a machine with less than ~3 GB free, use the sampled profile. It completes
in about 90 seconds and stays under 400 MB:

```bash
make train-dev              # equivalent to: make train LIMIT=2000
```

Artefacts built this way are stamped `sampled: true` in their metadata so they
can never be mistaken for a production run.

## CLI

```bash
medsearch doctor --full                  # preflight + artefact integrity checks
medsearch evaluate                       # Recall/nDCG/MRR vs TF-IDF and BM25
medsearch preprocess --field abstract    # clean and cache tokens
medsearch train --model fasttext         # train one model
medsearch index build --model all        # embed the corpus, write the index
medsearch search "lung failure" -k 5     # search
medsearch search "kidney injury" --json  # machine-readable output
medsearch index info --model skipgram    # inspect an index manifest
```

## Python API

```python
from medsearch.config import get_settings
from medsearch.pipelines import load_search_engine

engine = load_search_engine(get_settings(), "skipgram", "abstract")
response = engine.search("acute respiratory failure", top_n=5)

for result in response.results:
    print(f"{result.score:.3f}  {result.trial_id}  {result.title}")
```

## Documentation

Five files define this project. Read them in order before contributing —
they are also what an AI coding assistant should load first.

| File | Purpose |
|------|---------|
| [PRD.md](./PRD.md) | What we are building, for whom, and the requirement ids |
| [Architecture.md](./Architecture.md) | Layering, folder structure, tech stack, ADRs, resource budget |
| [Rules.md](./Rules.md) | Engineering standards, banned libraries, error handling, AI boundaries |
| [Phases.md](./Phases.md) | Sprint plan and definitions of done |
| [Memory.md](./Memory.md) | Running progress log — **read this first in a new session** |

> ⚠️ **These numbers were superseded by a re-judged evaluation set.** The
> original pool was built from the systems being scored, so `Recall@10 0.955`
> measured pool membership, not retrieval quality. All 1,532 outstanding
> candidates have since been judged (986 → 1,691 judgements). On the corrected
> set: **BM25 0.471, TF-IDF 0.459 — statistically indistinguishable (p = 0.47)**,
> and by nDCG@10 the lexical baselines *beat* the union (0.799 vs 0.746). The
> judgements are model-generated, calibrated at **Cohen's κ = 0.800** against
> the original human labels. Full analysis:
> **[EVALUATION_AUDIT.md](./EVALUATION_AUDIT.md)**.

## Retrieval quality — the keyword baseline wins

Measured over **97 labelled queries / 1,691 relevance judgements** — the
round-2, re-judged set described in the banner above:

| Method | Docs shown | Recall@10 | nDCG@10 | R-prec | MRR@10 | Precision@1 |
|--------|-----------|-----------|---------|--------|--------|-------------|
| **Union: FastText + TF-IDF** *(ships)* | 17.8 | **0.702** | 0.746 | **0.616** | 0.890 | 0.830 |
| Union: Skip-gram + TF-IDF | 17.6 | 0.687 | 0.733 | 0.607 | 0.892 | 0.804 |
| BM25 baseline | 10.0 | 0.471 | **0.799** | 0.458 | 0.909 | 0.856 |
| TF-IDF baseline | 10.0 | 0.459 | 0.797 | 0.449 | **0.952** | **0.918** |
| FastText | 10.0 | 0.353 | 0.662 | 0.351 | 0.818 | 0.732 |
| Skip-gram | 10.0 | 0.351 | 0.655 | 0.346 | 0.818 | 0.742 |

**Read Recall against the ceiling.** With 17.4 relevant documents per query,
the attainable Recall@10 is **0.626** at depth 10 and 0.951 at depth 20 — ten
slots cannot hold seventeen documents. BM25's 0.471 is 75 % of what is
reachable at its budget.

The union is scored to depth 20 because that is the budget it occupies; the
single rankers are scored to depth 10. "Docs shown" is what makes the rows
comparable — the union buys its recall with roughly twice the result list, and
**pays for it in ranking quality**: TF-IDF alone still has the best
Precision@1 (0.918 vs 0.830) and MRR@10 (0.952 vs 0.890), and both lexical
baselines beat the union on nDCG@10 (0.799 / 0.797 vs 0.746), the one metric
here that is not capped by the ceiling. Recall is the metric that matters for a
researcher who must not miss a relevant trial, so the union ships; a user who
wants the tightest, best-ranked list can turn it off.

**BM25 and TF-IDF are equivalent on this corpus** (Δ +0.0116, 95 % CI
[−0.0195, +0.0435], p = 0.47). An earlier 21-point gap in TF-IDF's favour was
an artefact of TF-IDF having helped build the judgement pool while BM25 had
not; it did not survive re-judging.

Among the depth-10 rankers, the lexical baselines win outright:

**The project's central premise does not hold as built.** A 40-line TF-IDF
baseline beats both in-domain embedding models on every metric. All gaps are
statistically significant and survive Bonferroni correction across the three
metrics (Recall@10: Δ −0.163, 95% CI [−0.249, −0.078], p = 0.0003, paired over
97 queries with 20k bootstrap resamples). TF-IDF wins 56 queries, the
embeddings 28, 13 ties. *Those intervals were computed on the round-1
judgements and have not been recomputed since; the direction survives
re-judging — 0.459 against 0.353 — and the gap widens rather than closes.*

Skip-gram and FastText are **statistically indistinguishable** from each other
(p = 0.28 / 0.86 / 1.00), so the character n-grams buy nothing measurable here.

An earlier evaluation at n=30 could not resolve this — the 95% CI included
zero. The set was resized by a power calculation (sd 0.308, target effect
0.088 → n = 97) before the comparison was trusted.

**The paraphrase advantage is real but narrow.** `mechanical ventilation
weaning` still retrieves ventilation-liberation trials although no abstract
contains the word "weaning". It just does not compensate: most realistic
clinical queries carry a rare decisive term — `colchicine`, `remdesivir`,
`ECMO` — that IDF weights heavily and mean pooling averages away.

**SIF weighting was tried and does not work.** The obvious fix — weight each
word vector by `a / (a + p(w))` before averaging — was implemented and measured
on the same 97 queries. Skip-gram 0.469 → 0.450 (ns); FastText 0.485 → 0.432
(**significantly worse**). An ablation shows the frequency weighting is what
costs recall, not the common-component removal, which does nothing measurable.

**Why:** reweighting cannot recover lexical precision from a dense average.
TF-IDF matches exactly in 40,012 sparse dimensions where `colchicine` is its own
coordinate; upweighting it inside a 100-dimensional average still blurs it
against every other word. Rare words also have the least reliable vectors, so
SIF puts the most weight on the least trustworthy directions.

**The hybrid was measured before being built, and is not worth building.**
Reciprocal Rank Fusion over both rankings scores 0.648 against TF-IDF's 0.648 —
a difference of +0.0005 (p = 0.98). Fusing deeper lists is significantly
*worse*.

**But the complementarity is real.** 326 of the 496 relevant documents the
embeddings retrieve — **66%** — are ones TF-IDF never returns. That is an
overlap measurement and re-judging does not touch it. The recall figures beside
it in round 1 (0.955 for the union against depth-matched TF-IDF's 0.715) were
pool-bound and are withdrawn; on the re-judged set the union reaches **0.702**
at depth 20 against TF-IDF's 0.459 at depth 10. At a fixed budget of ten
results a fusion must drop a reliable TF-IDF hit to admit an embedding hit, so
the gain is real and unusable at once.

**The lever is the result budget, not the model.** Returning the 20-document
union reaches **0.702**, just clearing the 0.70 target, with no new modelling —
a product decision about how many trials to show. What the re-judged set adds
is the cost: by nDCG@10 that same union ranks *below* a 40-line lexical
baseline (0.746 against 0.799). The budget buys coverage, not ranking quality.
See [PRD §8.3](./PRD.md).

**Tuning does not rescue the embeddings either.** A one-factor-at-a-time sweep
over `vector_size`, `window`, `min_count`, and `epochs` on the full corpus moves
Recall@10 by at most **+0.040** — against a 0.19 gap to TF-IDF. Retraining the
*identical* config shifts the score by 0.014 on its own (Gensim's multi-worker
training is not seed-deterministic), so the two largest effects are under 3× the
noise and the defaults stay put. See `reports/sweep.json`.

**FastText is the default model** because under the union it beats Skip-gram on
recall (0.955 vs 0.927, p = 0.019, round-1 judgements) and is no worse on
ranking (p = 0.16) — a difference that does not exist between them as
standalone rankers. On the re-judged set the gap is 0.702 against 0.687: same
direction, not re-tested for significance, and nothing in it argues for
switching the default.

Reproduce with `medsearch evaluate`; it exits non-zero while the target is
missed, so it can gate a release. `python scripts/sweep.py` and
`python scripts/significance.py <a> <b>` regenerate the sweep and the paired
bootstrap tests behind every p-value quoted here.

## Benchmarks

Measured on the reference machine — Intel i5-7300HQ, **4 logical cores, 7.89 GB
RAM** — over the full 10,666-document corpus.

| Stage | Wall time | Peak RSS | Artefact |
|-------|-----------|----------|----------|
| Preprocess (cold) | 46.3 s | 313 MB | 22 MB token cache |
| Train Skip-gram | 30.0 s | 319 MB | 10.2 MB |
| Train FastText | 56.0 s | 346 MB | 29.3 MB |
| Build index (each) | ~7 s | 152 MB | 4.1 MB |
| **Full `make train`** | **2 min 22 s** | **~350 MB** | 59 MB total |

Vocabulary 24,897 words. Out-of-vocabulary documents: **2 of 10,666 (0.02%)**.

**Query latency** over 120 queries after warm-up: p50 **1.4 ms**, p95 **3.3 ms**,
p99 10.7 ms. The PRD target is p95 < 300 ms. Cold engine load 2.96 s; serving
RSS **438 MB** for the shipped default (fasttext + union) against a 1.2 GB
budget; 342 MB was skipgram alone, which stopped being the default in Sprint 8.
Union retrieval queries both engines, so it costs p95 **128 ms** — 35× the
embedding-only path and ~43 MB more resident, still comfortably inside both
targets, and the one budget the default meaningfully spends.

### Against the predecessor project

| Artefact | Legacy | This | |
|----------|--------|------|---|
| FastText model | 762.9 MB | **29.3 MB** | 26× smaller |
| Skip-gram doc vectors | 20.7 MB | **4.2 MB** | 5× smaller |
| FastText doc vectors | 20.6 MB | **4.2 MB** | 5× smaller |
| **Total** | **804.3 MB** | **37.6 MB** | **21× smaller** |

## Built for a constrained machine

The reference machine is a design constraint, not an afterthought, and it is
why the pipeline looks the way it does.

| Decision | Effect |
|----------|--------|
| FastText `bucket` bounded to 50,000 | 20 MB n-gram matrix instead of 800 MB (ADR-001) |
| `workers = cores - 1` | One logical core always free; the desktop stays responsive |
| BLAS threads pinned to 1 | numpy and gensim stop fighting over 4 cores (ADR-008) |
| Index stored as `float32` `.npy`, memory-mapped | 4.1 MB and instant load, vs. a 21 MB CSV re-parsed each start (ADR-002) |
| Ranking as a single matmul on pre-normalised rows | Replaces a 10,666-iteration Python loop per query (ADR-003) |
| Vocabulary as a `frozenset` built once | Replaces rebuilding a 30k-element list per document (ADR-004) |
| Corpus streamed to the trainer | Avoids ~600 MB of materialised token lists (ADR-005) |
| Only 4 of 21 CSV columns read | ~90 MB resident instead of ~700 MB |
| Stages run as separate processes | The OS reclaims each stage's peak before the next begins |

`medsearch doctor` refuses to start training below 2 GB free RAM rather than
letting the machine swap. The floor scales down for a `--limit` run.

## Artefact integrity

Every artefact records a fingerprint of what produced it, and three mismatch
classes are guarded — each found the hard way:

| Mismatch | Consequence if unguarded | Guard |
|----------|--------------------------|-------|
| Index built by a different model | Wrong vectors, right-looking output | `DocumentIndex.load(expected_fingerprint=...)` |
| Sampled index vs. full corpus | Most documents silently unsearchable | `DocumentIndex.is_sampled`, UI banner |
| Corpus replaced after indexing | **Results resolve to the wrong documents** | `StaleIndexError` at load |

The third is the dangerous one: row ids are positional, so a stale index still
resolves and nothing looks broken. `medsearch doctor --full` checks all three.

## Project layout

```
src/medsearch/
├── exceptions.py  config.py  _typing.py      # L0  errors, settings, type aliases
├── runtime.py                                # L1  thread pinning, memory probes
├── logging_conf.py                           # L2  structured logs, stage timing
├── data/                                     # L3  load, validate, fingerprint
├── preprocessing/                            # L3  normalise, tokenize, cache
├── embeddings/                               # L4  train, persist, mean-pool
├── search/                                   # L5  index, rank, TF-IDF baseline
├── pipelines/                                # L6  train, evaluate, integrity
├── app/                                      # L7  Streamlit
└── cli.py                                    # L7  typer
```

Layers depend strictly downward, enforced mechanically by `import-linter` in
CI. See [Architecture.md](./Architecture.md#2-layered-design).

## Development

```bash
make check     # ruff + mypy --strict + import-linter + pytest  (what CI runs)
make fmt       # auto-format
make test      # fast suite only
make test-all  # include slow / full-corpus tests
```

`make test` is the fast loop — unit tests only, ~10 s, no network and no
training. `make test-all` adds the integration suite, which trains real models
over a 20-row fixture (~40 s) and is where the 80% coverage gate is enforced.

## Status

The pipeline runs end to end on the full corpus and every resource target is
met with wide margin. Numbers above are measured, not estimated.

**Retrieval quality is measured — twice.** 97 queries and 1,691 relevance
judgements, after a second judging round that added the candidates the first
pool had missed. Two conclusions from round 1 did not survive it: the
TF-IDF-over-BM25 win was pool bias, and the union's recall margin shrank from
0.955 to 0.702. The full account is in
[EVALUATION_AUDIT.md](./EVALUATION_AUDIT.md).

**Provenance of the judgements:** 986 are human, **705 are model-generated**,
calibrated at Cohen's κ = 0.800 against the human labels. That is a defensible
stand-in for a second annotator and *not* for a clinician — no independent
domain review has been done.

**That blind spot is now closed.** A third round added 22 queries in three
strata — alphanumeric entity identity, known-item registry codes, and negation
pairs — in `tests/fixtures/eval_queries_round3.json`, scored separately by
`python scripts/round3_evaluate.py`. The headline results: the tokeniser fix is
**confirmed with a large effect** (registry-code Recall@10 0.44 → 1.00; `CD4`
and `CD8` returned an identical top-10 before it), while the negation allowlist
is **doing much less than intended** — the gain comes from hyphen-joining, and
free-standing `not` is retained but carries less weight than the words it
inverts. Full analysis in [EVALUATION_AUDIT.md](./EVALUATION_AUDIT.md) §8.

**The `code` stratum needs no relevance judgements at all** — a document is
relevant to `NCT04446429` iff its text contains that string — which makes it
the only evaluation data here with no judgement provenance to caveat.

**The container is built and serving** as of 2026-08-28. Both targets run under
`docker run --memory=2g`: the mounted `runtime` image and the `standalone` one
that bakes artefacts for the Azure free tier. Warm union query p95 inside the
container is **103.5 ms**, peak RSS **662 MB**. Building it surfaced a defect no
test could reach — the shipped union path died on `PermissionError` for the
token-cache directory while the healthcheck stayed green. Fixed and re-verified.
The image is **941 MB** against a < 800 MB target, which Sprint 9's DoD now
fails on; the breakdown is in [Phases.md](./Phases.md).

**Still unverified:** the Azure pipeline has never been deployed, and no search
has ever been driven through the UI in a browser — the app serves, the engine
path is tested, and the same union code has now been exercised inside the
container, but the browser interaction itself is not.

**Gates, re-run 2026-08-28:** ruff, ruff-format, `mypy --strict` (32 files),
import-linter (2 contracts) all clean; **561 tests pass**, coverage **87.6 %**
against an 80 % gate.

Current state and the next action are tracked in [Memory.md](./Memory.md).

## Data

Dimensions COVID-19 publications and clinical trials dataset — 10,666 trials,
21 columns, of which `Title` and `Abstract` carry the signal.
<https://dimensions.figshare.com/articles/dataset/Dimensions_COVID-19_publications_datasets_and_clinical_trials/11961063>

## License

MIT
