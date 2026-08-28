# Medical Embeddings Search

Semantic search over COVID-19 clinical trials, using word embeddings trained
**in-domain** on the trial corpus itself rather than general web text.

Search `lung failure` and get back trials whose abstracts say *"acute
respiratory distress syndrome"* — records a keyword search would miss entirely.

> **Measured caveat:** that paraphrase capability is real and demonstrable, but
> a plain TF-IDF baseline still **outperforms** both embedding models on their
> own (Recall@10 0.648 vs 0.485, p = 0.0003 over 97 queries). What ships is
> therefore neither one alone but the **union** of both — the two methods miss
> different trials, and returning everything either finds reaches Recall@10
> **0.955**. See [Retrieval quality](#retrieval-quality--the-keyword-baseline-wins).

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
medsearch evaluate                       # Recall@k, MRR vs the TF-IDF baseline
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

## Retrieval quality — the keyword baseline wins

Measured over **97 labelled queries / 986 relevance judgements**:

| Method | Docs shown | Recall@10 | MRR@10 | Precision@1 |
|--------|-----------|-----------|--------|-------------|
| **Union: FastText + TF-IDF** *(ships)* | 17.7 | **0.955** | 0.852 | 0.753 |
| Union: Skip-gram + TF-IDF | 17.5 | 0.927 | 0.822 | 0.737 |
| TF-IDF baseline | 10.0 | 0.648 | **0.888** | **0.835** |
| FastText | 10.0 | 0.485 | 0.761 | 0.670 |
| Skip-gram | 10.0 | 0.469 | 0.757 | 0.670 |

The union is scored to depth 20 because that is the budget it occupies; the
single rankers are scored to depth 10. "Docs shown" is what makes the rows
comparable — the union buys its recall with roughly twice the result list, and
**pays for it at the top**: TF-IDF alone still has the best Precision@1 (0.835
vs 0.753) and MRR@10 (0.888 vs 0.852). Recall is the metric that matters for a
researcher who must not miss a relevant trial, so the union ships; a user who
wants the tightest possible list can turn it off.

Among the single rankers, TF-IDF wins outright:

**The project's central premise does not hold as built.** A 40-line TF-IDF
baseline beats both in-domain embedding models on every metric. All gaps are
statistically significant and survive Bonferroni correction across the three
metrics (Recall@10: Δ −0.163, 95% CI [−0.249, −0.078], p = 0.0003, paired over
97 queries with 20k bootstrap resamples). TF-IDF wins 56 queries, the
embeddings 28, 13 ties.

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
embeddings retrieve — **66%** — are ones TF-IDF never returns, and the union of
both top-10 lists reaches **0.955** recall against depth-matched TF-IDF's 0.715
(+0.240, p < 0.0001). At a fixed budget of ten results a fusion must drop a
reliable TF-IDF hit to admit an embedding hit, so the gain is real and
unusable at once.

**The lever is the result budget, not the model.** Returning the 20-document
union clears the 0.70 target at 0.955, with no new modelling — a product
decision about how many trials to show. See [PRD §8.3](./PRD.md).

**Tuning does not rescue the embeddings either.** A one-factor-at-a-time sweep
over `vector_size`, `window`, `min_count`, and `epochs` on the full corpus moves
Recall@10 by at most **+0.040** — against a 0.19 gap to TF-IDF. Retraining the
*identical* config shifts the score by 0.014 on its own (Gensim's multi-worker
training is not seed-deterministic), so the two largest effects are under 3× the
noise and the defaults stay put. See `reports/sweep.json`.

**FastText is the default model** because under the union it beats Skip-gram on
recall (0.955 vs 0.927, p = 0.019) and is no worse on ranking (p = 0.16) — a
difference that does not exist between them as standalone rankers.

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

**Not yet done:** retrieval quality has not been measured. Sprint 8's harness
is built — metrics, TF-IDF baseline, and a pooled candidate generator — but it
needs a human-labelled evaluation set, which is deliberately not machine
generated (see [Architecture.md §3.1](./Architecture.md)). Run
`python scripts/make_eval_candidates.py`, label the sheet, then
`medsearch evaluate`.

**Also unverified:** the container image has never been built (no Docker on the
development machine) and the Azure pipeline has never been deployed.

Current state and the next action are tracked in [Memory.md](./Memory.md).

## Data

Dimensions COVID-19 publications and clinical trials dataset — 10,666 trials,
21 columns, of which `Title` and `Abstract` carry the signal.
<https://dimensions.figshare.com/articles/dataset/Dimensions_COVID-19_publications_datasets_and_clinical_trials/11961063>

## License

MIT
