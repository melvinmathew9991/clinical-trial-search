# Medical Embeddings Search

Semantic search over COVID-19 clinical trials, using word embeddings trained
**in-domain** on the trial corpus itself rather than general web text.

Search `lung failure` and get back trials whose abstracts say *"acute
respiratory distress syndrome"* — records a keyword search would miss entirely.

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
git clone <repo> && cd medical-embeddings-search

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
medsearch doctor --full                  # preflight + artefact integrity
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

## Built for a constrained machine

The reference development machine is an Intel i5-7300HQ: **4 logical cores,
7.89 GB RAM**. That is a design constraint, not an afterthought, and it is why
the pipeline looks the way it does.

| Decision | Effect |
|----------|--------|
| FastText `bucket` bounded to 50,000 | 20 MB n-gram matrix instead of 800 MB (ADR-001) |
| `workers = cores - 1` | One logical core always free; the desktop stays responsive |
| BLAS threads pinned to 1 | numpy and gensim stop fighting over 4 cores (ADR-008) |
| Index stored as `float32` `.npy`, memory-mapped | 4.3 MB and instant load, vs. a 21 MB CSV re-parsed each start (ADR-002) |
| Ranking as a single matmul on pre-normalised rows | Replaces a 10,666-iteration Python loop per query (ADR-003) |
| Vocabulary as a `frozenset` built once | Replaces rebuilding a 30k-element list per document (ADR-004) |
| Corpus streamed to the trainer | Avoids ~600 MB of materialised token lists (ADR-005) |
| Only 4 of 21 CSV columns read | ~90 MB resident instead of ~700 MB |
| Stages run as separate processes | The OS reclaims each stage's peak before the next begins |

`medsearch doctor` refuses to start training below 2 GB free RAM rather than
letting the machine swap.

## Project layout

```
src/medsearch/
├── config.py  runtime.py  logging_conf.py  exceptions.py   # L0 foundation
├── data/            # L1  load, validate, fingerprint
├── preprocessing/   # L2  normalise, tokenize, cache
├── embeddings/      # L3  train, persist, mean-pool
├── search/          # L4  index, rank
├── pipelines/       # L5  orchestration
├── app/             # L5  Streamlit
└── cli.py           # L5  typer
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

The test suite runs in seconds and never touches the network, the full corpus,
or a trained model.

## Status

Sprints 0–7, 9, and 10 are implemented; evaluation (Sprint 8) and final
hardening (Sprint 11) are not. Current state, measured numbers, and the next
action are tracked in [Memory.md](./Memory.md).

Performance figures in `Architecture.md §9` are **estimates** until Sprint 8
replaces them with measurements. They are labelled as such.

## Data

Dimensions COVID-19 publications and clinical trials dataset — 10,666 trials,
21 columns, of which `Title` and `Abstract` carry the signal.
<https://dimensions.figshare.com/articles/dataset/Dimensions_COVID-19_publications_datasets_and_clinical_trials/11961063>

## License

MIT
