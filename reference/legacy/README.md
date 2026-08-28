# Legacy reference — the code this project replaced

**Frozen. Do not edit, do not import, do not run.** Nothing under `src/medsearch`
depends on anything here.

This is the original implementation, kept because a rewrite's decisions only
make sense next to what they replaced. Six files in this repository cite it by
name — `Architecture.md`, `PRD.md`, `Rules.md`, `Memory.md`,
`src/medsearch/exceptions.py`, and `tests/unit/test_regressions.py` — and until
this directory existed, every one of those references pointed at something a
person cloning the repository could not see. `test_regressions.py` in
particular defends against four specific defects, and the reader had no way to
look at them.

## Layout

| Path | Was | Cited by |
|---|---|---|
| `modular-code/src/ML_pipeline/` | `Part_1/Modular+Code/.../src/ML_pipeline/` | the regression tests, Rules §3 and §4 |
| `modular-code/Medical.py` | the single-file version of the same pipeline | — |
| `azure-pipeline/` | `Part_2/Azure_medical_embeddings/` | `deploy/`, Sprint 10 |
| `docs/methodology.pdf` | the original solution write-up | — |
| `docs/azure-solution-document.pdf` | the original cloud write-up | — |
| `docs/design-deck.pptx` | the original design deck | — |

## What to look at, and why

**`modular-code/src/ML_pipeline/top_n.py`** — the function Rules §3 names as
*the* anti-pattern: seven responsibilities in 45 lines. It is why this project
has a function-length gate (`scripts/check_function_length.py`).

**`modular-code/src/ML_pipeline/return_embed.py`** — `get_mean_vector` returns
`np.array([0]*100)` for an all-OOV document. That zero vector then produces a
`0/0` NaN in cosine similarity, which sorts as a silent, plausible-looking
result. Rules §4's "never swallow an error to return a default" is this bug,
and `SearchEngine` guards the zero norm explicitly because of it.

**`modular-code/src/ML_pipeline/train_model.py`** — the `K1`/`K2` variable swap
that wrote Skip-gram vectors into the FastText output files. Nothing detected
it: both files existed, both had plausible contents, and the results were
wrong. This is why every index carries the fingerprint of the model that built
it, and why `ArtefactMismatchError` exists.

**`azure-pipeline/`** — committed live Azure SAS tokens with
`sp=racwdymeop` (read, add, create, **write**, **delete**), hard-coded into
`src/read_data.py` and `src/top_n.py`. This is why a pre-commit hook blocks SAS
tokens and account keys, and why the rewrite authenticates with a managed
identity instead.

> **Redacted before this repository was made public.** The signature
> (`sig=…`), the storage account name, and the subscription GUID in
> `trigger/trigger1.json` are replaced with `REDACTED` markers. Everything
> else is verbatim, including `sp=racwdymeop` and `sv=2020-08-04`, because the
> permission string *is* the lesson: a read-only job was handed delete rights.
>
> The tokens expired on 2021-12-31, but **expiry is not revocation** — a
> signature stays valid if the account key behind it is ever rotated backwards,
> and it identifies a live target either way. `deploy/README.md` §1 has the
> rotation steps, and they still need running.

## What was deleted rather than kept

The generated outputs, on the authority of `scripts/migrate_legacy.py`'s
`SUPERSEDED` list — including `model_Fasttext.bin.wv.vectors_ngrams.npy` at
**800 MB**, which is exactly what ADR-001's bounded bucket exists to avoid. The
replacement FastText model is 29.3 MB. Also removed: duplicate copies of the
corpus (it existed three times) and of the exploration notebook (three times,
byte-identical). See `Memory.md` for the full accounting.
