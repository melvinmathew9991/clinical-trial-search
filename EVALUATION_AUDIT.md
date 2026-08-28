# Evaluation audit

> Read this before quoting any retrieval number from this project.

A domain review of the evaluation itself — not of the systems measured against
it — carried out before deployment. It found that the headline metric could not
support the claim built on it, and that the tokeniser was destroying the tokens
that identify a clinical intervention.

---

## 1. The eval set cannot measure a change to the pipeline

Relevance was labelled over a **pool** built from the three systems being
scored (Skip-gram, FastText, TF-IDF; top-10 each). That is standard TREC
practice and `scripts/make_eval_candidates.py` implements it carefully. The
consequence was never carried through to the claims.

A document no system retrieved was never shown to the labeller, so it **cannot
be relevant by construction**. Measured with the original tokeniser:

| | |
|---|---|
| Labelled relevant documents | 986 |
| Inside the 3-system pool | **986 (100.0 %)** |
| Outside the pool | **0** |
| Inside `fasttext ∪ tfidf` — what ships | 930 (94.3 %) |

The shipped union measured Recall@10 = **0.955**, against a pool of which its
own two members contributed **94.3 %**. The residual is 56 documents only
Skip-gram returned.

### The proof

Fixing the tokeniser (§3) changed what every system retrieves. Re-measuring:

| Configuration | pool membership | measured Recall@10 |
|---|---|---|
| Old tokeniser | 94.3 % | 0.955 |
| **New tokeniser** | **85.3 %** | **0.862** |

Recall tracks pool membership almost exactly, in both configurations. And
**every** method dropped — including TF-IDF (0.648 → 0.615), which the
embedding changes cannot touch.

**That is the signature of pool-bound labels, not a regression.** The metric
measures *"how much of the frozen pool do you still return"*. It can evaluate
agreement with the pipeline that built it, and nothing else.

> **Therefore: `Recall@10 0.955` was never a measure of retrieval quality, and
> the PRD §8 Definition of Done was declared met on that basis.** The correct
> reading is `recall@pool`. True recall against the corpus is unknown.

## 2. Recall@10 was the wrong headline metric anyway

44 of 97 queries hold **more than 10 relevant documents** (mean 10.2, max 19).
Recall@10 for a query with 19 relevant documents is capped at 0.526.

- **Mean achievable Recall@10: 0.879**, not 1.0
- Worst single-query ceiling: 0.526
- TF-IDF's 0.648 is **74 % of attainable**, not 65 % of perfect

`nDCG@10`, `R-precision` and the ceiling are now computed and reported
alongside every Recall figure. nDCG is not capped by `|relevant| > k` and
rewards every relevant document rather than only the first.

## 3. The tokeniser destroyed biomedical identity

`normalizer.py` stripped every digit, on the stated rationale that *"the
numbers carry no distributional signal for retrieval"*. True of general prose;
false of clinical text, where the digit **is** the identity.

| Term | Became | Consequence |
|---|---|---|
| `CD4` / `CD8` | `cd` / `cd` | two T-cell markers, one token |
| `ACE2` | `ace` | the SARS-CoV-2 entry receptor |
| `SARS-CoV-2` | `sars`, `cov` | strain number lost |
| `BNT162b2` | `bntb` | the Pfizer vaccine |
| `NCT04508933` | `nct` | **every registry ID identical** |
| `interleukin-6` | `interleukin` | IL-6 ≡ IL-1 |
| `type 2 diabetes` | `type`, `diabetes` | type 1 ≡ type 2 |

**6,077 such tokens in a 4,000-abstract sample.** Fixed: digits bound to
letters are preserved, and intra-word hyphens join (`sars-cov-2` → `sarscov2`)
rather than fragmenting. Vocabulary went from **24,897 to 31,189 (+25 %)**.

## 4. Negation was dropped, inconsistently and silently

`no` and `not` are NLTK stopwords and were removed; `without`, `never`,
`none`, `absent`, `negative` survived. So `"no evidence of thrombosis"`
tokenised **identically** to `"evidence of thrombosis"` — a ruled-out finding
and a confirmed one, indistinguishable.

Rules.md's backlog anticipated exactly this (*"Domain stopword allowlist
(F-12) — trigger: negation errors show up in evaluation"*) and
`TextPreprocessor` carried an unwired `keep_words` hook from Sprint 3. Now
wired, with `CLINICAL_KEEP_WORDS` as the default.

## 5. The eval set is blind to §3 and §4

Of 97 queries:

- **0 contain a digit** — so no query exercises `CD4`, `BNT162b2`, or a trial ID
- **1 contains a negation**, and it uses `without`, which survived anyway

Every query is a natural-language clinical phrase — precisely the case the old
tokeniser handled well. **The evaluation could not have detected either defect.**

---

## What holds up

Most of the methodology is better than typical, and the audit did not disturb it:

- **The relative comparison is sound.** Symmetric pooling, paired bootstrap over
  20k resamples, Bonferroni across three metrics. *"The keyword baseline beats
  the embeddings"* survives.
- **The complementarity finding is sound** — an overlap measurement, valid
  within the pool.
- **The power calculation** that resized n=30 → 97 before trusting the result is
  exactly right, and rare.
- **The labelling shows a thinking annotator** — exclusion notes like
  *"Excluded 2021-000988-68 (renal failure)"*.

The problem was never rigour. It was a **relative** methodology used to support
an **absolute** claim.

## Current status of the numbers

| Claim | Status |
|---|---|
| TF-IDF beats the embeddings standalone | ✅ holds |
| The two methods are complementary (66 % unique) | ✅ holds |
| Union Recall@10 = 0.955 | ❌ **withdrawn** — `recall@pool`, and pool-bound |
| PRD §8 DoD (Recall@10 ≥ 0.70) met | ⚠️ **unvalidated** pending re-judgement |
| Tokeniser fix improves retrieval | ⚠️ **unproven** — correct on domain first
  principles, but this eval set cannot test it |

## What is required to close it

`scripts/make_eval_round2.py` implements **incremental pooling** — how TREC
admits a new system to an existing collection. Old judgements stay valid; new
candidates get judged and added. It has already emitted the sheet:

- **1,073 unjudged candidates** across **95 of 97 queries**
- All currently scored as *not relevant*, so every Recall figure in
  `reports/evaluation.json` is a **lower bound**

Remaining work, in order:

1. **Judge the 1,073 candidates** (`reports/eval_round2_candidates.json`),
   merge the positives, re-run `medsearch evaluate`. Only then is the
   tokeniser fix measurable.
2. **Add digit-, code- and negation-bearing queries** to the eval set — the
   cases currently untested.
3. **Inter-annotator agreement** on a 20-query subset, so the judgements carry
   a reliability estimate. Single-annotator ground truth has no error bar.

Until step 1 is done, quote **nDCG@10** and **R-precision** in preference to
Recall, and state the 0.879 ceiling wherever Recall@10 appears.
