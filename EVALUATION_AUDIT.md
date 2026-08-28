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

## 6. BM25 was missing, and adding it exposed the bias again

Sprint 8's conclusion — *"a 40-line TF-IDF baseline beats both embedding
models"* — was measured against the **weak** lexical baseline. TF-IDF with
cosine similarity is the 1970s formulation; **BM25 has been the standard since
TREC-3 (1994)** and is what any IR reviewer expects a new method to be compared
against. It was absent from the project entirely.

Adding it produced a result that looks wrong and is instead diagnostic:

| System | In the round-1 pool? | Recall@10 | Top-10 never judged |
|---|---|---|---|
| TF-IDF | **yes** | 0.615 | **41.2 %** |
| BM25 | **no** | 0.403 | **63.2 %** |
| Skip-gram | yes* | 0.422 | 55.1 % |
| FastText | yes* | 0.410 | 56.2 % |

<sub>*pooled in round 1, but the tokeniser fix has since changed what they retrieve.</sub>

**BM25's 0.403 is not comparable to TF-IDF's 0.615.** Nearly two thirds of what
BM25 returns has never been looked at by a human, and every unjudged document
scores as wrong. The 22-point gap in unjudged rate is exactly the advantage of
having helped define the ground truth.

This is the same defect as §1, in its most direct form: **the eval set now
penalises any new method in proportion to how much it differs from the systems
that built the pool.** It does not merely fail to measure improvement — it
actively opposes it.

BM25 is now a pooled contributor for round 2, which raised the candidate count
from 1,073 to **1,532 across all 97 queries**.

## 7. Round 2: the pool was re-judged, and two conclusions reversed

All **1,532** candidates were judged and merged. The eval set grew from 986 to
**1,691** judgements across the same 97 queries — mean relevant per query 10.2
to **17.4**.

### Provenance: these judgements are model-generated, and calibrated

They were produced by an LLM, not a clinician, and that must travel with any
number computed from them. Before judging, agreement was measured against the
existing human labels on a blind, balanced 60-item sample:

| | |
|---|---|
| Raw agreement | **90.0 %** |
| **Cohen's κ** | **0.800** |
| Precision / recall vs the human | 96.2 % / 83.3 % |

For reference, human-vs-human agreement in TREC-style relevance judging is
typically κ 0.5–0.7. The measured bias was **strictness**: all five misses were
cases where the human treated the query as a topic area and the model treated
it as a specification. The threshold was loosened accordingly before judging.

*This is a defensible substitute for a second annotator, and not a substitute
for a clinician.* Section 3's remaining recommendation — an independent
inter-annotator check — still stands.

### Result 1: BM25 vs TF-IDF reverses entirely

| | Round 1 (biased pool) | Round 2 (re-judged) |
|---|---|---|
| TF-IDF | **0.615** | 0.459 |
| BM25 | 0.403 | **0.471** |
| Gap | TF-IDF +0.212 | BM25 +0.012 |

Nothing about either system changed. The 21-point deficit was **entirely** the
artefact of TF-IDF having helped build the pool while BM25 had not.

And the honest reading of the corrected number: **+0.0116, 95% CI
[−0.0195, +0.0435], p = 0.47 — not significant.** BM25 and TF-IDF are
indistinguishable on this corpus. Neither the original claim nor its reversal
survives; the truthful statement is that the two lexical baselines are
equivalent here.

### Result 2: nDCG says the union's advantage is mostly budget

| Method | Recall@10 | **nDCG@10** | R-precision | docs |
|---|---|---|---|---|
| BM25 | 0.471 | **0.799** | 0.458 | 10.0 |
| TF-IDF | 0.459 | **0.797** | 0.449 | 10.0 |
| union-fasttext | **0.702** | 0.746 | **0.616** | 17.8 |
| union-skipgram | 0.687 | 0.733 | 0.607 | 17.6 |
| fasttext | 0.353 | 0.662 | 0.351 | 10.0 |
| skipgram | 0.351 | 0.655 | 0.346 | 10.0 |

**The metrics disagree, and each is measuring something real.** By nDCG@10 —
the metric that is *not* capped by `|relevant| > k` — the plain lexical
baselines **beat the union** (0.799 vs 0.746). The union wins Recall@10 only
because it returns 1.8× as many documents, and wins R-precision because that
metric is evaluated at depth `|relevant|` ≈ 17, which happens to match its
result-set size.

So the union is not a better ranker. It is a **wider net**, and whether that is
worth it is the product question PRD §8.4 already framed — now with the
ranking cost made explicit rather than hidden by a single metric.

The Recall@10 ceiling is now **0.626** (was 0.879): with 17.4 relevant
documents per query on average, ten slots cannot hold them. BM25 at 0.471 is
**75 % of attainable**.

## What holds up

Most of the methodology is better than typical, and the audit did not disturb it:

- **The relative comparison is sound *between pooled systems*.** Symmetric
  pooling, paired bootstrap over 20k resamples, Bonferroni across three
  metrics. *"The keyword baseline beats the embeddings"* survives — Skip-gram,
  FastText and TF-IDF all contributed to the pool, so none had an unfair
  advantage over the others. §6 is what happens when a system that did *not*
  contribute is scored on the same set: the comparison stops being symmetric,
  and stops being valid.
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
| Tokeniser fix improves retrieval | ⚠️ **unproven** — correct on domain first principles, but this eval set cannot test it |
| BM25 vs TF-IDF (0.403 vs 0.615) | ❌ **not a valid comparison** — BM25 never entered the pool; 63.2 % of its results are unjudged |

## What is required to close it

`scripts/make_eval_round2.py` implements **incremental pooling** — how TREC
admits a new system to an existing collection. Old judgements stay valid; new
candidates get judged and added. It has already emitted the sheet:

- **1,532 unjudged candidates** across **all 97 queries** (1,073 before BM25
  joined the pool)
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
