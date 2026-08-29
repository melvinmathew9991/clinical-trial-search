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

> **Refined 2026-08-29.** "Not a better ranker" holds; "worse ranker" does not
> follow from this table. The union is scored to depth 20, so its nDCG ideal
> sums ~17 slots against the baselines' 10 — the comparison is not
> like-for-like. Truncated to an equal ten-document budget, and with the §9 fix
> applied, the union scores **nDCG@10 0.789** against BM25's 0.799. Level. The
> post-fix full-budget figures are nDCG@10 **0.762** and R-precision **0.639**
> (was 0.746 / 0.616).

The Recall@10 ceiling is now **0.626** (was 0.879): with 17.4 relevant
documents per query on average, ten slots cannot hold them. BM25 at 0.471 is
**75 % of attainable**.

## 8. Round 3: the strata the eval set was blind to

Sections 3 and 4 identified two real preprocessing defects. Section 5 then
showed the eval set could not see either of them, and re-judging the pool in
round 2 did not change that — a pool holds no candidates for a query nobody
asked. So 22 new queries were written in three strata and scored separately.

| Stratum | Queries | Relevant/query | Ground truth |
|---|---|---|---|
| `entity` | 11 | 12.8 | model-judged, round-2 protocol |
| `code` | 3 | 2.0 | **exact string containment — no judgement at all** |
| `negation` | 8 | 12.5 | model-judged, round-2 protocol |

The `entity` queries are built as **collapse pairs**: `IL-6`/`IL-1`,
`CD4`/`CD8`, `SARS-CoV-2`/`MERS-CoV`. Under the pre-audit chain each pair
becomes one token, so a system that cannot tell the members apart is visible
directly. The `negation` queries are **four positive/negated twins**, where the
measurement is the overlap between the pair rather than either query alone —
which needs no relevance judgements either.

*One query was written and dropped.* `MERS-CoV and SARS-CoV-1 comparison` has
no relevant document in a corpus of COVID-19 trials; the other coronaviruses
appear only as background prose. That is a finding about corpus scope, not a
labelling gap, and it is why the query survives only as a discrimination probe.

### Result 1: the tokeniser fix is confirmed, and the effect is large

`scripts/round3_ablation.py` runs both chains over the same corpus with the
same lemmatiser, rankers and queries, so the only difference is tokenisation.

**Known-item retrieval by registry code**, Recall@10 against exact-match truth:

| | old chain | new chain |
|---|---|---|
| TF-IDF | 0.44 | **1.00** |
| BM25 | 0.33 | **1.00** |

**Discrimination between entities the old chain collapsed**, overlap@10 —
lower is better, 1.00 means the two queries returned the identical list:

| Pair | tfidf old | tfidf new | bm25 old | bm25 new |
|---|---|---|---|---|
| CD4 vs CD8 | **1.00** | 0.50 | **1.00** | 0.40 |
| SARS-CoV-2 vs MERS-CoV | 0.70 | **0.00** | 0.00 | 0.00 |
| IL-6 vs IL-1 | 0.40 | 0.10 | 0.20 | 0.20 |

`CD4 T cell response` and `CD8 T cell response` returned **the same ten
documents** before the fix. Vocabulary grew 39,879 → 55,487. Section 3's claim
was correct on domain first principles and is now correct on measurement.

### Result 2: the negation keep-list is doing almost nothing

This is the uncomfortable half, and it does not match what PRD F-12 assumed.
Overlap@10 between each query and its negated twin:

| Negated query | tfidf old | tfidf new | bm25 old | bm25 new |
|---|---|---|---|---|
| non-hospitalized patients | 0.70 | **0.10** | 0.40 | **0.00** |
| non-severe covid-19 pneumonia | 0.70 | **0.40** | 0.10 | 0.00 |
| patients **not** requiring oxygen | 1.00 | 0.90 | 1.00 | 0.80 |
| treatment **without** ventilation | 0.80 | 0.80 | 0.30 | 0.30 |

The two that moved are prefix negations; the two that did not are free-standing
ones. `CLINICAL_KEEP_WORDS` is not what fixed the first two — intra-word hyphen
joining is. `non-hospitalized` becomes the single token `nonhospitalized`, and
the reason that works is IDF:

| token | df | idf |
|---|---|---|
| `nonhospitalized` | 45 | **6.45** |
| `hospitalized` | 1,259 | 3.14 |
| `nonsevere` | 37 | **6.64** |
| `severe` | 3,022 | 2.26 |
| `without` | 1,226 | 3.16 |
| `not` | 3,274 | **2.18** |

A prefix negation mints a rare token that dominates the query vector. A
free-standing one adds a token that appears in a third of the corpus and is
weighted *below* the content words it is supposed to invert. Retaining `not`
was necessary and is not sufficient: `treatment without mechanical ventilation`
scores identically under both chains, because `without` was never a stopword —
it survived the old chain too.

**So the negation fix is half-confirmed and half-refuted.** The
inconsistency section 4 describes was real, and removing it was right. But the
retrieval benefit attributed to it comes from the normaliser, not the
allowlist, and no term-weighting scheme will fix free-standing negation:
that needs the query parser to treat it as an operator, not a term.

### Result 3: on the queries the embeddings should lose, they lose by more

Scored with the shipped evaluator (`scripts/round3_evaluate.py`):

| `entity` stratum | P@1 | MRR@10 | nDCG@10 | R@10 |
|---|---|---|---|---|
| BM25 | **0.909** | **0.955** | **0.765** | 0.551 |
| TF-IDF | **0.909** | **0.955** | 0.742 | 0.545 |
| union-fasttext | 0.727 | 0.788 | 0.725 | **0.791** |
| fasttext | 0.636 | 0.718 | 0.510 | 0.370 |
| skipgram | 0.636 | 0.718 | 0.505 | 0.354 |

The gap is wider here than on the general set, and the reason is mechanical:
an entity query turns on one rare alphanumeric token, which is exactly what a
100-dimensional mean-pooled average destroys and what IDF weights heavily.

### Result 4: the union actively damages known-item retrieval

| `code` stratum | P@1 | MRR@10 | R@10 |
|---|---|---|---|
| BM25 / TF-IDF | **1.000** | **1.000** | 1.000 |
| union-fasttext | 0.500 | 0.500 | 1.000 |
| fasttext | 0.000 | 0.000 | 0.000 |

The lexical half answers these perfectly. The union keeps its recall and
**halves its precision at rank 1**, because the embedding half contributes a
document that cannot be relevant — FastText scores 0.000 on every metric here.
This is a concrete cost of shipping the union by default, on a query type users
of a trial-search tool will certainly issue, and it belongs in the PRD §8.4
decision.

> **Corrected 2026-08-29 — this was a defect in the fusion, not a property of
> the union.** The two runs share no documents on these queries, so both award
> the identical RRF score `1/(60+rank)` at every rank, and `sorted` broke the
> tie by dict insertion order — which is the embedding run, because its loop
> runs first. Every code query placed an irrelevant embedding document at rank
> 1 and the relevant keyword document at rank 2, at byte-identical scores.
> Weighting the keyword run and breaking the remainder explicitly takes this
> stratum to **MRR@10 1.000, nDCG@10 1.000, R-precision 1.000** — level with
> the lexical baselines. See §9. The measurements above are left as recorded.

### A capability gap this exposed

Only **71 of 10,666 abstracts** contain any registry code, because the ids live
in the `Trial ID` column and retrieval runs on `abstract`. The tokeniser
docstring's flagship example — `NCT04508933` collapsing to `nct` — is a true
statement about tokenisation and almost irrelevant to this system's retrieval.
If searching by trial id matters to users, the fix is to index the id column,
not to tune the tokeniser.

### Result 5: bigrams were the obvious fix, and they do not work

Result 2 suggested its own remedy. If prefix negation works because it mints a
rare token, give free-standing negation the same thing: `not requiring` has
idf 6.00 and `without mechanical` 7.97, squarely in the band where
`nonhospitalized` (6.45) already succeeds. Adding bigram features to both
lexical rankers is a small, cheap change, so it was measured before being
recommended any further (`scripts/round3_bigram_experiment.py`).

**It fails, and it costs.** Negation pair overlap barely moves --
`not requiring` 0.90 -> 0.80 for TF-IDF, `without mechanical` unchanged at
0.80 -- while retrieval quality falls everywhere:

| nDCG@10 | unigram | bigram |
|---|---|---|
| negation stratum, TF-IDF | **0.528** | 0.377 |
| entity stratum, TF-IDF | **0.742** | 0.663 |
| main 97-query set, TF-IDF | **0.797** | 0.720 |
| main 97-query set, BM25 | **0.799** | 0.608 |

Vocabulary grows 55,487 -> 887,024 (16x) and p95 latency roughly triples. One
prefix pair actually got *worse* (`non-hospitalized`, 0.10 -> 0.30).

**Why it fails is the interesting part, and it corrects Result 2's framing.**
Rare-token IDF was never the whole mechanism. Decompose a pair into features
shared with its positive twin versus features unique to the negated query, and
weight each by idf:

| Negated query | shared idf mass | unique idf mass |
|---|---|---|
| `patients not requiring supplemental oxygen` (bigrams) | **25.3** | 12.7 |
| `non-hospitalized patients with covid-19` | 5.6 | **13.8** |

Bigrams do hand the negated query a discriminating feature -- `not requiring`
at 6.00 -- but they hand *both* queries `requiring supplemental` (6.5) and
`supplemental oxygen` (5.2) at the same time. Shared evidence grows faster than
unique evidence, so the pair ends up *more* similar, not less.

Prefix negation escapes this because morphology performs a **substitution**:
`hospitalized` is replaced by `nonhospitalized`, so the shared term is
*removed* from the query, not merely joined by a new one. That is the general
statement, and it is stronger than "rare tokens win":

> **Negation requires removing or inverting shared evidence. Every additive
> feature scheme -- a stopword allowlist, bigrams, trigrams, term reweighting
> -- can only add evidence, and therefore none of them can express it.**

Which means the recommendation in Result 2 stands and is not negotiable by
feature engineering: free-standing negation needs the *query* to carry an
operator that subtracts, not a representation that carries one more term.

### Power

n = 11, 3 and 8. These strata are **diagnostic, not powered**: the main set was
sized to 97 by a power calculation precisely because small differences need it.
No p-value is quoted here and none should be. The effects reported above are
the ones large enough to read at this n — 1.00 versus 0.00, 0.909 versus 0.636
— and the pair-overlap and code measurements need no inference at all, since
their ground truth is exact.


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
## 9. The fusion had a defect, and fixing it settled the product decision

*2026-08-29.*

Section 8 Result 4 read a symptom as a property. `UnionRetriever._rank` fused
the two rankings with unweighted RRF and ordered by score alone. When the runs
are **disjoint** — which is exactly what happens on a known-item query, where
the lexical run holds the answer and the embedding run holds nothing — every
document scores `1/(RRF_K + rank)`, so the two runs' rank-1 documents tie at
byte-identical `0.016393`. Python's `sorted` is stable, so the tie fell to dict
insertion order, and the embedding loop populates the dict first.

Observed directly, on all three `code` queries:

```
1. NCT04397562   score=0.016393   by=embedding
2. NCT04853927   score=0.016393   by=keyword     RELEVANT
```

### The fix

Two changes, both in `search/hybrid.py`:

* **Weight the keyword run** (`KEYWORD_WEIGHT = 1.5`). The lexical ranker wins
  Precision@1 in every stratum measured — entity 0.909 vs 0.636, code 1.000 vs
  0.000, negation 0.625 vs 0.250 — so an unweighted fusion gives the embedding
  run more say at the top than it earned. A sweep over 1.0 / 1.5 / 2.0 / 3.0
  saturates at 1.5; 2.0 is identical and 3.0 slightly worse.
* **Break remaining ties explicitly**: consensus, then keyword-only, then
  embedding-only, so the order never depends on insertion again.

### Effect, and what it does not touch

**Recall is unchanged everywhere**, on every stratum and the main set —
reordering cannot change a set. That is the control: any metric that moved,
moved because of ranking alone.

| union-fasttext | MRR@10 | nDCG@10 | R-precision | R@10 |
|---|---|---|---|---|
| main 97-query | 0.890 → **0.923** | 0.746 → **0.762** | 0.616 → **0.639** | 0.702 → 0.702 |
| `entity` | 0.788 → **0.864** | 0.725 → **0.751** | 0.590 → **0.604** | 0.791 → 0.791 |
| `code` | 0.500 → **1.000** | 0.649 → **1.000** | 0.722 → **1.000** | 1.000 → 1.000 |
| `negation` | 0.513 → **0.532** | 0.519 → **0.529** | 0.372 → **0.427** | 0.643 → 0.643 |

### The depth mismatch, which is the other half of the story

`_update` scores the union to `k * depth_factor` with `depth_factor = 2`. Two
consequences were being misread:

1. **The union rows' "P@1" is P@2.** That is why an arithmetically impossible
   0.500 appeared over n = 3 queries, and why it stays at 0.833 after the fix
   even though the relevant document is now at rank 1 — the second slot holds
   an embedding document that cannot be relevant.
2. **nDCG@10 is not comparable across depth factors.** The union's ideal DCG
   sums `min(|relevant|, 20)` ≈ 17 slots against the baselines' 10, so the
   union is normalised against a taller ideal. Scoring the union truncated to
   ten documents removes the mismatch:

| method | docs | nDCG@10 | MRR@10 | R@10 | R-prec |
|---|---|---|---|---|---|
| BM25 | 10 | **0.799** | 0.909 | 0.471 | 0.458 |
| TF-IDF | 10 | 0.797 | **0.952** | 0.459 | 0.449 |
| union-fasttext @10 | 10 | 0.789 | 0.923 | 0.459 | 0.451 |
| union-fasttext (shipped) | 17.8 | 0.762 | 0.923 | **0.702** | **0.639** |
| fasttext | 10 | 0.662 | 0.818 | 0.353 | 0.351 |

At an equal budget the union is **level** with the lexical baselines, not
behind them. So the standing conclusion — the union is a wider net rather than
a better ranker — survives intact, but the claim that it *costs* ranking
quality does not.

### The decision

**Ship the union on by default** (PRD §8.4). The trade is one-directional:
17.8 documents for Recall@10 0.702 and R-precision 0.639, against 10 documents
for 0.471 and 0.458, with ranking now level at equal depth.

### A caveat on this section's own numbers

The fix was measured on the same eval set it was diagnosed from. The `code`
stratum result needs no inference — its ground truth is exact and the effect is
1.000 against 0.500 — but the main-set deltas (+0.016 nDCG, +0.023 R-precision)
are single unreplicated runs and were **not** significance-tested. They are
also inside the ±0.010–0.014 retraining noise band §8.4 of the PRD documents
for effects of this size. The defect is unambiguous and the fix is
directionally right on every stratum; the magnitude on the main set is not
established.

## 10. The two capability gaps, closed and measured

*2026-08-29, later.*

### 10.1 Known-item retrieval did not exist

Section 8 noted that only 71 of 10,666 abstracts contain a registry code and
suggested indexing the `Trial ID` column. Measured before doing so, over **60
real trial ids drawn five per registry from all twelve registries**:

| | before | after |
|---|---|---|
| requested trial at rank 1 | **0 / 60** | **60 / 60** |
| requested trial anywhere in the result set | 0 / 60 | 60 / 60 |

Not a ranking failure. Every row carries a unique `Trial ID`, retrieval ran on
`abstract`, and nothing indexed the identifier — so the single most common
known-item operation a trial-search tool offers returned the trial **zero**
times. The ground truth here needs no annotator: ids are unique, so the answer
is defined rather than judged. It is the only stratum in the project with no
provenance to declare (`tests/fixtures/eval_queries_known_item.json`,
`scripts/known_item_evaluate.py`).

An identifier is a key, so it gets a lookup, and the lookup precedes the
ranking. Ids are normalised to letters and digits, so `CTRI/2021/05/033883`,
`ctri 2021 05 033883` and `CTRI202105033883` all resolve; two ids colliding
after normalisation are served to nobody rather than served wrongly.

**What it costs, stated plainly.** Round 3's `code` stratum falls from MRR@10
1.000 to 0.667. That stratum scores a *different question* — which trials
**cite** this id — and its gold lists citing trials only, excluding the queried
trial by construction rather than by judgement. Two of its three queries are
themselves trials in the corpus, so promoting the trial inserts a document its
gold cannot contain. **The gold was not re-judged to remove this cost**, which
would have been the easy and dishonest move; both numbers stand, measuring
their own questions. The main 97-query set is unchanged.

### 10.2 Free-standing negation, as an operator

Section 8 Result 5 closed off feature engineering:

> Negation requires removing or inverting shared evidence. Every additive
> feature scheme can only add evidence, and therefore none of them can express
> it.

So the query carries the operator. Cue-and-scope detection in the NegEx style
(Chapman et al., 2001) runs on **both** sides: on the query to find what the
user rules out, and on the document to tell an abstract that asserts the
concept from one that denies it. That second half is the point — the documents
a negated query wants are mostly the ones that mention the concept in order to
negate it, and a plain exclusion filter throws them away.

**Two iterations, both driven by inspecting what the filter removed.**

1. *Cues covering only grammatical negation removed five of six gold documents*
   for `treatment without mechanical ventilation`. They read "reduces the need
   for", "decrease the need of", "risk of need for" — in this domain the
   negated sense is carried by **avoidance language**, not by "not". Adding
   those cues took it to six of eight kept.
2. *A one-word scope is too blunt to filter on.* `spread by people without
   symptoms` parsed to "exclude anything mentioning symptoms", removing the
   asymptomatic-transmission trials the query asked for and taking main-set
   Recall@10 from 0.702 to **0.698, under the PRD target, on one query**. Such
   a phrase names a concept, not an exclusion. Scopes now need two tokens.

**Result.** Overlap@10 between each query and its negated twin — 1.00 means the
negation changed nothing:

| pair | before | after |
|---|---|---|
| non-hospitalized patients with covid-19 | 0.10 | 0.10 |
| **patients not requiring supplemental oxygen** | 0.90 | **0.60** |
| non-severe covid-19 pneumonia | 0.40 | 0.40 |
| **treatment without mechanical ventilation** | 0.80 | **0.20** |
| **mean** | 0.55 | **0.33** |

The two prefix pairs do not move and the two free-standing pairs move a long
way, which is the mechanism claim tested directly: prefix negation already
worked by morphological substitution, and only free-standing negation needed
the operator.

On the negation stratum, union-fasttext:

| | before | after |
|---|---|---|
| P@1 | 0.375 | **0.438** |
| MRR@10 | 0.532 | **0.567** |
| R-precision | 0.427 | **0.448** |
| nDCG@10 | 0.529 | 0.521 |
| Recall@10 | 0.643 | **0.560** |

**It costs recall, and that is not hidden.** Filtering removes documents; two of
the eight gold documents inspected are still removed wrongly. The trade is
precision at the top and pair discrimination against Recall@10 on the two
queries the filter fires on.

**On the main 97-query set it fires on 0 of 97 queries and changes nothing** —
Recall@10 0.702, MRR@10 0.923, all PRD targets met. Set against the bigram
scheme section 8 Result 5 rejected, which moved one pair 0.90 → 0.80 and cost
0.08 to 0.19 nDCG@10 across *every* stratum, this is the better trade by a wide
margin.

### The caveat both share

Two queries drive the negation result and three drive `code`. The audit's own
warning applies: these strata are **diagnostic, not powered**. The cue lexicon
in particular was extended after inspecting failures on those two queries, so
it is fitted to them; the mechanism is validated by the prefix/free-standing
split, but the lexicon's coverage on unseen negations is unmeasured. The
known-item result needs no such caveat — n = 60, exact ground truth, 0.000
against 1.000.

## Current status of the numbers

*Updated 2026-08-29, after §9. Everything above §7 is round-1 history and is
left as written; §§7–8 carry inline correction notes where §9 superseded them.
This table is what currently holds.*

| Claim | Status |
|---|---|
| The lexical baselines beat the embeddings standalone | ✅ **holds — and now on an unbiased pool**: TF-IDF 0.459 / BM25 0.471 against FastText 0.353 / Skip-gram 0.351 |
| The two methods are complementary (66 % unique) | ✅ holds — an overlap measurement, unaffected by pooling |
| Union Recall@10 = 0.955 | ❌ **withdrawn** — `recall@pool`, and pool-bound. Re-judged: **0.702** |
| PRD §8 DoD (Recall@10 ≥ 0.70) met | ⚠️ **met on the number, with the reading stated** — union-fasttext 0.702 from 17.8 documents against a depth-10 ceiling of 0.626. The nDCG deficit was partly a defect and partly a depth mismatch: at an equal 10-document budget the union scores 0.789 against BM25's 0.799 (§9) |
| TF-IDF beats BM25 (0.615 vs 0.403) | ❌ **withdrawn** — the gap was pool membership. Re-judged 0.459 vs 0.471: Δ +0.0116, p = 0.47, **equivalent** |
| Recall@10 ceiling = 0.879 | ❌ **superseded** — 0.626 at depth 10, 0.951 at depth 20, with 17.4 relevant documents per query |
| Tokeniser fix improves retrieval | ✅ **CONFIRMED — §8.** Registry-code Recall@10 0.44 → 1.00 (TF-IDF) and 0.33 → 1.00 (BM25); `CD4` and `CD8` returned an identical top-10 under the old chain and are separated under the new one |
| Negation fix improves retrieval | ⚠️ **half-refuted — §8.** The gain comes from hyphen-joining (`nonhospitalized`, idf 6.45), not from `CLINICAL_KEEP_WORDS`. Free-standing `not` / `without` is retained but inert at idf 2.18 / 3.16, and the two pairs that turn on it are unchanged |
| Known-item retrieval by trial id | ✅ **closed — §10.1.** 0/60 → 60/60 at rank 1, exact ground truth, n = 60 |
| Free-standing negation is unreachable | ✅ **closed as an operator — §10.2.** Pair overlap 0.55 → 0.33, main set unchanged. Costs Recall@10 on the two queries it fires on |
| The union is safe to ship by default | ✅ **resolved — §9.** The known-item damage was an insertion-order tie-break, now fixed: the `code` stratum reaches MRR@10 and nDCG@10 1.000. Recall unchanged at 0.702; ships on by default (PRD §8.4) |
| The judgements are human ground truth | ⚠️ **no** — 986 human, **705 model-generated** (κ = 0.800 against the human labels). That provenance travels with every number in this table |

## What is closed, and what remains

**Closed — round 2, 2026-08-28.** `scripts/make_eval_round2.py` implements
**incremental pooling**, the way TREC admits a new system to an existing
collection: old judgements stay valid, new candidates get judged and added. All
**1,532** outstanding candidates were judged and merged, and `medsearch
evaluate` was re-run — `reports/evaluation.json` carries the corrected numbers
(regenerated 2026-08-28 12:14 UTC). The Recall figures are no longer lower
bounds: the pool now contains BM25's results and the current tokeniser's
output, so all six scored systems are symmetric again.

**Closed — round 3, 2026-08-28.** 22 queries in three strata, scored
separately (§8). Both preprocessing fixes are now measured rather than argued:
the tokeniser change is confirmed with a large effect, and the negation
allowlist is shown to be doing much less than PRD F-12 assumed. The `code`
stratum is the first evaluation data in this project whose ground truth
involves **no judgement of any kind**.

Remaining, in order:

1. ~~**Free-standing negation needs a query parser, not a token list.**~~
   **Built and measured — §10.2.** Pair overlap 0.55 → 0.33, main set
   untouched. What remains is the lexicon's coverage on negations outside the
   two queries it was fitted to.
2. **An independent check.** The κ = 0.800 calibration substitutes for a second
   annotator, not for a clinician. Single-source judgements still carry no
   external error bar, and 705 of them were produced by the same class of
   system being scored.
3. ~~**The product decision.**~~ **Closed 2026-08-29 — §9.** The known-item
   evidence against the union was a fusion defect; once fixed, the union is
   level with the lexical baselines at equal depth and ahead on recall and
   R-precision at its own. It ships on by default. What remains is to replicate
   §9's main-set deltas, which are unreplicated and untested.
4. ~~**Index the `Trial ID` column, or decide not to.**~~ **Done — §10.1.**
   Not by indexing the column into the text ranking, but by treating the
   identifier as a key: 0/60 → 60/60 at rank 1.

**How to quote these numbers.** Give nDCG@10 and R-precision alongside
Recall@10, never Recall alone. State the depth-10 ceiling (0.626) wherever
Recall@10 appears. State the result-set size whenever the union (17.8 docs) is
compared with a depth-10 system, and never compare nDCG across depth factors
without saying so (§9). State that 705 of the 1,691 judgements are
model-generated.
