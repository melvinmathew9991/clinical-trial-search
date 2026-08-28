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
## Current status of the numbers

*Updated 2026-08-28, after round 2. Everything above §7 is round-1 history and
is left as written. This table is what currently holds.*

| Claim | Status |
|---|---|
| The lexical baselines beat the embeddings standalone | ✅ **holds — and now on an unbiased pool**: TF-IDF 0.459 / BM25 0.471 against FastText 0.353 / Skip-gram 0.351 |
| The two methods are complementary (66 % unique) | ✅ holds — an overlap measurement, unaffected by pooling |
| Union Recall@10 = 0.955 | ❌ **withdrawn** — `recall@pool`, and pool-bound. Re-judged: **0.702** |
| PRD §8 DoD (Recall@10 ≥ 0.70) met | ⚠️ **met on the number, not on the reading** — union-fasttext 0.702, but from 17.8 documents against a depth-10 ceiling of 0.626, and by nDCG@10 it loses to both lexical baselines |
| TF-IDF beats BM25 (0.615 vs 0.403) | ❌ **withdrawn** — the gap was pool membership. Re-judged 0.459 vs 0.471: Δ +0.0116, p = 0.47, **equivalent** |
| Recall@10 ceiling = 0.879 | ❌ **superseded** — 0.626 at depth 10, 0.951 at depth 20, with 17.4 relevant documents per query |
| Tokeniser fix improves retrieval | ✅ **CONFIRMED — §8.** Registry-code Recall@10 0.44 → 1.00 (TF-IDF) and 0.33 → 1.00 (BM25); `CD4` and `CD8` returned an identical top-10 under the old chain and are separated under the new one |
| Negation fix improves retrieval | ⚠️ **half-refuted — §8.** The gain comes from hyphen-joining (`nonhospitalized`, idf 6.45), not from `CLINICAL_KEEP_WORDS`. Free-standing `not` / `without` is retained but inert at idf 2.18 / 3.16, and the two pairs that turn on it are unchanged |
| The union is safe to ship by default | ⚠️ **contested — §8.** On known-item code queries the union halves P@1 against the lexical baselines (0.500 vs 1.000): the embedding half contributes a document that cannot be relevant |
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

1. **Free-standing negation needs a query parser, not a token list.** §8
   Results 2 and 5 together show why no feature scheme reaches it: an additive
   representation cannot subtract shared evidence. Bigrams were tried and
   rejected on measurement -- they left the negation pairs where they were and
   cost 0.08 to 0.19 nDCG@10 across every other stratum. Treating negation as
   an operator is a design change, and it is the one open *modelling* question
   left.
2. **An independent check.** The κ = 0.800 calibration substitutes for a second
   annotator, not for a clinician. Single-source judgements still carry no
   external error bar, and 705 of them were produced by the same class of
   system being scored.
3. **The product decision, now with one more input.** Recall@10, nDCG@10 and
   R-precision already disagreed about the union. §8 Result 4 adds that on
   known-item queries the union *halves* precision at rank 1 against a
   baseline that answers them perfectly. Whether to ship 17.8 documents or 10
   is PRD §8.4's question, and it is a product call, not a modelling one.
4. **Index the `Trial ID` column, or decide not to.** Only 71 of 10,666
   abstracts contain a registry code (§8). Searching by trial id is a
   capability this system does not have, and no amount of tokeniser work
   gives it one.

**How to quote these numbers.** Give nDCG@10 and R-precision alongside
Recall@10, never Recall alone. State the depth-10 ceiling (0.626) wherever
Recall@10 appears. State the result-set size whenever the union (17.8 docs) is
compared with a depth-10 system. State that 705 of the 1,691 judgements are
model-generated.
