"""Round 3, part two: does the preprocessing fix actually change retrieval?

The probe in ``round3_probe.py`` measures how the *current* pipeline handles
codes and negation. It cannot say whether the domain-audit fix caused that,
because it only ever runs one chain. This script runs both.

The old chain, reconstructed from the audit (EVALUATION_AUDIT.md sections 3
and 4):

* every digit stripped, hyphens split rather than joined, so ``SARS-CoV-2``
  became ``sars`` + ``cov`` and ``NCT04446429`` became ``nct``
* plain NLTK English stopwords, so ``no`` and ``not`` were discarded while
  ``without`` survived -- the inconsistency PRD F-12 anticipated

Everything else is held constant: same corpus, same lemmatiser, same
``min_token_length``, same rankers, same queries. The only difference is the
token chain, so a difference in the metrics is attributable to it.

**Scope.** The A/B covers the two lexical rankers. Re-training the embeddings
on old-chain tokens would take a further ~4 minutes and is not done here: on
the code stratum the embeddings score ~0 under the *current* chain, so there is
no effect there for the old chain to remove.

Run::

    python scripts/round3_ablation.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from medsearch.runtime import configure_threads

configure_threads()

from medsearch.config import get_settings  # noqa: E402
from medsearch.data.loader import load_corpus  # noqa: E402
from medsearch.logging_conf import configure_logging  # noqa: E402
from medsearch.preprocessing.pipeline import TextPreprocessor  # noqa: E402
from medsearch.search.baseline import TfidfBaseline  # noqa: E402
from medsearch.search.bm25 import BM25Baseline  # noqa: E402

FIELD = "abstract"
TOP_N = 10

#: The pre-audit chain: strip everything that is not a letter or a space. One
#: line, and it removed 6,077 alphanumeric identity tokens from a 4,000-
#: abstract sample -- digits and intra-word hyphens go together.
_OLD_CHAIN_RE = re.compile(r"[^a-z ]")
_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"\w+:\/\/\S+|@[A-Za-z0-9_]+")


class OldChainPreprocessor(TextPreprocessor):
    """The pipeline as it stood before the domain audit."""

    def __init__(self) -> None:
        # keep_words=() restores plain NLTK stopwords: "no" and "not" are
        # dropped again while "without" survives, exactly as before the fix.
        super().__init__(keep_words=())

    def transform(self, text: str) -> list[str]:
        if not text:
            return []
        lowered = _URL_RE.sub(" ", text.lower())
        cleaned = _WS_RE.sub(" ", _OLD_CHAIN_RE.sub(" ", lowered)).strip()
        if not cleaned:
            return []
        return [
            self._lemmatizer.lemmatize(token)
            for token in self._tokenize(cleaned)
            if token not in self._stopwords and len(token) >= self._min_token_length
        ]


NEGATION_PAIRS = (
    ("hospitalized patients with covid-19", "non-hospitalized patients with covid-19"),
    ("patients requiring supplemental oxygen", "patients not requiring supplemental oxygen"),
    ("severe covid-19 pneumonia", "non-severe covid-19 pneumonia"),
    ("treatment with mechanical ventilation", "treatment without mechanical ventilation"),
)
CODES = ("NCT04446429", "NCT04317092", "NCT04372368")
ENTITY_PAIRS = (
    ("IL-6 inhibitor for cytokine storm", "IL-1 blockade in covid-19"),
    ("CD4 T cell response", "CD8 T cell response"),
    ("SARS-CoV-2 neutralising antibody", "MERS-CoV and SARS-CoV-1 comparison"),
)

HEADER = f"  {'pair':<42}{'tfidf old':>11}{'tfidf new':>11}{'bm25 old':>11}{'bm25 new':>11}"


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    corpus = load_corpus(settings.paths.corpus_file)
    trial_ids = corpus["trial_id"].astype(str).tolist()
    text = (corpus["title"].astype(str) + " " + corpus["abstract"].astype(str)).tolist()
    raw = corpus[cast(str, FIELD)].astype(str).tolist()

    chains: dict[str, Any] = {}
    for label, pre in (("new", TextPreprocessor()), ("old", OldChainPreprocessor())):
        tokens = [pre.transform(doc) for doc in raw]
        chains[label] = {
            "pre": pre,
            "tfidf": TfidfBaseline(tokens),
            "bm25": BM25Baseline(tokens),
            "vocab": len({t for doc in tokens for t in doc}),
        }
        print(f"  {label} chain vocabulary: {chains[label]['vocab']:,}")

    def rank(chain: str, system: str, query: str) -> list[str]:
        c = chains[chain]
        toks = c["pre"].transform(query)
        return [trial_ids[h.row_id] for h in c[system].search(toks, top_n=TOP_N)]

    def overlap(chain: str, system: str, a: str, b: str) -> float:
        return len(set(rank(chain, system, a)) & set(rank(chain, system, b))) / TOP_N

    report: dict[str, Any] = {"vocabulary": {k: v["vocab"] for k, v in chains.items()}}

    print("\n\nREGISTRY CODES -- Recall@10 against exact-match ground truth\n")
    print(
        f"  {'code':<14}{'|rel|':>6}{'tfidf old':>11}{'tfidf new':>11}{'bm25 old':>11}{'bm25 new':>11}"
    )
    code_rows = []
    for code in CODES:
        pattern = re.compile(re.escape(code), re.IGNORECASE)
        relevant = {trial_ids[i] for i, t in enumerate(text) if pattern.search(t)}
        row: dict[str, Any] = {"code": code, "n_relevant": len(relevant), "recall_at_10": {}}
        line = f"  {code:<14}{len(relevant):>6}"
        for system in ("tfidf", "bm25"):
            for chain in ("old", "new"):
                hits = len(set(rank(chain, system, code)) & relevant)
                score = hits / max(len(relevant), 1)
                row["recall_at_10"][f"{system}-{chain}"] = round(score, 3)
                line += f"{score:>11.2f}"
        code_rows.append(row)
        print(line)
    report["registry_codes"] = code_rows

    print("\n\nENTITY IDENTITY -- overlap@10 between two entities the old chain collapses")
    print("(high overlap = the system cannot tell them apart)\n")
    print(HEADER)
    entity_rows = []
    for a, b in ENTITY_PAIRS:
        row = {"a": a, "b": b, "overlap": {}}
        label = f"{a.split()[0]} vs {b.split()[0]}"
        line = f"  {label[:40]:<42}"
        for system in ("tfidf", "bm25"):
            for chain in ("old", "new"):
                value = overlap(chain, system, a, b)
                row["overlap"][f"{system}-{chain}"] = round(value, 3)
                line += f"{value:>11.2f}"
        entity_rows.append(row)
        print(line)
    report["entity_pairs"] = entity_rows

    print("\n\nNEGATION -- overlap@10 between a query and its negated twin\n")
    print(HEADER)
    negation_rows = []
    for positive, negated in NEGATION_PAIRS:
        row = {"positive": positive, "negated": negated, "overlap": {}}
        line = f"  {negated[:40]:<42}"
        for system in ("tfidf", "bm25"):
            for chain in ("old", "new"):
                value = overlap(chain, system, positive, negated)
                row["overlap"][f"{system}-{chain}"] = round(value, 3)
                line += f"{value:>11.2f}"
        negation_rows.append(row)
        print(line)
    report["negation_pairs"] = negation_rows

    out = Path("reports/round3_ablation.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
