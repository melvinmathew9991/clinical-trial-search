"""Incremental pooling: candidates the current systems surface that nobody judged.

Why this exists
---------------

The eval set's 986 relevance judgements were labelled over a pool built from
the *then-current* systems. When the tokeniser was fixed in the domain audit,
retrieval changed, and 113 previously-pooled relevant documents fell outside
what the systems now return. Measured Recall@10 dropped for **every** method,
including the TF-IDF baseline that the embedding changes cannot touch.

That is not a regression. It is the signature of a pool-bound eval set:

======================  =================  ==================
Configuration           pool membership    measured Recall@10
======================  =================  ==================
old tokeniser           94.3 %             0.955
new tokeniser           85.3 %             0.862
======================  =================  ==================

The metric tracks *"how much of the frozen pool do you still return"*, not
retrieval quality. It therefore cannot evaluate any change to the retrieval
pipeline -- only agreement with the pipeline that built it.

What this script does
---------------------

Incremental pooling, which is how TREC admits a new system to an existing
collection: the old judgements stay valid -- a human said those documents were
relevant and that does not expire -- and the *new* candidates get judged and
added. This emits only documents that

* the current systems return in their top-N, and
* carry no existing judgement for that query.

Every ``relevant`` field is left empty. This script does not decide relevance;
a machine-generated judgement would be the retrieval system grading its own
homework, which is the failure it exists to prevent.

Run::

    python scripts/make_eval_round2.py
    # label reports/eval_round2_candidates.json, then merge into
    # tests/fixtures/eval_queries.json and re-run `medsearch evaluate`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from medsearch.config import get_settings
from medsearch.data.loader import load_corpus
from medsearch.logging_conf import configure_logging, get_logger
from medsearch.pipelines.evaluate import load_eval_set
from medsearch.pipelines.train import load_search_engine, run_preprocessing
from medsearch.preprocessing.pipeline import TextPreprocessor
from medsearch.search.baseline import TfidfBaseline
from medsearch.search.bm25 import BM25Baseline

logger = get_logger("medsearch.eval_round2")

FIELD = "abstract"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=10, help="Candidates per method per query.")
    parser.add_argument("--abstract-chars", type=int, default=320)
    parser.add_argument("--output", type=Path, default=Path("reports/eval_round2_candidates.json"))
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    queries = load_eval_set(Path("tests/fixtures/eval_queries.json"))

    corpus = load_corpus(settings.paths.corpus_file)
    trial_ids = corpus["trial_id"].astype(str).tolist()
    by_id = {str(row.trial_id): row for row in corpus.itertuples() if hasattr(row, "trial_id")}
    cache, _, _ = run_preprocessing(settings, cast(Any, FIELD))
    baseline = TfidfBaseline(cache)
    # BM25 contributes to the pool from round 2 onward. It was never in round 1,
    # and it shows: 63.2% of its top-10 is unjudged against TF-IDF's 41.2%,
    # purely because TF-IDF helped define the ground truth and BM25 did not.
    # Scoring a system against a pool it never entered measures the pool.
    bm25 = BM25Baseline(cache)
    pre = TextPreprocessor()
    engines = {
        name: load_search_engine(settings, cast(Any, name), cast(Any, FIELD))
        for name in ("skipgram", "fasttext")
    }

    sheet: list[dict[str, Any]] = []
    new_total = 0

    for item in queries:
        judged = set(item.relevant)
        found: dict[str, list[str]] = {}
        for name, engine in engines.items():
            found[name] = [r.trial_id for r in engine.search(item.query, top_n=args.top_n).results]
        tokens = pre.transform(item.query)
        found["tfidf"] = [trial_ids[h.row_id] for h in baseline.search(tokens, top_n=args.top_n)]
        found["bm25"] = [trial_ids[h.row_id] for h in bm25.search(tokens, top_n=args.top_n)]

        sources: dict[str, list[str]] = {}
        for name, ids in found.items():
            for doc in ids:
                sources.setdefault(doc, []).append(name)

        unjudged = [doc for doc in sources if doc not in judged]
        new_total += len(unjudged)
        if not unjudged:
            continue

        candidates = []
        for doc in unjudged:
            row = by_id.get(doc)
            candidates.append(
                {
                    "trial_id": doc,
                    "found_by": sorted(sources[doc]),
                    "title": str(getattr(row, "title", ""))[:200] if row is not None else "",
                    "abstract": (
                        str(getattr(row, "abstract", ""))[: args.abstract_chars]
                        if row is not None
                        else ""
                    ),
                    "relevant": None,
                }
            )
        sheet.append(
            {
                "query": item.query,
                "already_judged_relevant": len(judged),
                "new_candidates": len(candidates),
                "candidates": candidates,
            }
        )

    payload = {
        "_instructions": (
            "Set every `relevant` field to true or false. null means unjudged and "
            "will be treated as NOT relevant, which is what currently biases the "
            "measurement against the new pipeline. Merge the true ones into "
            "tests/fixtures/eval_queries.json, then re-run `medsearch evaluate`."
        ),
        "_round": 2,
        "_reason": "Incremental pooling after the tokeniser fix changed what is retrieved.",
        "queries_with_new_candidates": len(sheet),
        "new_candidates_total": new_total,
        "queries": sheet,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n  queries needing new judgements   {len(sheet)} of {len(queries)}")
    print(f"  unjudged candidates surfaced     {new_total}")
    print(f"  mean per affected query          {new_total / max(len(sheet), 1):.1f}")
    print(f"\n  wrote {args.output}")
    print("  These are currently scored as NOT relevant. Until they are judged,")
    print("  every Recall number in reports/evaluation.json is a lower bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
