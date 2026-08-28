"""Generate a candidate sheet for human relevance labelling.

**This script does not decide relevance.** It cannot: relevance is what every
retrieval metric is measured against, and a machine-generated judgement would
be circular — the retrieval system grading its own homework.

What it does is turn labelling from a writing task into a reviewing one. For
each query it pools candidates from *three* sources so no single method's
biases dominate the pool:

* Skip-gram embeddings
* FastText embeddings
* TF-IDF keyword baseline

That pooling design is standard TREC practice, and it matters here: if
candidates came only from the embedding models, any document that only keyword
search can find would be invisible to the labeller, and the baseline would be
unfairly penalised on documents that were never shown.

Output is ``reports/eval_candidates.json`` with every ``relevant`` field left
**empty** for a human to fill in.

Usage::

    python scripts/make_eval_candidates.py                 # default queries
    python scripts/make_eval_candidates.py --top-n 15
    python scripts/make_eval_candidates.py --queries my_queries.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medsearch.runtime import configure_threads

configure_threads()

from medsearch.config import get_settings  # noqa: E402
from medsearch.logging_conf import configure_logging, get_logger  # noqa: E402

logger = get_logger("medsearch.eval_candidates")

#: Queries spanning the corpus's main clinical themes, phrased the way a
#: researcher would type them rather than the way abstracts are written --
#: which is the whole point of testing semantic retrieval.
DEFAULT_QUERIES: tuple[str, ...] = (
    "lung failure",
    "breathing difficulty",
    "acute respiratory distress",
    "mechanical ventilation weaning",
    "oxygen therapy at home",
    "kidney injury",
    "dialysis in critical illness",
    "vaccine immunogenicity",
    "antibody response after second dose",
    "immunocompromised vaccine recipients",
    "blood clots and anticoagulation",
    "pulmonary embolism risk",
    "cytokine storm treatment",
    "steroid therapy for inflammation",
    "antiviral drug early treatment",
    "viral load reduction",
    "long covid fatigue",
    "loss of smell and taste",
    "mental health of healthcare workers",
    "anxiety during the pandemic",
    "pregnancy outcomes",
    "children and adolescents",
    "elderly care home residents",
    "diabetes as a risk factor",
    "obesity and severe disease",
    "rehabilitation after hospital discharge",
    "telemedicine remote monitoring",
    "convalescent plasma therapy",
    "monoclonal antibody treatment",
    "heart damage myocarditis",
)


def _load_queries(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_QUERIES)
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=None, help="One query per line.")
    parser.add_argument("--top-n", type=int, default=10, help="Candidates per method per query.")
    parser.add_argument("--field", default="abstract", choices=["abstract", "title"])
    parser.add_argument("--abstract-chars", type=int, default=320)
    parser.add_argument(
        "--output", type=Path, default=None, help="Defaults to reports/eval_candidates.json"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)

    from medsearch.data.loader import load_corpus
    from medsearch.exceptions import MedSearchError
    from medsearch.pipelines.train import load_search_engine, run_preprocessing
    from medsearch.preprocessing.pipeline import TextPreprocessor
    from medsearch.search.baseline import TfidfBaseline

    queries = _load_queries(args.queries)
    logger.info("Pooling candidates for %d queries", len(queries))

    try:
        engines = {
            model: load_search_engine(settings, model, args.field)
            for model in ("skipgram", "fasttext")
        }
        corpus = load_corpus(settings.paths.corpus_file)
        trial_ids = corpus["trial_id"].astype(str).tolist()
        cache, _, _ = run_preprocessing(settings, args.field)
        baseline = TfidfBaseline(cache)
        preprocessor = TextPreprocessor()
    except MedSearchError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    by_trial = {
        str(row.trial_id): {
            "trial_id": str(row.trial_id),
            "title": str(row.title),
            "abstract": str(row.abstract)[: args.abstract_chars],
            "publication_date": str(row.publication_date),
        }
        for row in corpus.itertuples()
    }

    entries = []
    for query in queries:
        pooled: dict[str, set[str]] = {}

        for model, engine in engines.items():
            for result in engine.search(query, top_n=args.top_n).results:
                pooled.setdefault(result.trial_id, set()).add(model)

        tokens = preprocessor.transform(query)
        for hit in baseline.search(tokens, top_n=args.top_n):
            pooled.setdefault(trial_ids[hit.row_id], set()).add("tfidf")

        candidates = [
            {**by_trial[tid], "found_by": sorted(sources)}
            for tid, sources in pooled.items()
            if tid in by_trial
        ]
        # Documents every method agrees on first -- usually the easiest calls,
        # which lets a labeller build momentum before the contested ones.
        candidates.sort(key=lambda c: (-len(c["found_by"]), c["trial_id"]))

        entries.append(
            {
                "query": query,
                "relevant": [],
                "note": "",
                "_candidates": candidates,
            }
        )
        logger.info("  %-42s %d pooled candidates", query, len(candidates))

    output = args.output or (settings.paths.report_dir / "eval_candidates.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_instructions": [
            "For each entry, read the documents under '_candidates'.",
            "Copy the trial_id of every RELEVANT one into the 'relevant' list.",
            "'found_by' shows which methods surfaced it -- ignore it when judging;",
            "  it is recorded only so pooling bias can be audited afterwards.",
            "Add a short 'note' if a call was borderline.",
            "Aim for at least one relevant document per query; delete a query you",
            "  cannot label rather than leaving it empty.",
            "When finished, strip the '_candidates' keys and save the file as",
            "  tests/fixtures/eval_queries.json",
        ],
        "queries": entries,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(len(e["_candidates"]) for e in entries)
    print()
    print(f"Wrote {output}")
    print(
        f"  {len(entries)} queries, {total} pooled candidates "
        f"({total / max(len(entries), 1):.1f} per query)"
    )
    print()
    print("Next: label the 'relevant' lists, then save as tests/fixtures/eval_queries.json")
    print("      and run `medsearch evaluate`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
