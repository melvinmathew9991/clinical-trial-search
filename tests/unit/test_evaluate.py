"""Evaluation metrics and the TF-IDF baseline.

The metric implementations are hand-checked against worked examples rather
than a reference library: if Recall@k or MRR is subtly wrong, every conclusion
drawn in Sprint 8 is wrong with it, and a bug here would be invisible in the
final numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medsearch.exceptions import DataError
from medsearch.pipelines.evaluate import (
    EvalQuery,
    _Accumulator,
    _finalise,
    _update,
    check_targets,
    load_eval_set,
)
from medsearch.search.baseline import TfidfBaseline


def score(retrieved: list[str], relevant: set[str], k_values: tuple[int, ...] = (1, 5, 10)) -> dict:
    """Run one query through the accumulator and return its metrics."""
    accumulator = _Accumulator()
    _update(accumulator, retrieved, frozenset(relevant), k_values)
    result = _finalise("test", accumulator, 1)
    return {"recall": result.recall_at, "mrr": result.mrr_at, "precision": result.precision_at}


class TestRecall:
    def test_all_relevant_in_top_k(self) -> None:
        assert score(["a", "b", "c"], {"a", "b"})["recall"][5] == pytest.approx(1.0)

    def test_half_the_relevant_found(self) -> None:
        assert score(["a", "x", "y"], {"a", "b"})["recall"][5] == pytest.approx(0.5)

    def test_none_found(self) -> None:
        assert score(["x", "y"], {"a", "b"})["recall"][5] == pytest.approx(0.0)

    def test_cut_off_is_respected(self) -> None:
        # 'b' sits at rank 2, so it counts at k=5 but not k=1.
        result = score(["a", "b"], {"a", "b"})
        assert result["recall"][1] == pytest.approx(0.5)
        assert result["recall"][5] == pytest.approx(1.0)

    def test_empty_retrieval_scores_zero(self) -> None:
        assert score([], {"a"})["recall"][10] == pytest.approx(0.0)


class TestMRR:
    def test_first_position_is_one(self) -> None:
        assert score(["a", "x"], {"a"})["mrr"][10] == pytest.approx(1.0)

    def test_second_position_is_one_half(self) -> None:
        assert score(["x", "a"], {"a"})["mrr"][10] == pytest.approx(0.5)

    def test_third_position_is_one_third(self) -> None:
        assert score(["x", "y", "a"], {"a"})["mrr"][10] == pytest.approx(1 / 3)

    def test_only_the_first_hit_counts(self) -> None:
        # Two relevant documents at ranks 2 and 3 -> still 1/2, not 1/2 + 1/3.
        assert score(["x", "a", "b"], {"a", "b"})["mrr"][10] == pytest.approx(0.5)

    def test_miss_scores_zero(self) -> None:
        assert score(["x", "y"], {"a"})["mrr"][10] == pytest.approx(0.0)

    def test_hit_beyond_cut_off_scores_zero(self) -> None:
        assert score(["x", "y", "a"], {"a"})["mrr"][1] == pytest.approx(0.0)


class TestPrecision:
    def test_all_top_k_relevant(self) -> None:
        assert score(["a", "b"], {"a", "b"})["precision"][1] == pytest.approx(1.0)

    def test_one_of_five_relevant(self) -> None:
        assert score(["a", "w", "x", "y", "z"], {"a"})["precision"][5] == pytest.approx(0.2)


class TestEvalQuery:
    def test_from_dict(self) -> None:
        q = EvalQuery.from_dict({"query": "lung failure", "relevant": ["NCT1", "NCT2"]})
        assert q.query == "lung failure"
        assert q.relevant == frozenset({"NCT1", "NCT2"})

    def test_note_is_optional(self) -> None:
        assert EvalQuery.from_dict({"query": "q", "relevant": ["a"]}).note == ""

    def test_missing_keys_rejected(self) -> None:
        with pytest.raises(DataError, match="needs 'query' and 'relevant'"):
            EvalQuery.from_dict({"query": "q"})

    def test_ids_are_coerced_to_strings(self) -> None:
        assert EvalQuery.from_dict({"query": "q", "relevant": [123]}).relevant == frozenset({"123"})


class TestLoadEvalSet:
    def _write(self, tmp_path: Path, payload: object) -> Path:
        target = tmp_path / "eval_queries.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        return target

    def test_loads_a_list(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, [{"query": "q", "relevant": ["a"]}])
        assert len(load_eval_set(path)) == 1

    def test_loads_a_wrapped_object(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"queries": [{"query": "q", "relevant": ["a"]}]})
        assert len(load_eval_set(path)) == 1

    def test_missing_file_explains_how_to_make_one(self, tmp_path: Path) -> None:
        with pytest.raises(DataError) as exc_info:
            load_eval_set(tmp_path / "absent.json")
        message = str(exc_info.value)
        assert "make_eval_candidates" in message
        assert "human relevance judgements" in message

    def test_malformed_json_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / "eval_queries.json"
        target.write_text("{not json", encoding="utf-8")
        with pytest.raises(DataError, match="not valid JSON"):
            load_eval_set(target)

    def test_empty_set_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="no queries"):
            load_eval_set(self._write(tmp_path, []))

    def test_unlabelled_query_rejected(self, tmp_path: Path) -> None:
        # An unlabelled query scores 0 everywhere and would read as a
        # retrieval failure rather than a missing label.
        path = self._write(tmp_path, [{"query": "q", "relevant": []}])
        with pytest.raises(DataError, match="no relevant documents"):
            load_eval_set(path)


class TestCheckTargets:
    def _report(self, recall: float, mrr: float, p95: float = 5.0) -> dict:
        return {
            "targets": {"recall_at_10": 0.70, "mrr_at_10": 0.45, "latency_p95_ms": 300.0},
            "results": [
                {
                    "method": "skipgram",
                    "recall_at": {"10": recall},
                    "mrr_at": {"10": mrr},
                    "latency_ms": {"p95": p95},
                }
            ],
        }

    def test_all_targets_met(self) -> None:
        assert check_targets(self._report(0.80, 0.50)) == []

    def test_recall_miss_reported(self) -> None:
        failures = check_targets(self._report(0.50, 0.50))
        assert len(failures) == 1
        assert "Recall@10" in failures[0]

    def test_mrr_miss_reported(self) -> None:
        assert "MRR@10" in check_targets(self._report(0.80, 0.20))[0]

    def test_latency_miss_reported(self) -> None:
        assert "p95" in check_targets(self._report(0.80, 0.50, p95=500.0))[0]

    def test_baseline_is_not_judged_against_targets(self) -> None:
        report = self._report(0.80, 0.50)
        report["results"].append(
            {
                "method": "tfidf-baseline",
                "recall_at": {"10": 0.10},
                "mrr_at": {"10": 0.05},
                "latency_ms": {"p95": 1.0},
            }
        )
        assert check_targets(report) == []

    def test_no_embedding_method_is_an_error(self) -> None:
        report = self._report(0.8, 0.5)
        report["results"] = [
            {"method": "tfidf-baseline", "recall_at": {}, "mrr_at": {}, "latency_ms": {}}
        ]
        assert check_targets(report) != []


class TestTfidfBaseline:
    @pytest.fixture
    def baseline(self) -> TfidfBaseline:
        return TfidfBaseline(
            [
                ["lung", "failure", "respiratory"],
                ["breathing", "lung", "exercise"],
                ["vaccine", "antibody", "immune"],
                ["kidney", "renal", "dialysis"],
            ]
        )

    def test_size_and_vocabulary(self, baseline: TfidfBaseline) -> None:
        assert baseline.size == 4
        assert baseline.vocabulary_size == 11

    def test_exact_term_match_ranks_first(self, baseline: TfidfBaseline) -> None:
        assert baseline.search(["kidney"], top_n=1)[0].row_id == 3

    def test_shared_term_matches_both_documents(self, baseline: TfidfBaseline) -> None:
        assert {h.row_id for h in baseline.search(["lung"], top_n=4)} == {0, 1}

    def test_scores_descend(self, baseline: TfidfBaseline) -> None:
        scores = [h.score for h in baseline.search(["lung", "failure"], top_n=4)]
        assert scores == sorted(scores, reverse=True)

    def test_cosine_scores_are_bounded(self, baseline: TfidfBaseline) -> None:
        for hit in baseline.search(["lung", "failure"], top_n=4):
            assert 0.0 <= hit.score <= 1.0001

    def test_out_of_vocabulary_query_returns_nothing(self, baseline: TfidfBaseline) -> None:
        # This is the failure mode embeddings exist to fix, so it must be a
        # recorded outcome rather than an exception.
        assert baseline.search(["zzzz"], top_n=4) == []

    def test_empty_query_returns_nothing(self, baseline: TfidfBaseline) -> None:
        assert baseline.search([], top_n=4) == []

    def test_rare_term_outranks_common_term(self) -> None:
        # IDF must make 'dialysis' (1 doc) more discriminative than
        # 'patient' (all 3 docs).
        baseline = TfidfBaseline(
            [
                ["patient", "dialysis"],
                ["patient", "vaccine"],
                ["patient", "lung"],
            ]
        )
        assert baseline.search(["patient", "dialysis"], top_n=1)[0].row_id == 0

    def test_empty_corpus_is_safe(self) -> None:
        baseline = TfidfBaseline([])
        assert baseline.size == 0
        assert baseline.search(["lung"], top_n=5) == []

    def test_top_n_larger_than_corpus(self, baseline: TfidfBaseline) -> None:
        assert len(baseline.search(["lung"], top_n=999)) <= baseline.size
