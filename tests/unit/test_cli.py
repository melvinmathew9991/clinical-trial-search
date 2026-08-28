"""CLI surface: argument validation, exit codes, and error rendering.

Uses typer's ``CliRunner``, so no subprocess is spawned and the suite stays
fast. Commands that would train a model are covered in the integration test.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from medsearch.cli import _as_field, _as_model, app
from medsearch.exceptions import ConfigurationError

runner = CliRunner()


class TestValidators:
    @pytest.mark.parametrize("value", ["skipgram", "fasttext"])
    def test_valid_models_accepted(self, value: str) -> None:
        assert _as_model(value) == value

    def test_all_rejected_unless_permitted(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown model"):
            _as_model("all")

    def test_all_accepted_when_permitted(self) -> None:
        assert _as_model("all", allow_all=True) == "all"

    def test_unknown_model_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown model"):
            _as_model("word2vec")

    def test_model_error_lists_choices(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            _as_model("nope", allow_all=True)
        message = str(exc_info.value)
        assert "skipgram" in message and "fasttext" in message and "all" in message

    @pytest.mark.parametrize("value", ["abstract", "title"])
    def test_valid_fields_accepted(self, value: str) -> None:
        assert _as_field(value) == value

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown field"):
            _as_field("summary")

    def test_field_error_lists_choices(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            _as_field("summary")
        assert "abstract" in str(exc_info.value)

    def test_validation_is_case_sensitive(self) -> None:
        with pytest.raises(ConfigurationError):
            _as_field("Abstract")


class TestHelp:
    def test_bare_invocation_shows_help(self) -> None:
        result = runner.invoke(app, [])
        assert "medsearch" in result.output.lower()

    @pytest.mark.parametrize(
        "command", ["doctor", "train", "search", "preprocess", "evaluate", "index"]
    )
    def test_every_command_has_help(self, command: str) -> None:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0

    def test_index_subcommands_are_registered(self) -> None:
        result = runner.invoke(app, ["index", "--help"])
        assert "build" in result.output
        assert "info" in result.output


class TestDoctor:
    def test_reports_machine_capacity(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEDSEARCH_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("MEDSEARCH_MIN_FREE_MEMORY_GB", "0.0")
        from medsearch.config import get_settings

        get_settings.cache_clear()
        try:
            result = runner.invoke(app, ["doctor"])
            assert "Logical cores" in result.output
            assert "Training workers" in result.output
        finally:
            get_settings.cache_clear()

    def test_missing_corpus_is_a_failure(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEDSEARCH_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("MEDSEARCH_MIN_FREE_MEMORY_GB", "0.0")
        from medsearch.config import get_settings

        get_settings.cache_clear()
        try:
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 1
            assert "MISSING" in result.output
        finally:
            get_settings.cache_clear()

    def test_impossible_memory_floor_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEDSEARCH_MIN_FREE_MEMORY_GB", "9999")
        from medsearch.config import get_settings

        get_settings.cache_clear()
        try:
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 1
            assert "FAIL" in result.output
        finally:
            get_settings.cache_clear()

    def test_reports_the_ngram_matrix_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEDSEARCH_MIN_FREE_MEMORY_GB", "0.0")
        from medsearch.config import get_settings

        get_settings.cache_clear()
        try:
            result = runner.invoke(app, ["doctor"])
            assert "FastText n-gram matrix" in result.output
        finally:
            get_settings.cache_clear()


class TestInvalidArguments:
    def test_unknown_model_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["train", "--model", "word2vec"])
        assert result.exit_code != 0

    def test_unknown_field_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["search", "query", "--field", "summary"])
        assert result.exit_code != 0

    def test_unknown_command_exits_nonzero(self) -> None:
        assert runner.invoke(app, ["nonexistent"]).exit_code != 0


class TestEvaluate:
    """`evaluate` is implemented, but refuses to run without human labels."""

    def _isolated(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEDSEARCH_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("MEDSEARCH_MODEL_DIR", f"{tmp_path}/models")
        from medsearch.config import get_settings

        get_settings.cache_clear()

    def test_missing_eval_set_exits_nonzero(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._isolated(tmp_path, monkeypatch)
        try:
            result = runner.invoke(app, ["evaluate", "--eval-file", f"{tmp_path}/absent.json"])
            assert result.exit_code != 0
        finally:
            from medsearch.config import get_settings

            get_settings.cache_clear()

    def test_missing_eval_set_points_at_the_candidate_script(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # It must never silently fall back to machine-generated judgements.
        self._isolated(tmp_path, monkeypatch)
        try:
            result = runner.invoke(app, ["evaluate", "--eval-file", f"{tmp_path}/absent.json"])
            assert "make_eval_candidates" in result.output
            assert "Traceback" not in result.output
        finally:
            from medsearch.config import get_settings

            get_settings.cache_clear()


class TestUntrainedErrors:
    """A domain error must render as a message, never a traceback."""

    def test_search_without_a_model_is_actionable(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEDSEARCH_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("MEDSEARCH_MODEL_DIR", str(tmp_path) + "/models")
        from medsearch.config import get_settings

        get_settings.cache_clear()
        try:
            result = runner.invoke(app, ["search", "lung failure"])
            assert result.exit_code == 1
            assert "Traceback" not in result.output
            assert "medsearch train" in result.output
        finally:
            get_settings.cache_clear()

    def test_index_info_without_an_index_exits_one(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEDSEARCH_DATA_DIR", str(tmp_path))
        from medsearch.config import get_settings

        get_settings.cache_clear()
        try:
            result = runner.invoke(app, ["index", "info", "--model", "skipgram"])
            assert result.exit_code == 1
        finally:
            get_settings.cache_clear()


class TestDefaultModelResolution:
    """`--model` omitted must follow ``settings.default_model``.

    Sprint 8.4 moved the shipped model to FastText in config, and the change
    was silently inert: ``search`` carried its own ``"skipgram"`` literal in
    the signature, so the setting reached nothing a user touches. These pin
    the wiring rather than the current value.
    """

    def _run(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch, model: str) -> str:
        monkeypatch.setenv("MEDSEARCH_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("MEDSEARCH_MODEL_DIR", str(tmp_path) + "/models")
        monkeypatch.setenv("MEDSEARCH_DEFAULT_MODEL", model)
        from medsearch.config import get_settings

        get_settings.cache_clear()
        try:
            # No artefacts exist, so this exits 1 -- but only after the model
            # has been resolved and named in the error path.
            return runner.invoke(app, ["search", "lung failure"]).output
        finally:
            get_settings.cache_clear()

    @pytest.mark.parametrize("model", ["fasttext", "skipgram"])
    def test_search_uses_the_configured_default(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        assert model in self._run(tmp_path, monkeypatch, model)

    def test_explicit_flag_beats_the_configured_default(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEDSEARCH_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("MEDSEARCH_MODEL_DIR", str(tmp_path) + "/models")
        monkeypatch.setenv("MEDSEARCH_DEFAULT_MODEL", "fasttext")
        from medsearch.config import get_settings

        get_settings.cache_clear()
        try:
            output = runner.invoke(app, ["search", "lung failure", "-m", "skipgram"]).output
            assert "skipgram" in output
            assert "fasttext" not in output
        finally:
            get_settings.cache_clear()
