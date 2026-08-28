"""Tests for logging configuration and the ``stage`` timer.

Two things here are contracts rather than conveniences:

* **``stage`` must re-raise.** Rules section 4 forbids swallowing an error to
  return a default, and ``stage`` wraps every expensive operation in the
  pipeline. If it ever logged and continued, a failed training run would look
  like a successful one.
* **JSON logging must actually emit JSON.** App Service sets
  ``MEDSEARCH_LOG_JSON=true`` (``deploy/azure/app-service/site-config.json``)
  because the log pipeline parses structured records. A formatter that quietly
  fell back to plain text would make production logs unqueryable, and nothing
  else would notice.

Neither had a test before the pre-deployment audit.
"""

from __future__ import annotations

import json
import logging

import pytest

from medsearch.logging_conf import configure_logging, get_logger, stage


@pytest.fixture(autouse=True)
def _isolated_logging() -> object:
    """Restore handlers *and* the module-level configured latch.

    ``configure_logging`` is deliberately idempotent: after the first call it
    only adjusts the level and leaves the formatter alone. Without resetting
    the latch, ``json_output=True`` here would silently do nothing and the
    JSON test would assert against a plain formatter.
    """
    import medsearch.logging_conf as module

    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    saved_latch = module._CONFIGURED
    module._CONFIGURED = False
    yield
    root.handlers, root.level = saved_handlers, saved_level
    module._CONFIGURED = saved_latch


class TestConfigureLogging:
    """Level and formatter selection."""

    def test_sets_the_requested_level(self) -> None:
        configure_logging("WARNING")
        assert logging.getLogger().level == logging.WARNING

    def test_is_idempotent(self) -> None:
        """`_bootstrap` runs per CLI command; repeats must not stack handlers."""
        configure_logging("INFO")
        first = len(logging.getLogger().handlers)
        configure_logging("INFO")
        assert len(logging.getLogger().handlers) == first

    @staticmethod
    def _format(message: str) -> str:
        """Render one record through whatever formatter is configured.

        Formatting the record directly rather than capturing stdout: pytest's
        ``caplog`` plugin installs its own root handler, so ``capsys`` sees
        nothing and the test would pass for the wrong reason.
        """
        handler = logging.getLogger().handlers[-1]
        record = logging.LogRecord(
            name="medsearch.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=message,
            args=(),
            exc_info=None,
        )
        assert handler.formatter is not None
        return handler.formatter.format(record)

    def test_json_output_emits_parseable_records(self) -> None:
        configure_logging("INFO", json_output=True)
        assert json.loads(self._format("hello world"))["message"] == "hello world"

    def test_plain_output_is_not_json(self) -> None:
        configure_logging("INFO", json_output=False)
        rendered = self._format("plain line")
        assert "plain line" in rendered
        with pytest.raises(json.JSONDecodeError):
            json.loads(rendered)

    def test_json_output_only_applies_on_the_first_call(self) -> None:
        """The latch is deliberate, and worth knowing about.

        A second entrypoint cannot switch the format later, which is why the
        Streamlit app must pass ``json_output`` on its own first call rather
        than relying on something else having done it.
        """
        configure_logging("INFO", json_output=False)
        configure_logging("INFO", json_output=True)
        with pytest.raises(json.JSONDecodeError):
            json.loads(self._format("still plain"))


class TestGetLogger:
    """Namespacing."""

    def test_returns_the_named_logger(self) -> None:
        assert get_logger("medsearch.thing").name == "medsearch.thing"

    def test_same_name_returns_the_same_object(self) -> None:
        assert get_logger("medsearch.thing") is get_logger("medsearch.thing")


class TestStage:
    """The timer that wraps every expensive step."""

    def test_logs_start_and_finish(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO), stage("demo"):
            pass
        text = caplog.text
        assert "started" in text
        assert "finished" in text

    def test_reports_rss(self, caplog: pytest.LogCaptureFixture) -> None:
        """Sprint 8 caught two over-conservative memory floors with these."""
        with caplog.at_level(logging.INFO), stage("demo"):
            pass
        assert "rss" in caplog.text.lower()

    def test_reraises_the_original_exception(self) -> None:
        """Rules section 4: never swallow an error to return a default."""
        sentinel = RuntimeError("boom")
        with pytest.raises(RuntimeError) as caught, stage("demo"):
            raise sentinel
        assert caught.value is sentinel

    def test_logs_failure_with_a_traceback(self, caplog: pytest.LogCaptureFixture) -> None:
        """`exc_info=True` is what makes the re-raise diagnosable."""
        with caplog.at_level(logging.ERROR), pytest.raises(ValueError), stage("demo"):
            raise ValueError("boom")
        assert "FAILED" in caplog.text
        assert "ValueError" in caplog.text

    def test_accepts_an_explicit_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO), stage("demo", get_logger("medsearch.custom")):
            pass
        assert any(record.name == "medsearch.custom" for record in caplog.records)
