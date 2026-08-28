"""Text normalisation primitives and the full cleaning chain.

The domain cases in :class:`TestBiomedicalIdentity` are the reason this module
was rewritten. The previous chain stripped every digit, which collapsed
``CD4`` and ``CD8`` onto one token, turned every ``NCT…`` registry ID into
``nct``, and reduced ``BNT162b2`` to ``bntb``. Those tests exist so the
regression cannot return quietly.
"""

from __future__ import annotations

import pytest

from medsearch.preprocessing.normalizer import (
    clean_text,
    collapse_whitespace,
    join_intraword_marks,
    strip_symbols,
    strip_urls,
    to_lowercase,
)


class TestPrimitives:
    def test_lowercase(self) -> None:
        assert to_lowercase("COVID-19 Patients") == "covid-19 patients"

    def test_strip_urls(self) -> None:
        assert "http" not in strip_urls("see http://example.com/x for detail")

    def test_strip_handles(self) -> None:
        assert "@author" not in strip_urls("reported by @author today")

    def test_join_intraword_hyphen(self) -> None:
        assert join_intraword_marks("sars-cov-2") == "sarscov2"

    def test_join_intraword_decimal(self) -> None:
        assert join_intraword_marks("0.5 mg") == "05 mg"

    def test_spaced_dash_is_not_joined(self) -> None:
        """A dash used as punctuation must still separate words."""
        assert join_intraword_marks("severe - fatal") == "severe - fatal"

    def test_strip_symbols_keeps_alphanumerics(self) -> None:
        assert strip_symbols("p<0.05 (n=12)").split() == ["p", "0", "05", "n", "12"]

    def test_collapse_whitespace(self) -> None:
        assert collapse_whitespace("a  \n b \t c") == "a b c"

    def test_collapse_whitespace_trims(self) -> None:
        assert collapse_whitespace("  padded  ") == "padded"


class TestBiomedicalIdentity:
    """Digits carry meaning in clinical text. These pin that they survive."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("CD4 and CD8", "cd4 and cd8"),
            ("ACE2 receptor", "ace2 receptor"),
            ("SARS-CoV-2", "sarscov2"),
            ("COVID-19", "covid19"),
            ("BNT162b2", "bnt162b2"),
            ("NCT04508933", "nct04508933"),
            ("interleukin-6", "interleukin6"),
            ("HbA1c", "hba1c"),
            ("SpO2 and PaO2", "spo2 and pao2"),
        ],
    )
    def test_identity_is_preserved(self, text: str, expected: str) -> None:
        assert clean_text(text) == expected

    def test_cd4_and_cd8_stay_distinct(self) -> None:
        """The regression that motivated the rewrite: two markers, one token."""
        assert clean_text("CD4") != clean_text("CD8")

    def test_registry_ids_stay_distinct(self) -> None:
        """Every NCT id used to normalise to the bare string "nct"."""
        assert clean_text("NCT04508933") != clean_text("NCT04549636")

    def test_interleukin_variants_stay_distinct(self) -> None:
        assert clean_text("interleukin-6") != clean_text("interleukin-1")

    def test_unicode_dashes_join_like_ascii(self) -> None:
        """Publishers emit en-dashes and non-breaking hyphens in abstracts."""
        assert clean_text("SARS–CoV‑2") == "sarscov2"  # noqa: RUF001 - the ambiguous dashes are the subject of the test

    def test_slash_separates_distinct_measures(self) -> None:
        """PaO2/FiO2 is two measures, not one token."""
        assert clean_text("PaO2/FiO2").split() == ["pao2", "fio2"]


class TestCleanText:
    def test_full_chain(self) -> None:
        assert clean_text("COVID-19 affects 1,200 patients (see http://x.co)") == (
            "covid19 affects 1200 patients see"
        )

    def test_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_whitespace_only(self) -> None:
        assert clean_text("   \n\t  ") == ""

    def test_punctuation_only(self) -> None:
        assert clean_text("!!!???...") == ""

    def test_bare_numbers_survive_normalisation(self) -> None:
        """They are dropped later by min_token_length, not here."""
        assert clean_text("12345") == "12345"

    def test_newlines_are_removed(self) -> None:
        assert "\n" not in clean_text("line one\nline two")

    def test_output_has_no_double_spaces(self) -> None:
        assert "  " not in clean_text("COVID-19  ,  severe   illness")

    def test_output_is_stripped(self) -> None:
        result = clean_text("  leading and trailing  ")
        assert result == result.strip()

    def test_is_idempotent(self) -> None:
        once = clean_text("COVID-19 affects 1,200 patients")
        assert clean_text(once) == once

    def test_does_not_mutate_input(self) -> None:
        original = "COVID-19 Patients (n=1200)"
        clean_text(original)
        assert original == "COVID-19 Patients (n=1200)"

    def test_clinical_terms_survive(self) -> None:
        cleaned = clean_text("Acute Respiratory Distress Syndrome (ARDS) in ICU patients.")
        for term in ("acute", "respiratory", "distress", "syndrome", "ards", "icu"):
            assert term in cleaned

    @pytest.mark.parametrize(
        "text",
        [
            "SARS-CoV-2 infection",
            "p < 0.001",
            "dose: 5mg/kg twice daily",
            "patients aged 18-65 years",
            "see https://trials.gov/NCT04871893",
        ],
    )
    def test_realistic_inputs_never_raise(self, text: str) -> None:
        assert isinstance(clean_text(text), str)

    def test_result_is_lowercase(self) -> None:
        assert clean_text("MECHANICAL Ventilation") == clean_text("MECHANICAL Ventilation").lower()
