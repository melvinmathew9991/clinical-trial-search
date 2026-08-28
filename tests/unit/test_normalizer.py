"""Text normalisation primitives and the full cleaning chain."""

from __future__ import annotations

import pytest

from medsearch.preprocessing.normalizer import (
    clean_text,
    collapse_whitespace,
    strip_digits,
    strip_punctuation,
    strip_urls_and_symbols,
    to_lowercase,
)


class TestPrimitives:
    def test_lowercase(self) -> None:
        assert to_lowercase("COVID-19 Patients") == "covid-19 patients"

    def test_strip_digits(self) -> None:
        assert strip_digits("1200 patients aged 65") == " patients aged "

    def test_strip_punctuation(self) -> None:
        assert strip_punctuation("p<0.05, (n=12)") == "p005 n12"

    def test_strip_urls(self) -> None:
        assert "http" not in strip_urls_and_symbols("see http://example.com/x for detail")

    def test_strip_handles(self) -> None:
        assert "@author" not in strip_urls_and_symbols("reported by @author today")

    def test_collapse_whitespace(self) -> None:
        assert collapse_whitespace("a  \n b \t c") == "a b c"

    def test_collapse_whitespace_trims(self) -> None:
        assert collapse_whitespace("  padded  ") == "padded"


class TestCleanText:
    def test_full_chain(self) -> None:
        assert clean_text("COVID-19 affects 1,200 patients (see http://x.co)") == (
            "covid affects patients see"
        )

    def test_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_whitespace_only(self) -> None:
        assert clean_text("   \n\t  ") == ""

    def test_digits_only(self) -> None:
        assert clean_text("12345") == ""

    def test_punctuation_only(self) -> None:
        assert clean_text("!!!???...") == ""

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
