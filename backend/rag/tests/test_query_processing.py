"""Tests for query processing.

M3 tests verify deterministic, local query normalization:
- Whitespace trimming
- Repeated whitespace normalization
- Empty/None query rejection
- Original query preservation
- Deterministic output
"""

from django.test import SimpleTestCase

from rag.exceptions import QueryProcessingError
from rag.query_processing import ProcessedQuery, process_query


class ProcessQueryTests(SimpleTestCase):
    """Tests for process_query function."""

    def test_basic_query(self):
        result = process_query("What is Python?")
        self.assertIsInstance(result, ProcessedQuery)
        self.assertEqual(result.original, "What is Python?")
        self.assertEqual(result.normalized, "What is Python?")

    def test_trims_surrounding_whitespace(self):
        result = process_query("  hello world  ")
        self.assertEqual(result.normalized, "hello world")
        self.assertEqual(result.original, "  hello world  ")

    def test_normalizes_repeated_whitespace(self):
        result = process_query("hello    world    test")
        self.assertEqual(result.normalized, "hello world test")

    def test_normalizes_tabs_and_newlines(self):
        result = process_query("hello\t\tworld\n\ntest")
        self.assertEqual(result.normalized, "hello world test")

    def test_combined_whitespace_normalization(self):
        result = process_query("  hello   \t  world  \n  test  ")
        self.assertEqual(result.normalized, "hello world test")
        self.assertEqual(result.original, "  hello   \t  world  \n  test  ")

    def test_empty_string_raises(self):
        with self.assertRaises(QueryProcessingError):
            process_query("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(QueryProcessingError):
            process_query("   ")

    def test_tabs_only_raises(self):
        with self.assertRaises(QueryProcessingError):
            process_query("\t\t")

    def test_newlines_only_raises(self):
        with self.assertRaises(QueryProcessingError):
            process_query("\n\n")

    def test_none_raises(self):
        with self.assertRaises(QueryProcessingError):
            process_query(None)

    def test_single_word(self):
        result = process_query("Python")
        self.assertEqual(result.normalized, "Python")
        self.assertEqual(result.original, "Python")

    def test_preserves_original_query(self):
        """Original query is always preserved unmodified."""
        original = "  lots   of   spaces  "
        result = process_query(original)
        self.assertEqual(result.original, original)

    def test_deterministic_output(self):
        """Same input always produces same output."""
        r1 = process_query("  hello   world  ")
        r2 = process_query("  hello   world  ")
        self.assertEqual(r1.original, r2.original)
        self.assertEqual(r1.normalized, r2.normalized)

    def test_preserves_case(self):
        result = process_query("Hello World TEST")
        self.assertEqual(result.normalized, "Hello World TEST")

    def test_preserves_punctuation(self):
        result = process_query("What is Python?!!")
        self.assertEqual(result.normalized, "What is Python?!!")

    def test_unicode_preserved(self):
        result = process_query("  café résumé  ")
        self.assertEqual(result.normalized, "café résumé")
