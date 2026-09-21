"""Tests for text chunking."""

from django.test import TestCase

from rag.chunking import ChunkResult, ChunkingConfig, chunk_text
from rag.exceptions import ChunkingError


class ChunkingConfigTests(TestCase):
    """Tests for ChunkingConfig validation."""

    def test_default_config(self):
        config = ChunkingConfig()
        self.assertEqual(config.chunk_size, 512)
        self.assertEqual(config.chunk_overlap, 50)

    def test_custom_config(self):
        config = ChunkingConfig(chunk_size=100, chunk_overlap=10)
        self.assertEqual(config.chunk_size, 100)
        self.assertEqual(config.chunk_overlap, 10)

    def test_zero_chunk_size_raises(self):
        with self.assertRaises(ChunkingError):
            ChunkingConfig(chunk_size=0)

    def test_negative_chunk_size_raises(self):
        with self.assertRaises(ChunkingError):
            ChunkingConfig(chunk_size=-1)

    def test_negative_overlap_raises(self):
        with self.assertRaises(ChunkingError):
            ChunkingConfig(chunk_overlap=-1)

    def test_overlap_equal_to_size_raises(self):
        with self.assertRaises(ChunkingError):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_size_raises(self):
        with self.assertRaises(ChunkingError):
            ChunkingConfig(chunk_size=100, chunk_overlap=150)


class ChunkTextTests(TestCase):
    """Tests for chunk_text function."""

    def test_empty_string_returns_empty(self):
        result = chunk_text("")
        self.assertEqual(result, [])

    def test_whitespace_only_returns_empty(self):
        result = chunk_text("   \n\t  ")
        self.assertEqual(result, [])

    def test_none_config_uses_defaults(self):
        text = "Hello world"
        result = chunk_text(text)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_short_text_single_chunk(self):
        text = "This is a short text."
        config = ChunkingConfig(chunk_size=100, chunk_overlap=10)
        result = chunk_text(text, config)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, text)
        self.assertEqual(result[0].chunk_index, 0)
        self.assertEqual(result[0].start_offset, 0)
        self.assertEqual(result[0].end_offset, len(text))

    def test_exact_chunk_size(self):
        text = "a" * 100
        config = ChunkingConfig(chunk_size=100, chunk_overlap=10)
        result = chunk_text(text, config)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, text)

    def test_multiple_chunks_with_overlap(self):
        text = "a" * 200
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        result = chunk_text(text, config)
        # step = 100 - 20 = 80, so: 0-100, 80-180, 160-200
        self.assertEqual(len(result), 3)

        # Verify offsets
        self.assertEqual(result[0].start_offset, 0)
        self.assertEqual(result[0].end_offset, 100)
        self.assertEqual(result[1].start_offset, 80)
        self.assertEqual(result[1].end_offset, 180)
        self.assertEqual(result[2].start_offset, 160)
        self.assertEqual(result[2].end_offset, 200)

    def test_chunk_indexes_are_sequential(self):
        text = "a" * 500
        config = ChunkingConfig(chunk_size=100, chunk_overlap=10)
        result = chunk_text(text, config)
        for i, chunk in enumerate(result):
            self.assertEqual(chunk.chunk_index, i)

    def test_chunks_cover_full_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 20
        config = ChunkingConfig(chunk_size=50, chunk_overlap=5)
        result = chunk_text(text, config)
        # Every character should be covered by at least one chunk
        covered = set()
        for chunk in result:
            for i in range(chunk.start_offset, chunk.end_offset):
                covered.add(i)
        non_whitespace = {i for i, c in enumerate(text) if not c.isspace()}
        # All non-whitespace characters that are in meaningful chunks should be covered
        self.assertTrue(non_whitespace.issubset(covered))

    def test_zero_overlap_produces_non_overlapping_chunks(self):
        text = "a" * 200
        config = ChunkingConfig(chunk_size=100, chunk_overlap=0)
        result = chunk_text(text, config)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].start_offset, 0)
        self.assertEqual(result[0].end_offset, 100)
        self.assertEqual(result[1].start_offset, 100)
        self.assertEqual(result[1].end_offset, 200)

    def test_chunk_result_is_frozen_dataclass(self):
        chunk = ChunkResult(content="test", chunk_index=0, start_offset=0, end_offset=4)
        with self.assertRaises(AttributeError):
            chunk.content = "modified"

    def test_deterministic_output(self):
        text = "Hello world, this is a test document for chunking."
        config = ChunkingConfig(chunk_size=20, chunk_overlap=5)
        result1 = chunk_text(text, config)
        result2 = chunk_text(text, config)
        self.assertEqual(len(result1), len(result2))
        for c1, c2 in zip(result1, result2):
            self.assertEqual(c1.content, c2.content)
            self.assertEqual(c1.start_offset, c2.start_offset)
            self.assertEqual(c1.end_offset, c2.end_offset)
