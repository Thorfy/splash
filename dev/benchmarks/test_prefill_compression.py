from __future__ import annotations

import unittest
from dev.benchmarks.prefill_compression import chunks, extractive

class ExtractiveCompressionTests(unittest.TestCase):
    def setUp(self):
        self.context = '\n\n'.join([
            'Project overview: this document describes a distributed service.',
            'Build notes: install the package and configure the logging output.',
            'Database migration: the production database timeout is 45 seconds.',
            'Deployment notes: blue-green deployment uses a 10 minute drain.',
            'Troubleshooting: retry the database connection after a timeout.',
        ])

    def test_splits_paragraphs(self):
        self.assertEqual(len(chunks(self.context)), 5)

    def test_keeps_relevant_verbatim_chunk(self):
        compressed = extractive(self.context, 'What is the production database timeout?', 0.65)
        self.assertIn('production database timeout is 45 seconds', compressed)
        self.assertNotIn('blue-green deployment', compressed)

    def test_compression_reduces_context_size(self):
        compressed = extractive(self.context, 'What is the production database timeout?', 0.50)
        self.assertLess(len(compressed), len(self.context))

    def test_rejects_invalid_ratio(self):
        with self.assertRaises(ValueError):
            extractive(self.context, 'database timeout', 1.0)

    def test_empty_text_is_safe(self):
        self.assertEqual(extractive('', 'a question', 0.5), '')

if __name__ == '__main__':
    unittest.main()
