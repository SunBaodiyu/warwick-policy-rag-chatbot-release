import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from policy_rag.models import Chunk
from policy_rag.semantic_indexer import search_semantic_index


class FakeModel:
    """Return a fixed query vector without loading a real model."""

    def encode(self, texts, **kwargs):
        return np.array([[0.0, 1.0]], dtype=np.float32)


class SemanticIndexerTests(unittest.TestCase):
    def test_semantic_search_ranks_most_similar_chunk_first(self):
        first_chunk = Chunk(
            chunk_id="chunk-a",
            document_id="imp02",
            document_title="IMP02",
            source_path="imp02.html",
            section="1.1",
            text="AI policy introduction",
            word_count=3,
            metadata={},
        )

        second_chunk = Chunk(
            chunk_id="chunk-b",
            document_id="imp06",
            document_title="IMP06",
            source_path="imp06.html",
            section="6.1",
            text="Report security incidents",
            word_count=3,
            metadata={},
        )

        with TemporaryDirectory() as temporary_directory:
            index_dir = Path(temporary_directory)

            (index_dir / "chunks.json").write_text(
                json.dumps(
                    [
                        asdict(first_chunk),
                        asdict(second_chunk),
                    ]
                ),
                encoding="utf-8",
            )

            np.save(
                index_dir / "embeddings.npy",
                np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )

            (index_dir / "metadata.json").write_text(
                json.dumps({"model_name": "fake-model"}),
                encoding="utf-8",
            )

            with patch(
                "policy_rag.semantic_indexer._load_local_model",
                return_value=FakeModel(),
            ):
                results = search_semantic_index(
                    "What should happen after an incident?",
                    index_dir,
                    top_k=2,
                )

        self.assertEqual(results[0][1].chunk_id, "chunk-b")
        self.assertAlmostEqual(results[0][0], 1.0)


if __name__ == "__main__":
    unittest.main()