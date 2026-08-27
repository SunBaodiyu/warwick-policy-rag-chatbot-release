import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from policy_rag.chunking import chunk_documents
from policy_rag.indexer import build_index, search_index
from policy_rag.loaders import load_document


class PipelineTests(unittest.TestCase):
    def test_build_and_search_returns_incident_clause(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_path = root / "account_policy.txt"
            policy_path.write_text(
                "1 Purpose\nThis is a test policy.\n\n"
                "2 Passwords\nUsers must keep passwords secret.\n\n"
                "3 Incident Reporting\nA compromised account must be reported to the "
                "IT Service Desk immediately.",
                encoding="utf-8",
            )

            document = load_document(policy_path)
            chunks = chunk_documents(
                [document], strategy="policy", max_words=20, overlap_words=3
            )
            index_dir = root / "index"
            build_index(chunks, index_dir, strategy="policy")

            results = search_index(
                "Where should a compromised account be reported?", index_dir
            )

            self.assertTrue(results)
            self.assertIn("IT Service Desk", results[0][1].text)
            self.assertGreater(results[0][0], 0)


if __name__ == "__main__":
    unittest.main()
