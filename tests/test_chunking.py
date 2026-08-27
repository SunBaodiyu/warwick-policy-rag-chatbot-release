import unittest

from policy_rag.chunking import chunk_fixed, chunk_policy_aware
from policy_rag.models import Document


def make_document() -> Document:
    return Document(
        document_id="test-policy",
        title="Test Policy",
        source_path="test_policy.txt",
        text=(
            "1 Purpose\nThis policy explains the test.\n\n"
            "2 Scope\nThis policy applies to all users.\n\n"
            "3 Security\nUsers must protect passwords.\n\n"
            "3.1 Incident Reporting\nUsers must report a compromised account."
        ),
    )


class ChunkingTests(unittest.TestCase):
    def test_policy_aware_chunking_preserves_section_metadata(self) -> None:
        chunks = chunk_policy_aware(make_document(), max_words=18, overlap_words=3)

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.metadata["strategy"] == "policy" for chunk in chunks))
        self.assertTrue(any("Incident Reporting" in chunk.section for chunk in chunks))
        self.assertTrue(all(chunk.word_count <= 18 for chunk in chunks))

    def test_fixed_chunking_uses_overlap_and_size_limit(self) -> None:
        chunks = chunk_fixed(make_document(), max_words=10, overlap_words=2)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.word_count <= 10 for chunk in chunks))
        self.assertTrue(all(chunk.section == "Fixed window" for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
