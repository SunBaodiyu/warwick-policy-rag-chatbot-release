import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from policy_rag.evaluation import (
    evaluate_retrieval,
    section_matches,
)
from policy_rag.models import Chunk


class EvaluationTests(unittest.TestCase):
    def test_section_matching_uses_complete_identifiers(self):
        cases = [
            ("7.1", "7.1", True),
            ("7.1", "7.1 Technical controls", True),
            (
                "7.1 Expected heading",
                "7.1 Actual heading",
                True,
            ),
            ("7.1", "7.10 Access review", False),
            ("7.10", "7.1 Technical controls", False),
            ("5", "5 Principles", True),
            ("5", "5.1 Detailed principles", False),
        ]

        for expected, actual, matches in cases:
            with self.subTest(
                expected=expected,
                actual=actual,
            ):
                self.assertEqual(
                    section_matches(expected, actual),
                    matches,
                )

    def test_non_numeric_sections_require_full_text_match(self):
        self.assertTrue(
            section_matches("Introduction", " introduction ")
        )
        self.assertTrue(
            section_matches("SCOPE", "scope")
        )
        self.assertFalse(
            section_matches(
                "Introduction",
                "Introduction and scope",
            )
        )

    def test_evaluation_calculates_rank_metrics(self):
        questions = [
            {
                "question_id": "IMP02-Q01",
                "policy_id": "IMP02",
                "question": "Who is responsible for AI tool outputs?",
                "expected_answer": "Users of AI tools are responsible.",
                "evidence_section": "7.1",
                "question_type": "direct",
                "answerable": True,
            }
        ]

        wrong_chunk = Chunk(
            chunk_id="wrong",
            document_id="imp02-policy",
            document_title="IMP02",
            source_path="test.html",
            section="7.10 Access review",
            text="Incorrect section",
            word_count=2,
            metadata={},
        )

        correct_chunk = Chunk(
            chunk_id="correct",
            document_id="imp02-policy",
            document_title="IMP02",
            source_path="test.html",
            section="7.1 Technical controls",
            text="Users are responsible for AI outputs.",
            word_count=6,
            metadata={},
        )

        fake_results = [
            (0.8, wrong_chunk),
            (0.7, correct_chunk),
        ]

        with TemporaryDirectory() as temporary_directory:
            question_path = Path(temporary_directory) / "questions.json"
            question_path.write_text(
                json.dumps(questions),
                encoding="utf-8",
            )

            with patch(
                "policy_rag.indexer.search_index",
                return_value=fake_results,
            ):
                report = evaluate_retrieval(
                    question_path,
                    "unused-index-directory",
                    top_k=3,
                )

        self.assertEqual(report["question_count"], 1)
        self.assertEqual(report["hit_at_1"], 0.0)
        self.assertEqual(report["hit_at_3"], 1.0)
        self.assertEqual(report["mrr"], 0.5)
        self.assertEqual(
            report["details"][0]["correct_rank"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
