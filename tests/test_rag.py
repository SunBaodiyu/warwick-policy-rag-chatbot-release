import json
import unittest
from unittest.mock import patch

from policy_rag.rag import answer_question
from policy_rag.semantic_indexer import Chunk


class RagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunk = Chunk(
            chunk_id="IMP02-6.1-0",
            document_id="IMP02",
            document_title=(
                "IMP 02: AI Information Compliance Policy"
            ),
            source_path="data/raw/IMP02.html",
            section="6.1",
            text=(
                "Passwords and usernames must never be put into AI "
                "software without prior approval."
            ),
            word_count=14,
            metadata={},
        )

    def make_generation_result(
        self,
        evidence: str,
        answer: str,
    ) -> dict:
        """Create a complete mocked Ollama response."""

        return {
            "text": json.dumps(
                {
                    "evidence": evidence,
                    "answer": answer,
                }
            ),
            "model": "qwen2.5:1.5b",
            "total_duration_ns": 1_000_000,
            "load_duration_ns": 100_000,
            "prompt_eval_count": 20,
            "prompt_eval_duration_ns": 200_000,
            "eval_count": 10,
            "eval_duration_ns": 500_000,
        }

    def answer_with_raw_generation(
        self,
        raw_text: str,
    ) -> dict:
        generation = self.make_generation_result(
            evidence="",
            answer="",
        )
        generation["text"] = raw_text

        with patch(
            "policy_rag.rag.search_semantic_index",
            return_value=[(0.91, self.chunk)],
        ), patch(
            "policy_rag.rag.generate_local",
            return_value=generation,
        ):
            return answer_question(
                "What does the synthetic policy require?"
            )

    def assert_structured_generation_failure(
        self,
        raw_text: str,
        expected_error: str,
    ) -> None:
        result = self.answer_with_raw_generation(
            raw_text
        )

        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(
            result["response_mode"],
            "generation_failure",
        )
        self.assertEqual(
            result["generation_error"],
            expected_error,
        )
        self.assertEqual(
            result["answer"],
            (
                "I cannot answer this question from "
                "the provided policies."
            ),
        )
        self.assertEqual(result["model_evidence"], "")
        self.assertEqual(result["model_answer"], "")
        self.assertFalse(result["evidence_valid"])
        self.assertFalse(result["answer_grounded"])

    @patch("policy_rag.rag.generate_local")
    @patch("policy_rag.rag.search_semantic_index")
    def test_supported_answer_includes_verified_citation(
        self,
        mock_search,
        mock_generate,
    ) -> None:
        mock_search.return_value = [(0.91, self.chunk)]
        mock_generate.return_value = (
            self.make_generation_result(
                evidence=self.chunk.text,
                answer=(
                    "No, passwords and usernames require "
                    "prior approval."
                ),
            )
        )

        result = answer_question(
            "Can passwords and usernames be entered "
            "without approval?"
        )

        self.assertEqual(
            result["status"],
            "supported",
        )
        self.assertIn(
            "[IMP02, Section 6.1]",
            result["answer"],
        )
        self.assertEqual(
            result["model_evidence"],
            self.chunk.text,
        )
        self.assertEqual(
            result["response_mode"],
            "generated",
        )
        self.assertEqual(result["generation_error"], "")
        self.assertEqual(
            result["sources"][0]["text"],
            self.chunk.text,
        )

    @patch("policy_rag.rag.generate_local")
    @patch("policy_rag.rag.search_semantic_index")
    def test_unsupported_answer_uses_standard_refusal(
        self,
        mock_search,
        mock_generate,
    ) -> None:
        mock_search.return_value = [(0.50, self.chunk)]
        mock_generate.return_value = (
            self.make_generation_result(
                evidence="",
                answer="",
            )
        )

        result = answer_question(
            "Which AI vendor is approved by the University?"
        )

        self.assertEqual(
            result["status"],
            "unsupported",
        )
        self.assertEqual(
            result["answer"],
            (
                "I cannot answer this question from "
                "the provided policies."
            ),
        )
        self.assertEqual(
            result["model_evidence"],
            "",
        )
        self.assertEqual(
            result["response_mode"],
            "refusal",
        )
        self.assertEqual(result["generation_error"], "")

    @patch("policy_rag.rag.generate_local")
    @patch("policy_rag.rag.search_semantic_index")
    def test_section_number_is_rejected_as_evidence(
        self,
        mock_search,
        mock_generate,
    ) -> None:
        mock_search.return_value = [(0.80, self.chunk)]
        mock_generate.return_value = (
            self.make_generation_result(
                evidence="Section 6.1",
                answer=(
                    "Students and staff may use any approved "
                    "device for educational purposes."
                ),
            )
        )

        result = answer_question(
            "What systems and resources are covered?"
        )

        self.assertEqual(
            result["status"],
            "unsupported",
        )
        self.assertEqual(
            result["answer"],
            (
                "I cannot answer this question from "
                "the provided policies."
            ),
        )

    @patch("policy_rag.rag.generate_local")
    @patch("policy_rag.rag.search_semantic_index")
    def test_ungrounded_answer_uses_extractive_fallback(
        self,
        mock_search,
        mock_generate,
    ) -> None:
        mock_search.return_value = [(0.85, self.chunk)]
        mock_generate.return_value = (
            self.make_generation_result(
                evidence=self.chunk.text,
                answer=(
                    "Students may use tablets for educational "
                    "purposes."
                ),
            )
        )

        result = answer_question(
            "What systems and resources are covered?"
        )

        self.assertEqual(
            result["status"],
            "supported",
        )
        self.assertEqual(
            result["response_mode"],
            "extractive_fallback",
        )
        self.assertIn(
            self.chunk.text,
            result["answer"],
        )
        self.assertNotIn(
            "tablets",
            result["answer"].lower(),
        )

    @patch("policy_rag.rag.generate_local")
    @patch("policy_rag.rag.search_semantic_index")
    def test_invalid_json_fails_closed(
        self,
        mock_search,
        mock_generate,
    ) -> None:
        mock_search.return_value = [(0.91, self.chunk)]

        invalid_generation = (
            self.make_generation_result(
                evidence="",
                answer="",
            )
        )
        invalid_generation["text"] = (
            '{"evidence": "incomplete"'
        )
        mock_generate.return_value = invalid_generation

        result = answer_question(
            "How often must systems be tested?"
        )

        self.assertEqual(
            result["status"],
            "unsupported",
        )
        self.assertEqual(
            result["response_mode"],
            "generation_failure",
        )
        self.assertEqual(
            result["generation_error"],
            "invalid_structured_json",
        )
        self.assertEqual(
            result["answer"],
            (
                "I cannot answer this question from "
                "the provided policies."
            ),
        )

    def test_invalid_structured_schema_fails_closed(
        self,
    ) -> None:
        cases = {
            "empty_object": "{}",
            "missing_answer": '{"evidence": ""}',
            "missing_evidence": '{"answer": ""}',
            "extra_field": (
                '{"evidence": "", "answer": "", '
                '"extra": true}'
            ),
            "answer_null": (
                '{"evidence": "", "answer": null}'
            ),
            "evidence_number": (
                '{"evidence": 1, "answer": ""}'
            ),
            "evidence_boolean": (
                '{"evidence": true, "answer": ""}'
            ),
            "answer_list": (
                '{"evidence": "", "answer": []}'
            ),
            "top_level_list": "[]",
            "top_level_null": "null",
            "top_level_number": "1",
            "top_level_boolean": "true",
            "top_level_string": '"text"',
        }

        for name, raw_text in cases.items():
            with self.subTest(name=name):
                self.assert_structured_generation_failure(
                    raw_text,
                    "invalid_structured_schema",
                )

    def test_duplicate_structured_keys_fail_closed(
        self,
    ) -> None:
        cases = {
            "duplicate_answer": (
                '{"evidence": "", "answer": "first", '
                '"answer": "second"}'
            ),
            "duplicate_evidence": (
                '{"evidence": "first", '
                '"evidence": "second", "answer": ""}'
            ),
        }

        for name, raw_text in cases.items():
            with self.subTest(name=name):
                self.assert_structured_generation_failure(
                    raw_text,
                    "invalid_structured_schema",
                )


if __name__ == "__main__":
    unittest.main()
