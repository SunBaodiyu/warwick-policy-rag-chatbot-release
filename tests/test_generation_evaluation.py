"""Tests for generation evaluation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import policy_rag.generation_evaluation as generation_evaluation
from policy_rag.generation_evaluation import (
    FINAL_MODELS,
    FROZEN_FINAL_QUESTION_PATH,
    FROZEN_FINAL_QUESTION_SHA256,
    _build_and_write_json_exclusive,
    _evaluate_question,
    _summarise_results,
    _write_json_exclusive,
    run_final_generation_evaluation,
    run_generation_evaluation,
)
from policy_rag.models import Chunk


class GenerationEvaluationTests(unittest.TestCase):
    @staticmethod
    def make_question(answerable: bool) -> dict:
        return {
            "question_id": "TEST-Q01",
            "policy_id": "IMP02",
            "question": "Test question?",
            "expected_answer": "Test answer.",
            "evidence_section": "1.1" if answerable else "",
            "question_type": (
                "direct" if answerable else "unanswerable"
            ),
            "answerable": answerable,
        }

    @staticmethod
    def make_summary_result(
        *,
        answerable: bool,
        actual_status: str,
        status_correct: bool,
        response_mode: str,
        generation_error: str = "",
    ) -> dict:
        return {
            "answerable": answerable,
            "actual_status": actual_status,
            "status_correct": status_correct,
            "source_policy_correct": False,
            "source_section_correct": False,
            "response_mode": response_mode,
            "generation_error": generation_error,
            "end_to_end_seconds": 1.0,
        }

    def test_summary_calculates_generation_metrics(
        self,
    ) -> None:
        results = [
            {
                "answerable": True,
                "actual_status": "supported",
                "status_correct": True,
                "source_policy_correct": True,
                "source_section_correct": True,
                "response_mode": "model",
                "end_to_end_seconds": 2.0,
            },
            {
                "answerable": False,
                "actual_status": "unsupported",
                "status_correct": True,
                "source_policy_correct": False,
                "source_section_correct": False,
                "response_mode": "refusal",
                "end_to_end_seconds": 4.0,
            },
        ]

        summary = _summarise_results(
            "test-model",
            results,
        )

        self.assertEqual(summary["question_count"], 2)
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["generation_failure_count"], 0)
        self.assertEqual(summary["generation_failure_rate"], 0.0)
        self.assertEqual(summary["status_accuracy"], 1.0)
        self.assertEqual(
            summary["answerable_support_rate"],
            1.0,
        )
        self.assertEqual(
            summary["unanswerable_refusal_rate"],
            1.0,
        )
        self.assertEqual(
            summary["mean_end_to_end_seconds"],
            3.0,
        )

    @patch("policy_rag.rag.answer_question")
    def test_answerable_generation_failure_is_recorded(
        self,
        mock_answer_question,
    ) -> None:
        mock_answer_question.return_value = {
            "status": "unsupported",
            "response_mode": "generation_failure",
            "generation_error": "invalid_structured_json",
            "actual_model": "test-model",
            "sources": [],
        }

        result = _evaluate_question(
            self.make_question(answerable=True),
            model="test-model",
            top_k=1,
        )
        summary = _summarise_results(
            "test-model",
            [result],
        )

        self.assertEqual(
            result["generation_error"],
            "invalid_structured_json",
        )
        self.assertFalse(result["status_correct"])
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["generation_failure_count"], 1)
        self.assertEqual(summary["generation_failure_rate"], 1.0)
        self.assertEqual(summary["status_accuracy"], 0.0)

    @patch("policy_rag.rag.answer_question")
    def test_generation_error_overrides_matching_status(
        self,
        mock_answer_question,
    ) -> None:
        mock_answer_question.return_value = {
            "status": "supported",
            "response_mode": "generated",
            "generation_error": "generation_validation_failed",
            "actual_model": "test-model",
            "sources": [],
        }

        result = _evaluate_question(
            self.make_question(answerable=True),
            model="test-model",
            top_k=1,
        )

        self.assertEqual(result["expected_status"], "supported")
        self.assertEqual(result["actual_status"], "supported")
        self.assertEqual(result["response_mode"], "generated")
        self.assertEqual(
            result["generation_error"],
            "generation_validation_failed",
        )
        self.assertFalse(result["status_correct"])

    @patch("policy_rag.rag.answer_question")
    def test_generation_source_section_uses_exact_match(
        self,
        mock_answer_question,
    ) -> None:
        cases = [
            ("7.1", "7.10 Access review", False),
            ("7.10", "7.1 Technical controls", False),
            ("7.1", "7.1 Technical controls", True),
            ("5", "5.1 Detailed principles", False),
            ("5", "5 Principles", True),
        ]

        for expected, actual, is_correct in cases:
            with self.subTest(
                expected=expected,
                actual=actual,
            ):
                question = self.make_question(
                    answerable=True
                )
                question["evidence_section"] = expected
                mock_answer_question.return_value = {
                    "status": "supported",
                    "response_mode": "generated",
                    "generation_error": "",
                    "actual_model": "test-model",
                    "sources": [
                        {
                            "policy": "IMP02",
                            "section": actual,
                        }
                    ],
                }

                result = _evaluate_question(
                    question,
                    model="test-model",
                    top_k=1,
                )
                summary = _summarise_results(
                    "test-model",
                    [result],
                )

                self.assertEqual(
                    result["source_section_correct"],
                    is_correct,
                )
                self.assertEqual(
                    summary["source_section_accuracy"],
                    1.0 if is_correct else 0.0,
                )

    @patch("policy_rag.rag.answer_question")
    def test_matching_actual_model_keeps_normal_metrics(
        self,
        mock_answer_question,
    ) -> None:
        mock_answer_question.return_value = {
            "status": "supported",
            "response_mode": "generated",
            "generation_error": "",
            "actual_model": "test-model",
            "answer": "Synthetic supported answer.",
            "sources": [
                {
                    "policy": "IMP02",
                    "section": "1.1 Synthetic section",
                }
            ],
        }

        result = _evaluate_question(
            self.make_question(answerable=True),
            model="test-model",
            top_k=1,
        )
        summary = _summarise_results(
            "test-model",
            [result],
        )

        self.assertEqual(result["actual_status"], "supported")
        self.assertTrue(result["status_correct"])
        self.assertEqual(result["response_mode"], "generated")
        self.assertEqual(result["error"], "")
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["generation_failure_count"], 0)
        self.assertEqual(summary["answerable_support_rate"], 1.0)
        self.assertEqual(summary["source_policy_accuracy"], 1.0)
        self.assertEqual(summary["source_section_accuracy"], 1.0)

    @patch("policy_rag.rag.answer_question")
    def test_mismatched_actual_model_invalidates_success_metrics(
        self,
        mock_answer_question,
    ) -> None:
        sources = [
            {
                "policy": "IMP02",
                "section": "1.1 Synthetic section",
            }
        ]
        mock_answer_question.return_value = {
            "status": "supported",
            "response_mode": "extractive_fallback",
            "generation_error": "",
            "actual_model": "different-model",
            "answer": "Preserved original answer.",
            "model_answer": "Preserved model answer.",
            "model_evidence": "Preserved model evidence.",
            "sources": sources,
        }

        result = _evaluate_question(
            self.make_question(answerable=True),
            model="test-model",
            top_k=1,
        )
        summary = _summarise_results(
            "test-model",
            [result],
        )

        self.assertEqual(result["actual_model"], "different-model")
        self.assertEqual(
            result["answer"],
            "Preserved original answer.",
        )
        self.assertEqual(result["sources"], sources)
        self.assertEqual(result["actual_status"], "error")
        self.assertFalse(result["status_correct"])
        self.assertEqual(result["response_mode"], "error")
        self.assertEqual(
            result["error"],
            "model_identity_mismatch",
        )
        self.assertFalse(result["source_policy_correct"])
        self.assertFalse(result["source_section_correct"])
        self.assertEqual(summary["model"], "test-model")
        self.assertEqual(summary["requested_model"], "test-model")
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["generation_failure_count"], 0)
        self.assertEqual(summary["status_accuracy"], 0.0)
        self.assertEqual(summary["answerable_support_rate"], 0.0)
        self.assertEqual(summary["source_policy_accuracy"], 0.0)
        self.assertEqual(summary["source_section_accuracy"], 0.0)
        self.assertEqual(summary["extractive_fallback_rate"], 0.0)

    @patch("policy_rag.rag.answer_question")
    def test_invalid_actual_model_is_identity_error(
        self,
        mock_answer_question,
    ) -> None:
        cases = (
            ("missing", False, None),
            ("empty", True, ""),
            ("null", True, None),
            ("number", True, 1),
        )

        for label, include_field, actual_model in cases:
            with self.subTest(label=label):
                response = {
                    "status": "supported",
                    "response_mode": "generated",
                    "generation_error": "",
                    "answer": "Preserved original answer.",
                    "sources": [],
                }

                if include_field:
                    response["actual_model"] = actual_model

                mock_answer_question.return_value = response
                result = _evaluate_question(
                    self.make_question(answerable=True),
                    model="test-model",
                    top_k=1,
                )
                summary = _summarise_results(
                    "test-model",
                    [result],
                )

                self.assertEqual(
                    result["actual_model"],
                    actual_model,
                )
                self.assertEqual(result["actual_status"], "error")
                self.assertFalse(result["status_correct"])
                self.assertEqual(result["response_mode"], "error")
                self.assertEqual(
                    result["error"],
                    "model_identity_mismatch",
                )
                self.assertEqual(summary["error_count"], 1)
                self.assertEqual(
                    summary["generation_failure_count"],
                    0,
                )

    @patch("policy_rag.rag.answer_question")
    def test_identity_error_is_not_correct_unanswerable_refusal(
        self,
        mock_answer_question,
    ) -> None:
        mock_answer_question.return_value = {
            "status": "unsupported",
            "response_mode": "refusal",
            "generation_error": "",
            "actual_model": "different-model",
            "answer": "Preserved refusal answer.",
            "sources": [],
        }

        result = _evaluate_question(
            self.make_question(answerable=False),
            model="test-model",
            top_k=1,
        )
        summary = _summarise_results(
            "test-model",
            [result],
        )

        self.assertEqual(result["actual_status"], "error")
        self.assertEqual(result["response_mode"], "error")
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["unanswerable_refusal_rate"], 0.0)
        self.assertEqual(summary["generation_failure_count"], 0)

    @patch("policy_rag.rag.answer_question")
    def test_identity_error_takes_priority_over_generation_failure(
        self,
        mock_answer_question,
    ) -> None:
        mock_answer_question.return_value = {
            "status": "unsupported",
            "response_mode": "generation_failure",
            "generation_error": "invalid_structured_json",
            "actual_model": "different-model",
            "answer": "Preserved failure answer.",
            "sources": [],
        }

        result = _evaluate_question(
            self.make_question(answerable=True),
            model="test-model",
            top_k=1,
        )
        summary = _summarise_results(
            "test-model",
            [result],
        )

        self.assertEqual(
            result["generation_error"],
            "invalid_structured_json",
        )
        self.assertEqual(result["actual_status"], "error")
        self.assertEqual(
            result["error"],
            "model_identity_mismatch",
        )
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["generation_failure_count"], 0)
        self.assertEqual(summary["generation_failure_rate"], 0.0)

    @patch("policy_rag.rag.answer_question")
    def test_unanswerable_generation_failure_is_not_refusal(
        self,
        mock_answer_question,
    ) -> None:
        mock_answer_question.return_value = {
            "status": "unsupported",
            "response_mode": "generation_failure",
            "generation_error": "invalid_structured_json",
            "actual_model": "test-model",
            "sources": [],
        }

        result = _evaluate_question(
            self.make_question(answerable=False),
            model="test-model",
            top_k=1,
        )
        summary = _summarise_results(
            "test-model",
            [result],
        )

        self.assertEqual(result["actual_status"], "unsupported")
        self.assertFalse(result["status_correct"])
        self.assertEqual(summary["generation_failure_count"], 1)
        self.assertEqual(summary["generation_failure_rate"], 1.0)
        self.assertEqual(summary["unanswerable_refusal_rate"], 0.0)
        self.assertEqual(summary["status_accuracy"], 0.0)

    def test_invalid_schema_from_rag_is_not_correct_refusal(
        self,
    ) -> None:
        chunk = Chunk(
            chunk_id="test-chunk",
            document_id="IMP02",
            document_title="Test policy",
            source_path="test.html",
            section="1.1 Test section",
            text=(
                "This synthetic policy sentence contains enough "
                "words for evidence validation."
            ),
            word_count=10,
            metadata={},
        )

        with patch(
            "policy_rag.rag.search_semantic_index",
            return_value=[(0.5, chunk)],
        ), patch(
            "policy_rag.rag.generate_local",
            return_value={
                "text": "{}",
                "model": "test-model",
                "actual_model": "test-model",
            },
        ):
            result = _evaluate_question(
                self.make_question(answerable=False),
                model="test-model",
                top_k=1,
            )

        summary = _summarise_results(
            "test-model",
            [result],
        )

        self.assertEqual(
            result["generation_error"],
            "invalid_structured_schema",
        )
        self.assertEqual(
            result["response_mode"],
            "generation_failure",
        )
        self.assertFalse(result["status_correct"])
        self.assertEqual(summary["generation_failure_count"], 1)
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["unanswerable_refusal_rate"], 0.0)
        self.assertEqual(summary["status_accuracy"], 0.0)

    def test_unanswerable_refusal_rate_counts_failures(
        self,
    ) -> None:
        results = [
            self.make_summary_result(
                answerable=False,
                actual_status="unsupported",
                status_correct=True,
                response_mode="refusal",
            ),
            self.make_summary_result(
                answerable=False,
                actual_status="unsupported",
                status_correct=False,
                response_mode="generation_failure",
                generation_error="invalid_structured_json",
            ),
        ]

        summary = _summarise_results(
            "test-model",
            results,
        )

        self.assertEqual(summary["unanswerable_refusal_rate"], 0.5)

    @patch(
        "policy_rag.rag.answer_question",
        side_effect=RuntimeError("Ollama unavailable"),
    )
    def test_runtime_error_is_not_generation_failure(
        self,
        mock_answer_question,
    ) -> None:
        result = _evaluate_question(
            self.make_question(answerable=True),
            model="test-model",
            top_k=1,
        )
        summary = _summarise_results(
            "test-model",
            [result],
        )

        self.assertEqual(result["actual_status"], "error")
        self.assertEqual(result["generation_error"], "")
        self.assertEqual(result["question_type"], "direct")
        self.assertEqual(result["requested_model"], "test-model")
        self.assertIsNone(result["actual_model"])
        self.assertIsNone(result["prompt_eval_duration_ns"])
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["generation_failure_count"], 0)
        self.assertEqual(summary["generation_failure_rate"], 0.0)

    def test_summary_separates_failure_types(
        self,
    ) -> None:
        results = [
            self.make_summary_result(
                answerable=True,
                actual_status="supported",
                status_correct=True,
                response_mode="generated",
            ),
            self.make_summary_result(
                answerable=True,
                actual_status="unsupported",
                status_correct=False,
                response_mode="generation_failure",
                generation_error="invalid_structured_json",
            ),
            self.make_summary_result(
                answerable=True,
                actual_status="error",
                status_correct=False,
                response_mode="error",
            ),
        ]

        summary = _summarise_results(
            "test-model",
            results,
        )

        self.assertEqual(summary["generation_failure_count"], 1)
        self.assertAlmostEqual(
            summary["generation_failure_rate"],
            1 / 3,
            places=4,
        )
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(
            summary["generation_failure_count"]
            + summary["error_count"],
            2,
        )

    @patch(
        "policy_rag.generation_evaluation."
        "_evaluate_question"
    )
    @patch(
        "policy_rag.generation_evaluation."
        "load_questions"
    )
    def test_evaluation_saves_json_report(
        self,
        mock_load_questions,
        mock_evaluate_question,
    ) -> None:
        mock_load_questions.return_value = [
            {
                "question_id": "TEST-Q01",
                "policy_id": "IMP02",
                "question": "Test question?",
                "expected_answer": "Test answer.",
                "evidence_section": "1.1",
                "question_type": "direct",
                "answerable": True,
            }
        ]
        mock_evaluate_question.return_value = {
            "answerable": True,
            "actual_status": "unsupported",
            "status_correct": False,
            "source_policy_correct": False,
            "source_section_correct": False,
            "response_mode": "generation_failure",
            "generation_error": "invalid_structured_json",
            "end_to_end_seconds": 1.0,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_generation_evaluation(
                "synthetic/questions.json",
                temp_dir,
                ("test-model",),
                1,
                1,
            )

            result_path = Path(
                report["result_path"]
            )

            self.assertTrue(result_path.is_file())
            saved_report = json.loads(
                result_path.read_text(encoding="utf-8")
            )
            saved_model_report = saved_report["models"][0]
            saved_summary = saved_model_report["summary"]
            saved_result = saved_model_report["results"][0]

            self.assertEqual(saved_report, report)
            self.assertTrue(
                {
                    "created_at_utc",
                    "question_path",
                    "top_k",
                    "models",
                    "result_path",
                }.issubset(saved_report)
            )
            self.assertEqual(
                saved_report["run_kind"],
                "development_generation",
            )
            self.assertEqual(
                saved_report["question_path"],
                "synthetic/questions.json",
            )

            self.assertEqual(
                saved_result["generation_error"],
                "invalid_structured_json",
            )
            self.assertEqual(
                saved_summary["generation_failure_count"],
                1,
            )
            self.assertEqual(
                saved_summary["generation_failure_rate"],
                1.0,
            )
            self.assertEqual(
                report["models"][0]["summary"][
                    "question_count"
                ],
                1,
            )


class FinalGenerationEvaluationTests(unittest.TestCase):
    @staticmethod
    def make_question() -> dict:
        return {
            "question_id": "SYNTHETIC-Q01",
            "policy_id": "IMP02",
            "question": "What does the synthetic rule require?",
            "expected_answer": "It requires the synthetic action.",
            "evidence_section": "1.1",
            "question_type": "direct",
            "answerable": True,
        }

    def test_frozen_final_identity_constants_are_locked(
        self,
    ) -> None:
        self.assertEqual(
            FROZEN_FINAL_QUESTION_PATH.as_posix(),
            "data/evaluation/final_questions.json",
        )
        self.assertEqual(
            FROZEN_FINAL_QUESTION_SHA256,
            (
                "ad8a474a0f5c19f4acba9d77dafeb903"
                "57cb208e2f5fe44463b4b34f60f4bd28"
            ),
        )
        self.assertEqual(
            FINAL_MODELS,
            (
                "qwen2.5:1.5b",
                "llama3.2:3b",
            ),
        )

    @classmethod
    def make_repository(
        cls,
        temp_dir: str,
        *,
        index_model: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
    ) -> tuple[Path, Path, Path, str, dict[str, bytes]]:
        root = Path(temp_dir)
        question_path = (
            root
            / "data"
            / "evaluation"
            / "final_questions.json"
        )
        question_path.parent.mkdir(parents=True)
        question_bytes = json.dumps(
            [cls.make_question()],
            indent=2,
        ).encode("utf-8")
        question_path.write_bytes(question_bytes)

        index_dir = root / "artifacts" / "semantic_index"
        index_dir.mkdir(parents=True)
        metadata = {
            "model_name": index_model,
            "document_count": 6,
            "chunk_count": 1,
        }
        index_files = {
            "metadata.json": json.dumps(
                metadata,
                sort_keys=True,
            ).encode("utf-8"),
            "chunks.json": b"[]",
            "embeddings.npy": b"synthetic semantic vectors",
        }

        for name, contents in index_files.items():
            (index_dir / name).write_bytes(contents)

        question_digest = hashlib.sha256(
            question_bytes
        ).hexdigest()
        return (
            root,
            question_path,
            index_dir,
            question_digest,
            index_files,
        )

    @staticmethod
    def clean_git() -> dict:
        return {
            "commit": "0123456789abcdef",
            "branch": "main",
            "clean": True,
            "errors": None,
        }

    @staticmethod
    def environment_identity() -> dict:
        return {
            "python_version": "3.11.synthetic",
            "python_implementation": "CPython",
            "platform": {
                "system": "Windows",
                "machine": "AMD64",
            },
            "platform_errors": None,
            "package_versions": {
                "pymupdf": "1.test",
                "scikit-learn": "2.test",
                "numpy": "3.test",
                "sentence-transformers": "4.test",
                "torch": "5.test",
                "transformers": "6.test",
                "streamlit": "7.test",
            },
            "package_errors": None,
        }

    @staticmethod
    def ollama_identity(
        names: tuple[str, ...] = FINAL_MODELS,
    ) -> dict:
        return {
            "version": "ollama version synthetic",
            "models": [
                {
                    "name": name,
                    "id": f"digest-{position}",
                    "size": f"{position}.0 GB",
                }
                for position, name in enumerate(
                    names,
                    start=1,
                )
            ],
            "errors": None,
        }

    def assert_final_guard_rejects(
        self,
        *,
        wrong_question_path: bool = False,
        expected_digest: str | None = None,
        git_identity: dict | None = None,
        top_k: object = 1,
        models: tuple[str, ...] = FINAL_MODELS,
        wrong_index_path: bool = False,
        index_model: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        ollama_names: tuple[str, ...] = FINAL_MODELS,
        ollama_result: dict | None = None,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            (
                root,
                question_path,
                index_dir,
                question_digest,
                _,
            ) = self.make_repository(
                temp_dir,
                index_model=index_model,
            )
            output_dir = root / "reports"

            if wrong_question_path:
                alternate = (
                    root
                    / "data"
                    / "evaluation"
                    / "other_questions.json"
                )
                alternate.write_bytes(
                    question_path.read_bytes()
                )
                question_path = alternate

            if wrong_index_path:
                alternate_index = root / "artifacts" / "other_index"
                alternate_index.mkdir(parents=True)
                index_dir = alternate_index

            frozen_digest = (
                question_digest
                if expected_digest is None
                else expected_digest
            )
            git_result = (
                self.clean_git()
                if git_identity is None
                else git_identity
            )
            ollama_fixture = (
                self.ollama_identity(ollama_names)
                if ollama_result is None
                else ollama_result
            )

            with patch.object(
                generation_evaluation,
                "REPOSITORY_ROOT",
                root,
            ), patch.object(
                generation_evaluation,
                "FROZEN_FINAL_QUESTION_SHA256",
                frozen_digest,
            ), patch.object(
                generation_evaluation,
                "collect_git_info",
                return_value=git_result,
            ), patch.object(
                generation_evaluation,
                "collect_python_environment",
                return_value=self.environment_identity(),
            ), patch.object(
                generation_evaluation,
                "collect_ollama_identity",
                return_value=ollama_fixture,
            ), patch(
                "policy_rag.rag.answer_question"
            ) as mock_answer_question:
                with self.assertRaises(
                    (ValueError, RuntimeError)
                ):
                    run_final_generation_evaluation(
                        question_path=question_path,
                        output_dir=output_dir,
                        models=models,
                        top_k=top_k,
                        index_dir=index_dir,
                    )

            mock_answer_question.assert_not_called()
            self.assertFalse(
                output_dir.exists()
                and any(output_dir.glob("*.json"))
            )

    def test_final_guard_rejects_wrong_question_hash(
        self,
    ) -> None:
        self.assert_final_guard_rejects(
            expected_digest="0" * 64,
        )

    def test_final_guard_rejects_wrong_question_path(
        self,
    ) -> None:
        self.assert_final_guard_rejects(
            wrong_question_path=True,
        )

    def test_final_guard_rejects_dirty_worktree(
        self,
    ) -> None:
        self.assert_final_guard_rejects(
            git_identity={
                "commit": "0123456789abcdef",
                "branch": "main",
                "clean": False,
                "errors": None,
            },
        )

    def test_final_guard_rejects_non_one_top_k(
        self,
    ) -> None:
        for top_k in (3, True):
            with self.subTest(top_k=top_k):
                self.assert_final_guard_rejects(
                    top_k=top_k,
                )

    def test_final_guard_rejects_incorrect_model_list(
        self,
    ) -> None:
        cases = (
            ("llama3.2:3b",),
            tuple(reversed(FINAL_MODELS)),
            FINAL_MODELS + ("extra-model",),
        )

        for models in cases:
            with self.subTest(models=models):
                self.assert_final_guard_rejects(
                    models=models,
                )

    def test_final_guard_rejects_wrong_index_path(
        self,
    ) -> None:
        self.assert_final_guard_rejects(
            wrong_index_path=True,
        )

    def test_final_guard_rejects_non_semantic_index_metadata(
        self,
    ) -> None:
        self.assert_final_guard_rejects(
            index_model="synthetic-tfidf-index",
        )

    def test_final_guard_rejects_unverified_ollama_models(
        self,
    ) -> None:
        self.assert_final_guard_rejects(
            ollama_names=("llama3.2:3b",),
        )

    def test_final_guard_requires_ollama_version_and_model_ids(
        self,
    ) -> None:
        missing_version = self.ollama_identity()
        missing_version["version"] = None
        missing_id = self.ollama_identity()
        missing_id["models"][0]["id"] = ""
        command_error = self.ollama_identity()
        command_error["errors"] = {
            "list": "command_failed_exit_1"
        }

        for ollama_result in (
            missing_version,
            missing_id,
            command_error,
        ):
            with self.subTest(ollama_result=ollama_result):
                self.assert_final_guard_rejects(
                    ollama_result=ollama_result,
                )

    def test_final_report_persists_identity_and_new_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            (
                root,
                question_path,
                index_dir,
                question_digest,
                index_files,
            ) = self.make_repository(temp_dir)
            output_dir = root / "reports"
            git_identity = self.clean_git()
            environment = self.environment_identity()
            ollama = self.ollama_identity()
            created_at = datetime(
                2026,
                8,
                5,
                12,
                34,
                56,
                123,
                tzinfo=timezone.utc,
            )

            def synthetic_answer(
                question,
                *,
                model,
                top_k,
                index_dir,
            ):
                self.assertEqual(top_k, 1)
                self.assertEqual(
                    Path(index_dir),
                    index_dir_fixture,
                )
                return {
                    "status": "supported",
                    "answer": "Synthetic supported answer.",
                    "response_mode": "generated",
                    "generation_error": "",
                    "model_evidence": (
                        "A complete synthetic evidence sentence."
                    ),
                    "model_answer": "Synthetic supported answer.",
                    "evidence_valid": True,
                    "answer_grounded": True,
                    "grounding_coverage": 1.0,
                    "model": model,
                    "actual_model": model,
                    "sources": [
                        {
                            "policy": "IMP02",
                            "section": "1.1 Synthetic section",
                        }
                    ],
                    "generation_metrics": {
                        "total_duration_ns": 900,
                        "load_duration_ns": 100,
                        "prompt_eval_count": 20,
                        "prompt_eval_duration_ns": 200,
                        "eval_count": 10,
                        "eval_duration_ns": 500,
                    },
                }

            index_dir_fixture = index_dir.resolve()

            with patch.object(
                generation_evaluation,
                "REPOSITORY_ROOT",
                root,
            ), patch.object(
                generation_evaluation,
                "FROZEN_FINAL_QUESTION_SHA256",
                question_digest,
            ), patch.object(
                generation_evaluation,
                "collect_git_info",
                return_value=git_identity,
            ), patch.object(
                generation_evaluation,
                "collect_python_environment",
                return_value=environment,
            ), patch.object(
                generation_evaluation,
                "collect_ollama_identity",
                return_value=ollama,
            ), patch.object(
                generation_evaluation,
                "_new_run_identity",
                return_value=(created_at, "run123"),
            ), patch(
                "policy_rag.rag.answer_question",
                side_effect=synthetic_answer,
            ) as mock_answer_question:
                report = run_final_generation_evaluation(
                    question_path=question_path,
                    output_dir=output_dir,
                    models=FINAL_MODELS,
                    top_k=1,
                    index_dir=index_dir,
                )

            self.assertEqual(
                mock_answer_question.call_count,
                2,
            )
            result_path = Path(report["result_path"])
            self.assertEqual(
                result_path.name,
                (
                    "final_generation_"
                    "20260805T123456000123Z_run123.json"
                ),
            )
            saved_report = json.loads(
                result_path.read_text(encoding="utf-8")
            )
            self.assertEqual(saved_report, report)
            self.assertEqual(
                saved_report["run_kind"],
                "final_generation",
            )
            self.assertEqual(
                saved_report["created_at_utc"],
                "2026-08-05T12:34:56.000123+00:00",
            )
            self.assertEqual(
                saved_report["question_path"],
                "data/evaluation/final_questions.json",
            )
            self.assertEqual(
                saved_report["question_sha256"],
                hashlib.sha256(
                    question_path.read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(saved_report["git"], git_identity)
            self.assertEqual(
                saved_report["environment"],
                environment,
            )
            self.assertEqual(saved_report["ollama"], ollama)
            self.assertEqual(
                saved_report["semantic_index"]["path"],
                "artifacts/semantic_index",
            )

            for name, contents in index_files.items():
                self.assertEqual(
                    saved_report["semantic_index"][
                        "files"
                    ][name],
                    hashlib.sha256(contents).hexdigest(),
                )

            self.assertTrue(
                {
                    "created_at_utc",
                    "question_path",
                    "top_k",
                    "models",
                    "result_path",
                }.issubset(saved_report)
            )

            for model_report, requested_model in zip(
                saved_report["models"],
                FINAL_MODELS,
            ):
                self.assertTrue(
                    {"model", "summary", "results"}.issubset(
                        model_report
                    )
                )
                result = model_report["results"][0]
                self.assertEqual(
                    result["question_type"],
                    "direct",
                )
                self.assertEqual(
                    result["requested_model"],
                    requested_model,
                )
                self.assertEqual(
                    result["actual_model"],
                    requested_model,
                )
                self.assertEqual(
                    result["prompt_eval_duration_ns"],
                    200,
                )
                self.assertEqual(
                    result["generation_metrics"][
                        "prompt_eval_duration_ns"
                    ],
                    200,
                )
                self.assertTrue(
                    {
                        "question_id",
                        "expected_answer",
                        "expected_status",
                        "actual_status",
                        "status_correct",
                        "source_policy_correct",
                        "source_section_correct",
                        "answer",
                        "response_mode",
                        "sources",
                        "end_to_end_seconds",
                    }.issubset(result)
                )

    def test_prompt_eval_duration_flows_through_rag_result(
        self,
    ) -> None:
        evidence = (
            "This synthetic policy evidence directly answers "
            "the synthetic question without outside knowledge."
        )
        chunk = Chunk(
            chunk_id="synthetic-chunk",
            document_id="IMP02",
            document_title="Synthetic policy",
            source_path="synthetic.html",
            section="1.1 Synthetic section",
            text=evidence,
            word_count=11,
            metadata={},
        )
        generation = {
            "text": json.dumps(
                {
                    "evidence": evidence,
                    "answer": evidence,
                }
            ),
            "model": "requested-test-model",
            "actual_model": "requested-test-model",
            "prompt_eval_duration_ns": 321,
        }

        with patch(
            "policy_rag.rag.search_semantic_index",
            return_value=[(0.9, chunk)],
        ), patch(
            "policy_rag.rag.generate_local",
            return_value=generation,
        ):
            result = _evaluate_question(
                self.make_question(),
                model="requested-test-model",
                top_k=1,
            )

        self.assertEqual(
            result["actual_model"],
            "requested-test-model",
        )
        self.assertEqual(
            result["prompt_eval_duration_ns"],
            321,
        )
        self.assertEqual(
            result["generation_metrics"][
                "prompt_eval_duration_ns"
            ],
            321,
        )

    def test_final_output_collision_stops_before_answering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            (
                root,
                question_path,
                index_dir,
                question_digest,
                _,
            ) = self.make_repository(temp_dir)
            output_dir = root / "reports"
            output_dir.mkdir()
            created_at = datetime(
                2026,
                8,
                5,
                13,
                0,
                0,
                456,
                tzinfo=timezone.utc,
            )
            result_path = output_dir / (
                "final_generation_"
                "20260805T130000000456Z_collision.json"
            )
            historical_bytes = b'{"historical": true}'
            result_path.write_bytes(historical_bytes)

            with patch.object(
                generation_evaluation,
                "REPOSITORY_ROOT",
                root,
            ), patch.object(
                generation_evaluation,
                "FROZEN_FINAL_QUESTION_SHA256",
                question_digest,
            ), patch.object(
                generation_evaluation,
                "collect_git_info",
                return_value=self.clean_git(),
            ), patch.object(
                generation_evaluation,
                "collect_python_environment",
                return_value=self.environment_identity(),
            ), patch.object(
                generation_evaluation,
                "collect_ollama_identity",
                return_value=self.ollama_identity(),
            ), patch.object(
                generation_evaluation,
                "_new_run_identity",
                return_value=(created_at, "collision"),
            ), patch(
                "policy_rag.rag.answer_question"
            ) as mock_answer_question:
                with self.assertRaises(FileExistsError):
                    run_final_generation_evaluation(
                        question_path=question_path,
                        output_dir=output_dir,
                        index_dir=index_dir,
                    )

            mock_answer_question.assert_not_called()
            self.assertEqual(
                result_path.read_bytes(),
                historical_bytes,
            )

    @patch(
        "policy_rag.generation_evaluation.load_questions",
        return_value=[make_question.__func__()],
    )
    def test_development_output_collision_stops_before_answering(
        self,
        mock_load_questions,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            created_at = datetime(
                2026,
                8,
                5,
                14,
                0,
                0,
                789,
                tzinfo=timezone.utc,
            )
            result_path = output_dir / (
                "generation_evaluation_"
                "20260805T140000000789Z_collision.json"
            )
            historical_bytes = b'{"historical": true}'
            result_path.write_bytes(historical_bytes)

            with patch.object(
                generation_evaluation,
                "_new_run_identity",
                return_value=(created_at, "collision"),
            ), patch(
                "policy_rag.rag.answer_question"
            ) as mock_answer_question:
                with self.assertRaises(FileExistsError):
                    run_generation_evaluation(
                        question_path="synthetic/questions.json",
                        output_dir=output_dir,
                        models=("test-model",),
                        top_k=1,
                    )

            mock_load_questions.assert_called_once_with(
                "synthetic/questions.json"
            )
            mock_answer_question.assert_not_called()
            self.assertEqual(
                result_path.read_bytes(),
                historical_bytes,
            )

    def test_exclusive_json_write_refuses_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "existing.json"
            original = b'{"historical": true}'
            result_path.write_bytes(original)

            with self.assertRaises(FileExistsError):
                _write_json_exclusive(
                    result_path,
                    {"replacement": True},
                )

            self.assertEqual(
                result_path.read_bytes(),
                original,
            )

    def test_exclusive_json_write_cleans_failed_new_report(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "failed.json"

            def fail_report() -> dict:
                raise RuntimeError("synthetic evaluation failure")

            with self.assertRaisesRegex(
                RuntimeError,
                "synthetic evaluation failure",
            ):
                _build_and_write_json_exclusive(
                    result_path,
                    fail_report,
                )

            self.assertFalse(result_path.exists())


if __name__ == "__main__":
    unittest.main()
